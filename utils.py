# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Adapted for AutoFocusFormer by Ziwen 2023

import builtins
import datetime
import os
import torch
import numpy as np
import random
import torch.distributed as dist
import copy


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger, use_ema=False):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    if use_ema:
        msg = model.load_state_dict(checkpoint['model_ema'], strict=False)
        logger.info(msg)
        del checkpoint
        torch.cuda.empty_cache()
        return
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
        if 'rng' in checkpoint:
            np.random.set_state(checkpoint['np_rng'])
            torch.set_rng_state(checkpoint['rng'])
            torch.random.set_rng_state(checkpoint['random'])
            random.setstate(checkpoint['prng'])

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy

def load_pretrained(config, model, logger, use_ema=False):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')

    if use_ema:
        state_dict = checkpoint['model_ema']
        hw_name = 'module.head.weight'
        hb_name = 'module.head.bias'
    else:
        state_dict = checkpoint['model']
        hw_name = 'head.weight'
        hb_name = 'head.bias'

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict[hb_name]
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0] if not use_ema else model.module.head.bias.shape[0]
    if (Nc1 != Nc2):
        #if Nc1 == 21841 and Nc2 == 1000:
        if Nc1 == 10450 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map1kto21kp = json.load(open("data/map22kpto1k.txt"))
            invalid_cls = (torch.Tensor(map1kto21kp) == -1).nonzero()
            #print("invalid cls",invalid_cls)
            head_w_mapped = copy.deepcopy(state_dict[hw_name][map1kto21kp, :])
            head_b_mapped = copy.deepcopy(state_dict[hb_name][map1kto21kp])
            del state_dict[hw_name]
            del state_dict[hb_name]
            state_dict[hw_name] = head_w_mapped
            state_dict[hb_name] = head_b_mapped
            state_dict[hw_name][invalid_cls] = 0
            state_dict[hb_name][invalid_cls] = 0
        else:
            torch.nn.init.constant_(model.head.bias if not use_ema else model.module.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight if not use_ema else model.module.head.weight, 0.)
            del state_dict[hw_name]
            del state_dict[hb_name]
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()

def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema=None, total_epochs=None):
    if total_epochs is None:
        total_epochs = config.TRAIN.EPOCHS
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'loss_scaler': loss_scaler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'rng': torch.get_rng_state(),
                  'random': torch.random.get_rng_state(),
                  'np_rng': np.random.get_state(),
                  'prng': random.getstate()}
    if model_ema is not None:
        save_state['model_ema'] = model_ema.state_dict()

    save_path = os.path.join(config.OUTPUT, 'ckpt_epoch.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    if ((epoch+1) % config.SAVE_FREQ == 0 or epoch == (total_epochs - 1) or epoch == 0):
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
        torch.save(save_state, save_path)


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class NativeScalerWithGradNormCount:
    def __init__(self, config):
        self._scaler = torch.cuda.amp.GradScaler(enabled=config.AMP_ENABLE)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):

        self._scaler.scale(loss).backward(create_graph=create_graph)

        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad, error_if_nonfinite=False)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

    def is_enabled(self):
        return self._scaler.is_enabled()
