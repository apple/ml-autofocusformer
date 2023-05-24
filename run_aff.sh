# number of gpus for data parallel
GPUS=2

# dataset path
DATA=imagenet/

# config file path
CONFIG=configs/aff_small.yaml

# checkpoint path for resume
RESUME=checkpoints/aff_small.pth

python -m torch.distributed.launch --nproc_per_node $GPUS --master_port 12345 main.py \
    --data-path $DATA \
    --cfg $CONFIG \
    --eval \
    --resume $RESUME \

# Comment out '--eval' and '--resume' to start training from fresh.
# To enlarge the effective batch size, use '--accumulation-steps'. For example, '--accumulation-steps 2' doubles the effective total batch size.
