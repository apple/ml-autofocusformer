#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from torch.autograd import Function

try:
    import clustenqk_cuda
    import clustenav_cuda
    import clustenwf_cuda
except ImportError:
    raise RuntimeError("Could not load CLUSTEN CUDA extension. " +
                       "Please make sure your device has CUDA, the CUDA toolkit for PyTorch is installed, and that you've compiled CLUSTEN correctly.")


class CLUSTENQKFunction(Function):
    """
    query times key function
    """
    @staticmethod
    def forward(ctx, query, key, nbhd_idx):
        query = query.contiguous()
        key = key.contiguous()
        if key.dtype != query.dtype:
            key = key.to(query.dtype)
        nbhd_idx = nbhd_idx.contiguous()
        attn = clustenqk_cuda.forward(
            query,
            key.permute(0, 1, 3, 2).contiguous(),
            nbhd_idx)
        ctx.save_for_backward(query, key, nbhd_idx)
        return attn

    @staticmethod
    def backward(ctx, grad_attn):
        outputs = clustenqk_cuda.backward(
            grad_attn.contiguous(), *ctx.saved_tensors)
        d_query, d_key = outputs
        return d_query, d_key, None


class CLUSTENAVFunction(Function):
    """
    attention times value function
    """
    @staticmethod
    def forward(ctx, attn, v, nbhd_idx):
        attn = attn.contiguous()
        v = v.contiguous()
        nbhd_idx = nbhd_idx.contiguous()
        if attn.dtype != v.dtype:
            v = v.to(attn.dtype)
        feat = clustenav_cuda.forward(
            attn,
            v,
            nbhd_idx)
        ctx.save_for_backward(attn, v, nbhd_idx)
        return feat

    @staticmethod
    def backward(ctx, grad_feat):
        outputs = clustenav_cuda.backward(
            grad_feat.contiguous(), *ctx.saved_tensors)
        d_attn, d_v = outputs
        return d_attn, d_v, None


class CLUSTENWFFunction(Function):
    """
    weight times feature function
    """
    @staticmethod
    def forward(ctx, weights, feat, nbhd_idx):
        weights = weights.contiguous()
        feat = feat.contiguous()
        nbhd_idx = nbhd_idx.contiguous()
        if feat.dtype != weights.dtype:
            feat = feat.to(weights.dtype)
        feat_new = clustenwf_cuda.forward(
            weights,
            feat,
            nbhd_idx)
        ctx.save_for_backward(weights, feat, nbhd_idx)
        return feat_new

    @staticmethod
    def backward(ctx, grad_feat_new):
        outputs = clustenwf_cuda.backward(
            grad_feat_new.contiguous(), *ctx.saved_tensors)
        d_weights, d_feat = outputs
        return d_weights, d_feat, None
