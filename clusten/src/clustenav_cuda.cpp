/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2023 Apple Inc. All Rights Reserved.
 */

#include <torch/extension.h>
#include <vector>

torch::Tensor clusten_av_cuda_forward(
    const torch::Tensor &attn,                          // b x h x n x m
    const torch::Tensor &v,                             // b x h x n x c
    const torch::Tensor &nbhd_idx);                     // b x n x m

std::vector<torch::Tensor> clusten_av_cuda_backward(
    const torch::Tensor &d_feat, 
    const torch::Tensor &attn,
    const torch::Tensor &v,
    const torch::Tensor &nbhd_idx);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor clusten_av_forward(           
    const torch::Tensor &attn,
    const torch::Tensor &v,
    const torch::Tensor &nbhd_idx) {
    CHECK_INPUT(attn);
    CHECK_INPUT(v);
    CHECK_INPUT(nbhd_idx);
    return clusten_av_cuda_forward(attn, v, nbhd_idx);
}

std::vector<torch::Tensor> clusten_av_backward(
    const torch::Tensor &d_feat,
    const torch::Tensor &attn,
    const torch::Tensor &v,
    const torch::Tensor &nbhd_idx) {
    CHECK_INPUT(d_feat);
    CHECK_INPUT(attn);
    CHECK_INPUT(v);
    CHECK_INPUT(nbhd_idx);
    return clusten_av_cuda_backward(d_feat, attn, v, nbhd_idx);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &clusten_av_forward, "CLUSTENAV forward (CUDA)");
  m.def("backward", &clusten_av_backward, "CLUSTENAV backward (CUDA)");
}
