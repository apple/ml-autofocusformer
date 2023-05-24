/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2023 Apple Inc. All Rights Reserved.
 */

#include <torch/extension.h>
#include <vector>

torch::Tensor clusten_wf_cuda_forward(
    const torch::Tensor &weights,                           // b x n_ x m x ic
    const torch::Tensor &feat,                              // b x n x c
    const torch::Tensor &nbhd_idx);                         // b x n_ x m

std::vector<torch::Tensor> clusten_wf_cuda_backward(
    const torch::Tensor &d_feat_new, 
    const torch::Tensor &weights,
    const torch::Tensor &feat,
    const torch::Tensor &nbhd_idx);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor clusten_wf_forward(           
    const torch::Tensor &weights,
    const torch::Tensor &feat,
    const torch::Tensor &nbhd_idx) {
    CHECK_INPUT(weights);
    CHECK_INPUT(feat);
    CHECK_INPUT(nbhd_idx);
    return clusten_wf_cuda_forward(weights, feat, nbhd_idx);
}

std::vector<torch::Tensor> clusten_wf_backward(
    const torch::Tensor &d_feat_new,
    const torch::Tensor &weights,
    const torch::Tensor &feat,
    const torch::Tensor &nbhd_idx) {
    CHECK_INPUT(d_feat_new);
    CHECK_INPUT(weights);
    CHECK_INPUT(feat);
    CHECK_INPUT(nbhd_idx);
    return clusten_wf_cuda_backward(d_feat_new, weights, feat, nbhd_idx);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &clusten_wf_forward, "CLUSTENWF forward (CUDA)");
  m.def("backward", &clusten_wf_backward, "CLUSTENWF backward (CUDA)");
}
