/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2023 Apple Inc. All Rights Reserved.
 */

#include <torch/extension.h>
#include <vector>

torch::Tensor clusten_qk_cuda_forward(
    const torch::Tensor &query,             // b x h x n x c
    const torch::Tensor &key,               // b x h x n x c
    const torch::Tensor &nbhd_idx);         // b x n x m

std::vector<torch::Tensor> clusten_qk_cuda_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &nbhd_idx);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor clusten_qk_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &nbhd_idx) {
    CHECK_INPUT(query);
    CHECK_INPUT(key);
    CHECK_INPUT(nbhd_idx);
    return clusten_qk_cuda_forward(query, key, nbhd_idx);
}

std::vector<torch::Tensor> clusten_qk_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &nbhd_idx) {
    CHECK_INPUT(d_attn);
    CHECK_INPUT(query);
    CHECK_INPUT(key);
    CHECK_INPUT(nbhd_idx);
    return clusten_qk_cuda_backward(d_attn, query, key, nbhd_idx);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &clusten_qk_forward, "CLUSTENQK forward (CUDA)");
  m.def("backward", &clusten_qk_backward, "CLUSTENQK backward (CUDA)");
}
