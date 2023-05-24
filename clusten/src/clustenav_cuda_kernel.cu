/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2023 Apple Inc. All Rights Reserved.
 */

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/AccumulateType.h>

#define CUDA_NUM_THREADS 1024

template <typename scalar_t>
__global__ void clusten_av_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> attn,               // b x h x n x m
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> v,                  // b x h x n x c
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,            // b x n x m
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> feat,                     // b x n x c
    const int length,               // n
    const int batch_size,           // b
    const int heads,                // h
    const int nbhd_size,            // m
    const int dim) {                // c

    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length){
            const int c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c < dim){
                const int b = z / heads;
                const int h = z - b * heads;
                int64_t nbi;
                // calculate a@v
                scalar_t updt = scalar_t(0);
                #pragma unroll
                for (unsigned int ni=0; ni < nbhd_size; ++ni) {
                    nbi = nbhd_idx[b][i][ni];
                    updt += attn[b][h][i][ni] * v[b][h][nbi][c];
                }
                feat[b][h][i][c] = updt;
            }
        }
    }
}


torch::Tensor clusten_av_cuda_forward(
    const torch::Tensor &attn,             
    const torch::Tensor &v,               
    const torch::Tensor &nbhd_idx) { 

    int64_t batch_size = attn.size(0);
    int64_t heads = attn.size(1);
    int64_t length = attn.size(2);
    int64_t dim = v.size(3);
    int64_t nbhd_size = nbhd_idx.size(2);
    int zsize = batch_size * heads;

    int CHANNELTHREADS = min(int64_t(CUDA_NUM_THREADS), dim);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / CHANNELTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * CHANNELTHREADS));

    auto feat = torch::zeros(
            {batch_size, heads, length, dim}, v.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (dim + CHANNELTHREADS - 1) / CHANNELTHREADS,
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(CHANNELTHREADS, TOKENTHREADS, BATCHTHREADS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(attn.scalar_type(), "clusten_av_cuda_forward", ([&] {
        const auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto v_a = v.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto nbhd_idx_a = nbhd_idx.packed_accessor32<int64_t,3,torch::DefaultPtrTraits>();
        auto feat_a = feat.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();

        clusten_av_cuda_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                attn_a, v_a, nbhd_idx_a, feat_a,
                length, batch_size, heads, nbhd_size, dim);
    }));
    return feat;
}


template <typename scalar_t>
__global__ void clusten_av_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_feat,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> attn,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_v,
    const int length,
    const int batch_size,
    const int heads,
    const int nbhd_size,
    const int dim,
    const size_t d_v_numel) {

    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length){
            const int c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c < dim){
                const int b = z / heads;
                const int h = z - b * heads;
                int64_t nbi;
                size_t index;
                #pragma unroll
                for (unsigned int ni=0; ni < nbhd_size; ++ni) {
                    nbi = nbhd_idx[b][i][ni];
                    // calculate d_v = att * d_feat
                    index = b*d_v.stride(0) + h*d_v.stride(1) + nbi*d_v.stride(2) + c;
                    at::native::fastAtomicAdd(d_v.data(), index, d_v_numel, d_feat[b][h][i][c] * attn[b][h][i][ni], true);
                    // atomicAdd(&(d_v[b][h][nbi][c]), d_feat[b][h][i][c] * attn[b][h][i][ni]); // avoid race condition
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void clusten_av_attn_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_feat,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> v,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const int length,
    const int batch_size,
    const int heads,
    const int nbhd_size,
    const int dim) {

    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length){
            const int ni = blockIdx.x * blockDim.x + threadIdx.x;
            if (ni < nbhd_size){
                const int b = z / heads;
                const int h = z - b * heads;
                int64_t nbi = nbhd_idx[b][i][ni];
                scalar_t updt = scalar_t(0);
                #pragma unroll
                for (unsigned int c=0; c < dim; ++c) {
                    // calculate d_attn = v * d_feat
                    updt += v[b][h][nbi][c] * d_feat[b][h][i][c];
                }
                d_attn[b][h][i][ni] = updt;
            }
        }
    }
}

std::vector<torch::Tensor> clusten_av_cuda_backward(
    const torch::Tensor &d_feat,
    const torch::Tensor &attn,
    const torch::Tensor &v,
    const torch::Tensor &nbhd_idx) {

    int64_t batch_size = attn.size(0);
    int64_t heads = attn.size(1);
    int64_t length = attn.size(2);
    int64_t dim = v.size(3);
    int64_t nbhd_size = nbhd_idx.size(2);
    int zsize = batch_size * heads;

    int CHANNELTHREADS = min(int64_t(CUDA_NUM_THREADS), dim);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / CHANNELTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS* CHANNELTHREADS));

    int NBHDTHREADS = min(int64_t(CUDA_NUM_THREADS), nbhd_size);
    int TOKENTHREADS_NB = min(int64_t(CUDA_NUM_THREADS / NBHDTHREADS), length);
    int BATCHTHREADS_NB = max(1, CUDA_NUM_THREADS / (TOKENTHREADS_NB* NBHDTHREADS));

    auto d_attn = torch::zeros_like(attn);
    auto d_v = torch::zeros_like(v);

    const auto stream = c10::cuda::getCurrentCUDAStream();

    const dim3 blocks(
            (dim + CHANNELTHREADS - 1) / CHANNELTHREADS,
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(CHANNELTHREADS, TOKENTHREADS, BATCHTHREADS);

    const dim3 blocks_nb(
            (nbhd_size + NBHDTHREADS - 1) / NBHDTHREADS,
            (length + TOKENTHREADS_NB - 1) / TOKENTHREADS_NB,
            (zsize + BATCHTHREADS_NB - 1) / BATCHTHREADS_NB);
    const dim3 threads_nb(NBHDTHREADS, TOKENTHREADS_NB, BATCHTHREADS_NB);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(attn.scalar_type(), "clusten_av_cuda_backward", ([&] {
        const auto d_feat_a = d_feat.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto v_a = v.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto nbhd_idx_a = nbhd_idx.packed_accessor32<int64_t,3,torch::DefaultPtrTraits>();
        auto d_attn_a = d_attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_v_a = d_v.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();

        const size_t d_v_numel = d_v.numel();
        clusten_av_cuda_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                d_feat_a, attn_a, nbhd_idx_a, d_v_a,
                length, batch_size, heads, nbhd_size, dim, d_v_numel);
        clusten_av_attn_cuda_backward_kernel<scalar_t><<<blocks_nb, threads_nb, 0, stream>>>(
                d_feat_a, v_a, nbhd_idx_a, d_attn_a,
                length, batch_size, heads, nbhd_size, dim);
    }));

    return {d_attn, d_v.to(v.dtype())};
}
