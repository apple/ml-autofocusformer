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
__global__ void clusten_qk_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> query,      // b x h x n x c
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> key,        // b x h x c x n (reordered by cluster)
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,    // b x n x m
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> attn,             // b x h x n x m
    const int length,           // n
    const int batch_size,       // b
    const int heads,            // h
    const int nbhd_size,        // m
    const int dim) {            // c

    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length){
            const int ni = blockIdx.x * blockDim.x + threadIdx.x;
            if (ni < nbhd_size){
                const int b = z / heads;
                const int h = z - b * heads;
                int64_t nbi = nbhd_idx[b][i][ni];
                // calculate q@k
                scalar_t updt = scalar_t(0);
                #pragma unroll
                for (unsigned int c=0; c < dim; ++c) {
                    updt += query[b][h][i][c] * key[b][h][c][nbi];
                }
                attn[b][h][i][ni] = updt;
            }
        }
    }
}


torch::Tensor clusten_qk_cuda_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &nbhd_idx) {

    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t length = query.size(2);
    int64_t dim = query.size(3);
    int64_t nbhd_size = nbhd_idx.size(2);
    int zsize = batch_size * heads;

    int NBHDTHREADS = min(int64_t(CUDA_NUM_THREADS), nbhd_size);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / NBHDTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * NBHDTHREADS));

    auto attn = torch::zeros(
            {batch_size, heads, length, nbhd_size}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (dim + NBHDTHREADS - 1) / NBHDTHREADS,
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(NBHDTHREADS, TOKENTHREADS, BATCHTHREADS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "clusten_qk_cuda_forward", ([&] {
        const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto nbhd_idx_a = nbhd_idx.packed_accessor32<int64_t,3,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();

        clusten_qk_cuda_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                query_a, key_a, nbhd_idx_a, attn_a, 
                length, batch_size, heads, nbhd_size, dim);
    }));
    return attn;
}

template <typename scalar_t>
__global__ void clusten_qk_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_query,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_key,
    const int length,
    const int batch_size,
    const int heads,
    const int nbhd_size,
    const int dim,
    const size_t d_key_numel) {

    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length){
            const int c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c < dim){
                const int b = z / heads;
                const int h = z - b * heads;
                size_t index;
                scalar_t dq_update = scalar_t(0);
                scalar_t d_attn_tmp;
                #pragma unroll
                for (unsigned int ni=0; ni < nbhd_size; ++ni) {
                    const int64_t nbi = nbhd_idx[b][i][ni];
                    // calculate d_query = key * d_att
                    // calculate d_key = query * d_att
                    d_attn_tmp = d_attn[b][h][i][ni];
                    dq_update += key[b][h][nbi][c] * d_attn_tmp;
                    index = b*d_key.stride(0) + h*d_key.stride(1) + nbi*d_key.stride(2) + c;
                    at::native::fastAtomicAdd(d_key.data(), index, d_key_numel, query[b][h][i][c] * d_attn_tmp, true);
                    //atomicAdd(&(d_key[b][h][nbi][c]), query[b][h][i][c] * d_attn_tmp); // avoid race condition
                }
                d_query[b][h][i][c] = dq_update;
            }
        }
    }
}

std::vector<torch::Tensor> clusten_qk_cuda_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &nbhd_idx) {

    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t length = query.size(2);
    int64_t dim = query.size(3);
    int64_t nbhd_size = nbhd_idx.size(2);
    int zsize = batch_size * heads;

    int CHANNELTHREADS = min(int64_t(CUDA_NUM_THREADS), dim);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / CHANNELTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * CHANNELTHREADS));

    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);

    const auto stream = c10::cuda::getCurrentCUDAStream();

    const dim3 blocks(
            (dim + CHANNELTHREADS - 1) / CHANNELTHREADS,
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);

    const dim3 threads(CHANNELTHREADS, TOKENTHREADS, BATCHTHREADS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "clusten_qk_cuda_backward", ([&] {
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto nbhd_idx_a = nbhd_idx.packed_accessor32<int64_t,3,torch::DefaultPtrTraits>();
        auto d_query_a = d_query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_key_a = d_key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();

        const size_t d_key_numel = d_key.numel();
        clusten_qk_cuda_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                d_attn_a, query_a, key_a, nbhd_idx_a, d_query_a, d_key_a,
                length, batch_size, heads, nbhd_size, dim, d_key_numel);
    }));

    return {d_query, d_key.to(key.dtype())};
}
