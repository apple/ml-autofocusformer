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
__global__ void clusten_wf_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> weights,            // b x n_ x m x ic
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> feat,               // b x n x c
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,            // b x n_ x m
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> feat_new,                 // b x n_ x ic x c
    const int length,               // n
    const int length_out,           // n_
    const int batch_size,           // b
    const int nbhd_size,            // m
    const int dim,                  // c
    const int dim_inner) {          // ic

    const int b = blockIdx.z * blockDim.z + threadIdx.z;
    if (b < batch_size){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length_out){
            const int c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c < dim){
                int64_t nbi;
                // calculate weights@feat
                scalar_t updt;
                #pragma unroll
                for (unsigned int ic=0; ic < dim_inner; ++ic) {
                    updt = scalar_t(0);
                    #pragma unroll
                    for (unsigned int ni=0; ni < nbhd_size; ++ni) {
                        nbi = nbhd_idx[b][i][ni];
                        updt += weights[b][i][ni][ic] * feat[b][nbi][c];
                    }
                    feat_new[b][i][ic][c] = updt;
                }
            }
        }
    }
}


torch::Tensor clusten_wf_cuda_forward(
    const torch::Tensor &weights,
    const torch::Tensor &feat,
    const torch::Tensor &nbhd_idx) {

    int64_t batch_size = weights.size(0);
    int64_t length_out = weights.size(1);
    int64_t nbhd_size = weights.size(2);
    int64_t dim_inner = weights.size(3);
    int64_t length = feat.size(1);
    int64_t dim = feat.size(2);

    int CHANNELTHREADS = min(int64_t(CUDA_NUM_THREADS), dim);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / CHANNELTHREADS), length_out);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * CHANNELTHREADS));

    auto feat_new = torch::zeros(
            {batch_size, length_out, dim_inner, dim}, weights.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (dim + CHANNELTHREADS - 1) / CHANNELTHREADS,
            (length_out + TOKENTHREADS - 1) / TOKENTHREADS,
            (batch_size + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(CHANNELTHREADS, TOKENTHREADS, BATCHTHREADS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(weights.scalar_type(), "clusten_wf_cuda_forward", ([&] {
                const auto weights_a = weights.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
                const auto feat_a = feat.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
                const auto nbhd_idx_a = nbhd_idx.packed_accessor32<int64_t,3,torch::DefaultPtrTraits>();
                auto feat_new_a = feat_new.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();

                clusten_wf_cuda_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                        weights_a, feat_a, nbhd_idx_a, feat_new_a,
                        length, length_out, batch_size, nbhd_size, dim, dim_inner);
                }));
    return feat_new;
}


template <typename scalar_t>
__global__ void clusten_wf_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_feat_new,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> weights,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> d_feat,
    const int length,               // n
    const int length_out,           // n_
    const int batch_size,           // b
    const int nbhd_size,            // m
    const int dim,                  // c
    const int dim_inner,            // ic
    const size_t d_feat_numel) {

    const int b = blockIdx.z * blockDim.z + threadIdx.z;
    if (b < batch_size){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length_out){
            const int c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c < dim){
                int64_t nbi;
                size_t index;
                scalar_t updt;
                #pragma unroll
                for (unsigned int ni=0; ni < nbhd_size; ++ni) {
                    nbi = nbhd_idx[b][i][ni];
                    updt = scalar_t(0);
                    // calculate d_feat = weights * d_feat_new
                    #pragma unroll
                    for (unsigned int ic=0; ic < dim_inner; ++ic) {
                        updt += d_feat_new[b][i][ic][c] * weights[b][i][ni][ic];
                    }
                    index = b*d_feat.stride(0) + nbi*d_feat.stride(1) + c;
                    at::native::fastAtomicAdd(d_feat.data(), index, d_feat_numel, updt, true);
                    // atomicAdd(&(d_feat[b][nbi][c]), updt); // avoid race condition
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void clusten_wf_weights_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_feat_new,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> feat,
    const torch::PackedTensorAccessor32<int64_t,3,torch::DefaultPtrTraits> nbhd_idx,
    torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits> d_weights,
    const int length,               // n
    const int length_out,           // n_
    const int batch_size,           // b
    const int nbhd_size,            // m
    const int dim,                  // c
    const int dim_inner){           // ic

    const int b = blockIdx.z * blockDim.z + threadIdx.z;
    if (b < batch_size){
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < length_out){
            const int z = blockIdx.x * blockDim.x + threadIdx.x;
            if (z < nbhd_size * dim_inner){
                const int ni = z / dim_inner;
                const int ic = z - ni * dim_inner;
                int64_t nbi = nbhd_idx[b][i][ni];
                scalar_t updt = scalar_t(0);
                #pragma unroll
                for (unsigned int c=0; c < dim; ++c) {
                    // calculate d_weights = feat * d_feat_new
                    updt += feat[b][nbi][c] * d_feat_new[b][i][ic][c];
                }
                d_weights[b][i][ni][ic] = updt;
            }
        }
    }
}

std::vector<torch::Tensor> clusten_wf_cuda_backward(
    const torch::Tensor &d_feat_new,
    const torch::Tensor &weights,
    const torch::Tensor &feat,
    const torch::Tensor &nbhd_idx) {

    int64_t batch_size = weights.size(0);
    int64_t length_out = weights.size(1);
    int64_t nbhd_size = weights.size(2);
    int64_t dim_inner = weights.size(3);
    int64_t length = feat.size(1);
    int64_t dim = feat.size(2);

    int64_t zsize = nbhd_size * dim_inner;

    int CHANNELTHREADS = min(int64_t(CUDA_NUM_THREADS), dim);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / CHANNELTHREADS), length_out);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS* CHANNELTHREADS));

    int NBHDTHREADS = min(int64_t(CUDA_NUM_THREADS), zsize);
    int TOKENTHREADS_NB = min(int64_t(CUDA_NUM_THREADS / NBHDTHREADS), length_out);
    int BATCHTHREADS_NB = max(1, CUDA_NUM_THREADS / (TOKENTHREADS_NB* NBHDTHREADS));

    auto d_weights = torch::zeros_like(weights);
    auto d_feat = torch::zeros_like(feat);

    const auto stream = c10::cuda::getCurrentCUDAStream();

    const dim3 blocks(
            (dim + CHANNELTHREADS - 1) / CHANNELTHREADS,
            (length_out + TOKENTHREADS - 1) / TOKENTHREADS,
            (batch_size + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(CHANNELTHREADS, TOKENTHREADS, BATCHTHREADS);

    const dim3 blocks_nb(
            (zsize + NBHDTHREADS - 1) / NBHDTHREADS,
            (length_out + TOKENTHREADS_NB - 1) / TOKENTHREADS_NB,
            (batch_size + BATCHTHREADS_NB - 1) / BATCHTHREADS_NB);
    const dim3 threads_nb(NBHDTHREADS, TOKENTHREADS_NB, BATCHTHREADS_NB);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(weights.scalar_type(), "clusten_wf_cuda_backward", ([&] {
                const auto d_feat_new_a = d_feat_new.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
                const auto weights_a = weights.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
                const auto feat_a = feat.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
                const auto nbhd_idx_a = nbhd_idx.packed_accessor32<int64_t,3,torch::DefaultPtrTraits>();
                auto d_weights_a = d_weights.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
                auto d_feat_a = d_feat.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();

                const size_t d_feat_numel = d_feat.numel();
                clusten_wf_cuda_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                        d_feat_new_a, weights_a, nbhd_idx_a, d_feat_a,
                        length, length_out, batch_size, nbhd_size, dim, dim_inner, d_feat_numel);
                clusten_wf_weights_cuda_backward_kernel<scalar_t><<<blocks_nb, threads_nb, 0, stream>>>(
                        d_feat_new_a, feat_a, nbhd_idx_a, d_weights_a,
                        length, length_out, batch_size, nbhd_size, dim, dim_inner);
                }));

    return {d_weights, d_feat.to(feat.dtype())};
}
