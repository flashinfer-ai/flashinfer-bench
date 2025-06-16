import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

inorm_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// A small epsilon for numerical stability
#ifndef EPS
#define EPS 1e-5
#endif

// Block-wide reduction in shared memory for partial sums
template <typename T>
__inline__ __device__ void blockReduceSum(T* smem) {
    // Assumes blockDim.x is a power of 2
    unsigned int t = threadIdx.x;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            smem[t] += smem[t + s];
        }
        __syncthreads();
    }
}

// First pass: partial sums for each (N, C) across spatial (H*W)
__global__ void partial_sums_kernel(
    const float* __restrict__ input,
    float* __restrict__ sums,
    float* __restrict__ sums_sq,
    const int N, const int C, const int H, const int W
) {
    // One block per (N*C), each block accumulates sums across H*W
    int channel_idx = blockIdx.x; // ranges from 0 to N*C-1
    if (channel_idx >= N * C) {
        return;
    }

    int n = channel_idx / C;
    int c = channel_idx % C;
    int total_pixels = H * W;

    extern __shared__ float sdata[]; // used for partial sums
    float* sdata_sq = sdata + blockDim.x; // second half for squares

    // Initialize shared memory
    sdata[threadIdx.x] = 0.0f;
    sdata_sq[threadIdx.x] = 0.0f;
    __syncthreads();

    // Base pointer for this (n, c)
    int base_idx = (n * C + c) * H * W;

    // Stride through all pixels of (n,c)
    for (int i = threadIdx.x; i < total_pixels; i += blockDim.x) {
        float val = input[base_idx + i];
        sdata[threadIdx.x] += val;
        sdata_sq[threadIdx.x] += val * val;
    }
    __syncthreads();

    // Reduce within the block
    blockReduceSum(sdata);
    blockReduceSum(sdata_sq);

    // The first thread in the block writes to global memory
    if (threadIdx.x == 0) {
        atomicAdd(&sums[channel_idx], sdata[0]);
        atomicAdd(&sums_sq[channel_idx], sdata_sq[0]);
    }
}

// Second pass: compute means and variances for each (N, C)
__global__ void finalize_means_vars_kernel(
    const float* __restrict__ sums,
    const float* __restrict__ sums_sq,
    float* __restrict__ means,
    float* __restrict__ vars,
    const int N, const int C, const int H, const int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_channels = N * C;
    if (idx >= total_channels) return;

    float denom = static_cast<float>(H * W);
    float mean = sums[idx] / denom;
    float mean_sq = sums_sq[idx] / denom;
    float var = mean_sq - mean * mean;
    if (var < 0.0f) {
        var = 0.0f; // numerical safeguard
    }
    means[idx] = mean;
    vars[idx] = var;
}

// Third pass: apply instance normalization
__global__ void instance_norm_fwd_kernel(
    const float* __restrict__ input,
    const float* __restrict__ means,
    const float* __restrict__ vars,
    float* __restrict__ output,
    const int N, const int C, const int H, const int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    tmp /= H;
    int c = tmp % C;
    int n = tmp / C;
    int channel_idx = n * C + c;

    float mean_val = means[channel_idx];
    float var_val = vars[channel_idx];
    float x = input[idx];

    // normalize (no affine transform here, just the simple version)
    float normed = (x - mean_val) / sqrtf(var_val + EPS);
    output[idx] = normed;
}

// Combined interface function: calls the three kernels in sequence
torch::Tensor instance_norm_cuda(
    torch::Tensor input
) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto sums = torch::zeros({N*C}, input.options());
    auto sums_sq = torch::zeros({N*C}, input.options());
    auto means = torch::zeros({N*C}, input.options());
    auto vars = torch::zeros({N*C}, input.options());
    auto output = torch::empty_like(input);

    // Launch partial sums kernel
    int block = 256;
    int grid = N*C;
    size_t shmem_size = 2 * block * sizeof(float);
    partial_sums_kernel<<<grid, block, shmem_size>>>(
        input.data_ptr<float>(),
        sums.data_ptr<float>(),
        sums_sq.data_ptr<float>(),
        N, C, H, W
    );

    // Launch finalize means and vars kernel
    int total_channels = N*C;
    int block2 = 256;
    int grid2 = (total_channels + block2 - 1) / block2;
    finalize_means_vars_kernel<<<grid2, block2>>>(
        sums.data_ptr<float>(),
        sums_sq.data_ptr<float>(),
        means.data_ptr<float>(),
        vars.data_ptr<float>(),
        N, C, H, W
    );

    // Launch instance norm fwd kernel
    int total_elems = N*C*H*W;
    int block3 = 256;
    int grid3 = (total_elems + block3 - 1) / block3;
    instance_norm_fwd_kernel<<<grid3, block3>>>(
        input.data_ptr<float>(),
        means.data_ptr<float>(),
        vars.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );

    return output;
}
'''

inorm_cpp_decl = r'''
torch::Tensor instance_norm_cuda(torch::Tensor input);
'''

# Build the custom inline extension
inorm_extension = load_inline(
    name="inorm_extension",
    cpp_sources=inorm_cpp_decl,
    cuda_sources=inorm_source,
    functions=["instance_norm_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Custom Instance Norm model using a fused custom CUDA kernel.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the custom instance normalization kernel
        return inorm_extension.instance_norm_cuda(x)
