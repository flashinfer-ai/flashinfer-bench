import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mse_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Optimized kernel to compute partial sums of (pred - targ)^2.
// Uses shared memory reduction per block.
__global__ void compute_mse_partial_kernel(const float* __restrict__ pred,
                                           const float* __restrict__ targ,
                                           float* __restrict__ partial_sums,
                                           int size) {
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Each thread computes a local sum
    float val = 0.0f;
    if (idx < size) {
        float diff = pred[idx] - targ[idx];
        val = diff * diff;
    }
    sdata[tid] = val;
    __syncthreads();

    // Reduce within the block (power-of-two reductions)
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write out per-block results to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Final reduction kernel to sum up block-partial sums and produce a scalar
__global__ void final_reduce_kernel(const float* __restrict__ partial_sums,
                                    int num_blocks,
                                    int size,
                                    float* __restrict__ out) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float val = 0.0f;

    // Load each block's partial sum
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        val += partial_sums[i];
    }
    sdata[tid] = val;
    __syncthreads();

    // In-block reduce
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write final scalar to global memory (divide by size to get mean)
    if (tid == 0) {
        out[0] = sdata[0] / static_cast<float>(size);
    }
}

torch::Tensor mse_cuda_forward(torch::Tensor predictions, torch::Tensor targets) {
    // Check that both inputs are on the same device
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");

    // Get size info
    int size = predictions.numel();
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    // Allocate partial sums for each block
    auto partial_sums = torch::zeros({grid_size}, predictions.options());

    // Launch partial sums kernel
    int shared_mem_size = block_size * sizeof(float);
    compute_mse_partial_kernel<<<grid_size, block_size, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        size
    );

    // Final output: 0-dim (scalar) tensor
    auto out = torch::empty({}, predictions.options()); // shape[]

    // Launch final reduction kernel
    final_reduce_kernel<<<1, block_size, shared_mem_size>>>(
        partial_sums.data_ptr<float>(),
        grid_size,
        size,
        out.data_ptr<float>()
    );

    return out;
}
"""

mse_cpp_source = r"""
torch::Tensor mse_cuda_forward(torch::Tensor predictions, torch::Tensor targets);
""".strip()

# Compile the custom CUDA code for MSE
mse_extension = load_inline(
    name="mse_extension",
    cpp_sources=mse_cpp_source,
    cuda_sources=mse_source,
    functions=["mse_cuda_forward"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized MSE model using a custom CUDA kernel that returns a scalar.
    """
    def __init__(self):
        super().__init__()
        self.mse_extension = mse_extension

    def forward(self, predictions, targets):
        return self.mse_extension.mse_cuda_forward(predictions, targets)
