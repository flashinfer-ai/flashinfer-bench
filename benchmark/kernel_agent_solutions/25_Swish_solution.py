import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

swish_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Tile size for partial unrolling in the kernel
#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

// Swish kernel with partial unrolling, coalesced accesses, and error checking
__global__ void swish_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    int global_idx = blockIdx.x * blockDim.x * TILE_SIZE + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i++) {
        int idx = global_idx + i * blockDim.x;
        if (idx < size) {
            float val = x[idx];
            float s   = 1.0f / (1.0f + expf(-val));
            out[idx]  = val * s;
        }
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    // Configure block and grid sizes
    const int block_size = 256;
    const int grid_size = (size + block_size * TILE_SIZE - 1) / (block_size * TILE_SIZE);

    swish_kernel<<<grid_size, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    // Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\\n", cudaGetErrorString(err));
    }
    return out;
}
"""

swish_cpp_source = r"""
torch::Tensor swish_forward(torch::Tensor x);
"""

swish_module = load_inline(
    name="swish_module",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=["swish_forward"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a Swish activation using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish_module.swish_forward(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim, device="cuda")
    return [x]

def get_init_inputs():
    return []
