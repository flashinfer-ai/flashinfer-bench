import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

leaky_relu_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Simple error checking macro
#define CHECK_CUDA_ERROR(call)                                         \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            printf("CUDA Error: %s:%d, ", __FILE__, __LINE__);         \
            printf("code:%d, reason: %s\n", err, cudaGetErrorString(err)); \
            return torch::Tensor();                                    \
        }                                                              \
    } while(0)

__global__ void leaky_relu_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  const float negative_slope,
                                  const int size) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float val = in[idx];
        out[idx] = (val >= 0.0f) ? val : (val * negative_slope);
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor x, float negative_slope) {
    // Ensure input is contiguous and on CUDA
    x = x.contiguous();
    auto out = torch::empty_like(x);

    const int size = x.numel();
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    // Launch kernel
    leaky_relu_kernel<<<grid_size, block_size>>>(x.data_ptr<float>(),
                                                 out.data_ptr<float>(),
                                                 negative_slope,
                                                 size);

    // Check for errors after kernel launch
    CHECK_CUDA_ERROR(cudaGetLastError());
    // Optional: synchronize here for debugging or timing
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return out;
}
"""

leaky_relu_cpp_source = r"torch::Tensor leaky_relu_cuda(torch::Tensor x, float negative_slope);"

# Compile the inline CUDA code
leaky_relu_extension = load_inline(
    name="leaky_relu_extension",
    cpp_sources=leaky_relu_cpp_source,
    cuda_sources=leaky_relu_source,
    functions=["leaky_relu_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a LeakyReLU activation with a custom CUDA kernel.
    """
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return leaky_relu_extension.leaky_relu_cuda(x, self.negative_slope)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []
