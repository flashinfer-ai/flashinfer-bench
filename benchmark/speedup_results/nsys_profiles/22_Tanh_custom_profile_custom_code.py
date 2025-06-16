import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

tanh_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(err) \
  do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
      printf("CUDA error: %s (err_num=%d)\n", cudaGetErrorString(err_), err_); \
      exit(1); \
    } \
  } while(0)

__global__ void tanh_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

torch::Tensor tanh_cuda_forward(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    // Determine launch configuration
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    // Launch kernel
    tanh_kernel<<<grid_size, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    // Error checks
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return out;
}
"""

tanh_cuda_header = """
torch::Tensor tanh_cuda_forward(torch::Tensor x);
"""

tanh_cuda_module = load_inline(
    name="tanh_cuda_module",
    cpp_sources=tanh_cuda_header,
    cuda_sources=tanh_cuda_source,
    functions=["tanh_cuda_forward"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Model that replaces torch.tanh with a custom CUDA tanh kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh_cuda_op = tanh_cuda_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the custom CUDA Tanh kernel to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        return self.tanh_cuda_op.tanh_cuda_forward(x)


batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
