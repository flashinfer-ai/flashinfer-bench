import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define constants for SELU
SELU_ALPHA = 1.6732632423543772848170429916717
SELU_SCALE = 1.0507009873554804934193349852946

# Inline CUDA code for SELU activation
selu_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Error checking macro
#define CHECK_CUDA_ERROR(err) \
  if (err != cudaSuccess) { \
    printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  }

__global__ void selu_kernel(const float* __restrict__ inp, float* __restrict__ out,
                            const float alpha, const float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = inp[idx];
        if (x > 0.0f) {
            out[idx] = scale * x;
        } else {
            out[idx] = scale * (alpha * (expf(x) - 1.0f));
        }
    }
}

torch::Tensor selu_activation(torch::Tensor x, double alpha, double scale) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    // Launch kernel
    selu_kernel<<<grid_size, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(),
                                           static_cast<float>(alpha), static_cast<float>(scale),
                                           size);

    // Check for launch error
    cudaError_t err = cudaGetLastError();
    CHECK_CUDA_ERROR(err);
    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    return out;
}
""".replace('\\', '\\\\')

selu_cpp_source = "torch::Tensor selu_activation(torch::Tensor x, double alpha, double scale);"

# Compile the inline CUDA code
selu_ext = load_inline(
    name="selu_ext",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=["selu_activation"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    """
    Model that performs a custom SELU activation using the above CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_op = selu_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu_op.selu_activation(x, SELU_ALPHA, SELU_SCALE)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
