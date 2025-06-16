import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

gelu_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Approximate GELU device function
__device__ __forceinline__ float gelu_approx(float x) {
    // Constant from sqrt(2 / pi)
    const float kBeta = 0.7978845608f;
    // Coefficient from OpenAI approximation
    const float kKappa = 0.044715f;
    float inner = kBeta * (x + kKappa * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// GELU kernel
__global__ void gelu_kernel(const float* __restrict__ in, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = gelu_approx(in[idx]);
    }
}

// C++ function to launch the kernel
torch::Tensor gelu_cuda_forward(torch::Tensor x) {
    // Ensure memory is contiguous
    x = x.contiguous();
    auto out = torch::empty_like(x);

    const int size = x.numel();
    const float* in_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    gelu_kernel<<<grid_size, block_size>>>(in_ptr, out_ptr, size);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed with error: ", cudaGetErrorString(err));
    }

    return out;
}
"""

gelu_cpp_source = r"""
    torch::Tensor gelu_cuda_forward(torch::Tensor x);
"""

gelu_extension = load_inline(
    name="gelu_extension",
    cpp_sources=[gelu_cpp_source],
    cuda_sources=[gelu_source],
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O2"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a custom GELU activation with a CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gelu_extension.gelu_cuda_forward(x)
