import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

hardtanh_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

// HardTanh kernel using a grid-stride loop
__global__ void hardtanh_kernel(const float* __restrict__ inp,
                                float* __restrict__ out,
                                int size,
                                float min_val,
                                float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        float val = inp[i];
        // clamp value between min_val and max_val
        if (val < min_val) val = min_val;
        if (val > max_val) val = max_val;
        out[i] = val;
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor inp, float min_val, float max_val) {
    auto out = torch::empty_like(inp);
    int size = inp.numel();

    // Configure the block and grid sizes
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    // Launch kernel
    hardtanh_kernel<<<blocks, threads>>>(inp.data_ptr<float>(),
                                         out.data_ptr<float>(),
                                         size,
                                         min_val,
                                         max_val);
    return out;
}
"""

hardtanh_cpp_source = r"""
torch::Tensor hardtanh_cuda(torch::Tensor inp, float min_val, float max_val);
"""

# Compile the inline CUDA code for hardtanh
hardtanh = load_inline(
    name="hardtanh",
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=["hardtanh_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a HardTanh activation using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardtanh_lib = hardtanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardtanh_lib.hardtanh_cuda(x, -1.0, 1.0)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
