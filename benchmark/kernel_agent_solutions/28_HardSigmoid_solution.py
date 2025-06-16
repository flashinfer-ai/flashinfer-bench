import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hardsigmoid_cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void hardsigmoid_kernel(const float* __restrict__ inp,
                                   float* __restrict__ out,
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = inp[idx];
        // HardSigmoid: max(0, min(6, x + 3)) / 6
        float val = fmaxf(0.0f, fminf(6.0f, x + 3.0f)) / 6.0f;
        out[idx] = val;
    }
}

torch::Tensor hardsigmoid_cuda(torch::Tensor inp) {
    auto size = inp.numel();
    auto out = torch::empty_like(inp);

    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / blockSize;
    hardsigmoid_kernel<<<gridSize, blockSize>>>(inp.data_ptr<float>(),
                                               out.data_ptr<float>(),
                                               size);
    return out;
}
"""

hardsigmoid_cuda_header = r"""
torch::Tensor hardsigmoid_cuda(torch::Tensor inp);
"""

hardsigmoid_mod = load_inline(
    name="hardsigmoid_cuda",
    cpp_sources=hardsigmoid_cuda_header,
    cuda_sources=hardsigmoid_cuda_src,
    functions=["hardsigmoid_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hardsigmoid_mod.hardsigmoid_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
