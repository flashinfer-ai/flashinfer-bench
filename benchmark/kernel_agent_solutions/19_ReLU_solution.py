import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

relu_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor relu_cuda(torch::Tensor x) {
    auto out = torch::zeros_like(x);
    auto size = x.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    relu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

relu_cuda_header = "torch::Tensor relu_cuda(torch::Tensor x);"

relu_ext = load_inline(
    name="relu_ext",
    cpp_sources=relu_cuda_header,
    cuda_sources=relu_cuda_source,
    functions=["relu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu_ext = relu_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_ext.relu_cuda(x)


batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
