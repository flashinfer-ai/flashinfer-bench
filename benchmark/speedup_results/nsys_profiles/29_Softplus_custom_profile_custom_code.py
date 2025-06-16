import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

softplus_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// A simple custom Softplus kernel with threshold to avoid overflow
__global__ void softplus_kernel(const float* x, float* y, float threshold, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // Mimic PyTorch's default Softplus with threshold=20.0
        // if val > threshold: out = val
        // else: out = log(1 + exp(val))
        if (val > threshold) {
            y[idx] = val;
        } else {
            y[idx] = log1pf(expf(val));
        }
    }
}

torch::Tensor softplus_cuda(torch::Tensor x, float threshold) {
    auto out = torch::zeros_like(x);
    int size = x.numel();
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    softplus_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        threshold,
        size
    );
    return out;
}
""";

softplus_cpp_source = r"""
torch::Tensor softplus_cuda(torch::Tensor x, float threshold);
""";

softplus_extension = load_inline(
    name="softplus_extension",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a Softplus activation with a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.threshold = 20.0  # default Softplus threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return softplus_extension.softplus_cuda(x, self.threshold)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
