import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA code for exclusive cumsum
exclusive_cumsum_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void exclusive_cumsum_kernel(const float* __restrict__ x,
                                        float* __restrict__ out,
                                        const int batch,
                                        const int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch) {
        int offset = i * length;
        out[offset] = 0.0f; 
        for (int j = 1; j < length; ++j) {
            out[offset + j] = out[offset + j - 1] + x[offset + j - 1];
        }
    }
}

torch::Tensor exclusive_cumsum_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D: [batch, length]");
    int batch = x.size(0);
    int length = x.size(1);

    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int grid_size = (batch + block_size - 1) / block_size;

    exclusive_cumsum_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        length
    );

    return out;
}
"""

exclusive_cumsum_cpp_source = r"torch::Tensor exclusive_cumsum_cuda(torch::Tensor x);"

# Compile the inline CUDA code for exclusive cumsum
exclusive_cumsum = load_inline(
    name="exclusive_cumsum",
    cpp_sources=exclusive_cumsum_cpp_source,
    cuda_sources=exclusive_cumsum_source,
    functions=["exclusive_cumsum_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs an exclusive cumsum on dim=1 with a custom CUDA kernel.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        # The custom kernel is designed to handle 2D [batch, length] along dim=1
        # For other dims, additional handling would be needed
        if self.dim != 1:
            raise ValueError("This custom kernel only supports dim=1 for 2D inputs.")

    def forward(self, x):
        # x is [batch, length]
        # The kernel itself handles the exclusive cumsum
        return exclusive_cumsum.exclusive_cumsum_cuda(x)

batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]
