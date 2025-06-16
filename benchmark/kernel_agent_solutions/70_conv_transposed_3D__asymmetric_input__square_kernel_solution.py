import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# A simple custom CUDA kernel for ReLU
relu_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ inp, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = inp[idx];
        out[idx] = val > 0.f ? val : 0.f;
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    int size = input.numel();

    const int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    relu_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );

    return output;
}
"""

relu_cpp_source = r"""
torch::Tensor relu_cuda(torch::Tensor input);
"""

# Build the inline extension for the custom ReLU kernel
relu_module = load_inline(
    name="custom_relu",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    An optimized model that applies a transposed 3D convolution, followed by a custom CUDA-based ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, output_padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, 
            out_channels, 
            (kernel_size, kernel_size, kernel_size),
            stride=stride, 
            padding=padding, 
            output_padding=output_padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias
        )
        self.relu_mod = relu_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_transpose3d(x)
        out = self.relu_mod.relu_cuda(out)
        return out
