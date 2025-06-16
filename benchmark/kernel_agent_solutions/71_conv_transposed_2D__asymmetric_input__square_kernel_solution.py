import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for an elementwise ReLU operation
relu_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    relu_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

relu_cpp_source = "torch::Tensor relu_cuda(torch::Tensor input);"

my_relu = load_inline(
    name="my_relu",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    verbose=False,
    functions=["relu_cuda"],
    extra_cflags=[],
    extra_cuda_cflags=["-O2"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False
    ):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_transpose2d(x)
        out = my_relu.relu_cuda(out)
        return out

def get_init_inputs():
    return [32, 64, 3]

def get_inputs():
    return [torch.randn(16, 32, 128, 256)]
