import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# A simple custom CUDA kernel for elementwise ReLU
relu_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor elementwise_relu_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    int numel = input.numel();
    const int blockSize = 256;
    const int gridSize = (numel + blockSize - 1) / blockSize;

    relu_kernel<<<gridSize, blockSize>>>(input.data_ptr<float>(), output.data_ptr<float>(), numel);
    return output;
}
""";

relu_cpp_source = r"""
torch::Tensor elementwise_relu_cuda(torch::Tensor input);
""";

# Build the custom elementwise ReLU extension
elementwise_relu = load_inline(
    name="elementwise_relu",
    cpp_sources=[relu_cpp_source],
    cuda_sources=[relu_source],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution followed by a custom CUDA-based elementwise ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        # Adjust output_padding to avoid conflicts with stride/dilation
        valid_output_padding = min(output_padding, max(0, stride - 1))
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            output_padding=valid_output_padding,
            groups=groups,
            bias=bias
        )
        self.elementwise_relu = elementwise_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_transpose3d(x)
        out = self.elementwise_relu.elementwise_relu_cuda(out)
        return out
