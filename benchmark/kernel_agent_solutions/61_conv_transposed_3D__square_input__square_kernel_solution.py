import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA code for fused add-bias + ReLU after 3D transpose convolution
add_bias_relu_cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void add_bias_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N,
    const int C,
    const int D,
    const int H,
    const int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * D * H * W;
    if (idx < total) {
        // Compute the (n, c, d, h, w) index from the flattened index
        int w = idx % W;
        int h = (idx / W) % H;
        int d = (idx / (W * H)) % D;
        int c = (idx / (D * H * W)) % C;
        // n = idx / (C * D * H * W) // not used separately below

        float val = input[idx] + bias[c];
        // Apply ReLU
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor add_bias_relu_cuda(
    torch::Tensor input,
    torch::Tensor bias
) {
    // Expect input as (N, C, D, H, W)
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto output = torch::zeros_like(input);

    int total = N * C * D * H * W;
    const int blockSize = 256;
    const int gridSize = (total + blockSize - 1) / blockSize;

    add_bias_relu_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W
    );

    return output;
}
"""

add_bias_relu_cpp_src = r"""
torch::Tensor add_bias_relu_cuda(
    torch::Tensor input,
    torch::Tensor bias
);
"""

# Compile the inline CUDA code
fused_add_bias_relu = load_inline(
    name="fused_add_bias_relu",
    cpp_sources=add_bias_relu_cpp_src,
    cuda_sources=add_bias_relu_cuda_src,
    functions=["add_bias_relu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized version of the 3D transposed convolution model with a fused
    custom CUDA kernel that applies bias and ReLU after the convolution.
    """
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
        super(ModelNew, self).__init__()
        # Use bias=False in ConvTranspose3d so we can handle bias ourselves
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=False
        )
        self.use_bias = bias
        if bias:
            # Manually register a learnable bias parameter
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias_param = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_transpose3d(x)
        if self.use_bias and self.bias_param is not None:
            # Apply fused add-bias + ReLU
            out = fused_add_bias_relu.add_bias_relu_cuda(out, self.bias_param)
        return out
