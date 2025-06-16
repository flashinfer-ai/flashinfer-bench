import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv2d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Naive depthwise convolution for kernel_size=(kernel_size,1) with optional bias.
// input:  (N, C, H, W)
// weight: (C, 1, kernel_size, 1)
// bias:   (C) or empty tensor if no bias
// Returns output: (N, C, H_out, W_out)

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int kernel_size, int stride, int padding, int dilation,
    int outH, int outW,
    bool has_bias
) {
    // Each thread corresponds to one element in output: (n, c, oh, ow)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N*C*outH*outW) {
        return;
    }
    
    // Decode idx into n, c, oh, ow
    int ow = idx % outW;
    int temp = idx / outW;
    int oh = temp % outH;
    temp /= outH;
    int c = temp % C;
    int n = temp / C;

    // Compute the input coordinates
    float val = 0.0f;
    // We only have kernel_size along H dimension, and 1 along W dimension
    for(int kh = 0; kh < kernel_size; kh++){
        int ih = oh * stride - padding + kh * dilation;
        int iw = ow;  // since kernel width is 1, no shift in W
        if (ih >= 0 && ih < H && iw >= 0 && iw < W){
            val += input[((n*C + c)*H + ih)*W + iw] * weight[((c*1 + 0)*kernel_size + kh)*1 + 0];
        }
    }
    if (has_bias) {
        val += bias[c];
    }

    output[((n*C + c)*outH + oh)*outW + ow] = val;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Shapes and sizes
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    // Output size
    // outH = floor((H + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    // outW = floor((W + 2*padding - dilation*(1-1) - 1)/stride + 1) = floor((W + 2*padding - 1)/stride + 1)
    const int outH = (H + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1;
    const int outW = (W + 2*padding - dilation*(1 - 1) - 1) / stride + 1;

    TORCH_CHECK(outH > 0 && outW > 0, "Invalid output shape computed, check kernel/stride/padding/dilation.");

    auto options = input.options().dtype(input.dtype());
    torch::Tensor output = torch::zeros({N, C, outH, outW}, options);

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    // Launch kernel
    int totalThreads = N * C * outH * outW;
    int blockSize = 256;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;

    depthwise_conv2d_kernel<<<gridSize, blockSize>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        N, C, H, W,
        kernel_size, stride, padding, dilation,
        outH, outW,
        (bias.numel() > 0)
    );

    return output;
}
""";

depthwise_conv2d_cpp_source = r"""
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);
"""

# Compile the inline CUDA code for depthwise convolution
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=False
)


class ModelNew(nn.Module):
    """
    Optimized version of the depthwise 2D convolution with (kernel_size, 1) using a custom CUDA kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias

        # Weight shape: (in_channels, 1, kernel_size, 1)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter("bias", None)

        self.depthwise_conv2d = depthwise_conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise_conv2d.depthwise_conv2d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )


def get_inputs():
    # Example shape: (batch_size=16, in_channels=3, height=256, width=256)
    batch_size = 16
    in_channels = 3
    height = 256
    width = 256
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    in_channels = 3
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    return [in_channels, kernel_size, stride, padding, dilation]
