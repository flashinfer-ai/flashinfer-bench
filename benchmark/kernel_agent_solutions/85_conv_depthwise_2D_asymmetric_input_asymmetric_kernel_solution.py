import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv2d_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Depthwise 2D Convolution Kernel
// Assumes out_channels == in_channels and groups == in_channels for depthwise behavior.
// Weight shape: (in_channels, 1, kernel_h, kernel_w)
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    bool has_bias)
{
    // Compute the output spatial dimensions
    int H_out = (H + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int W_out = (W + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    // Total number of output elements
    int total_outputs = B * C * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) {
        return;
    }

    // Decompose linear index into (b, c, oh, ow)
    int ow = idx % W_out;
    int oh = (idx / W_out) % H_out;
    int c_ = (idx / (W_out * H_out)) % C;
    int b_ = idx / (W_out * H_out * C);

    // Initialize accumulator with bias if present
    float out_val = 0.0f;
    if (has_bias) {
        out_val = bias[c_];
    }

    // Offset for the weights of this channel
    const int weight_offset = c_ * kernel_h * kernel_w;

    // Perform the depthwise convolution (naive loop)
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw_ = 0; kw_ < kernel_w; ++kw_) {
            int ih = oh * stride_h - pad_h + kh * dilation_h;
            int iw = ow * stride_w - pad_w + kw_ * dilation_w;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                float in_val =
                    input[((b_ * C + c_) * H + ih) * W + iw];
                float w_val =
                    weight[weight_offset + kh * kernel_w + kw_];
                out_val += in_val * w_val;
            }
        }
    }
    output[idx] = out_val;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    bool has_bias)
{
    // Input shape: (B, C, H, W)
    // Weight shape: (C, 1, kernel_h, kernel_w)
    // Bias shape: (C,)
    const auto B = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    // Calculate output size
    const auto H_out = (H + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const auto W_out = (W + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    // Create output tensor
    auto output = torch::zeros({B, C, H_out, W_out}, input.options());

    // Launch kernel
    int threads = 256;
    int total_outputs = B * C * H_out * W_out;
    int blocks = (total_outputs + threads - 1) / threads;

    depthwise_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, C, H, W,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        has_bias
    );

    return output;
}
''';

depthwise_conv2d_cpp_source = r'''
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    bool has_bias);
''';

depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Depthwise 2D Convolution with a custom CUDA kernel.
    Matches the original constructor signature.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_h: int,
        kernel_size_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        bias: bool = False
    ):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.use_bias = bias

        # Initialize weight for depthwise convolution
        # PyTorch depthwise: weight shape [in_channels, 1, kernel_size_h, kernel_size_w]
        self.weight = nn.Parameter(
            torch.randn(in_channels, 1, kernel_size_h, kernel_size_w)
        )
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using custom CUDA depthwise conv kernel.
        """
        return depthwise_conv2d.depthwise_conv2d_cuda(
            x,
            self.weight,
            self.bias if self.use_bias else torch.zeros(1, device=x.device),
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
            self.use_bias
        )
