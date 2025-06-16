import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

transposed_conv2d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void transposed_conv2d_kernel(
    const float* input,        // [N, C_in, H_in, W_in]
    const float* weight,       // [C_in, C_out, KH, KW]
    const float* bias,         // [C_out] (optional if bias is false)
    float* output,             // [N, C_out, H_out, W_out]
    const int N,
    const int C_in,
    const int C_out,
    const int H_in,
    const int W_in,
    const int KH,
    const int KW,
    const int H_out,
    const int W_out,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int out_pad_h,
    const int out_pad_w,
    const bool use_bias
) {
    // 2D indices within output feature map
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    // Combine batch index and out_channel index into blockIdx.z
    int n_cout = blockIdx.z;
    if (ow >= W_out || oh >= H_out || n_cout >= N * C_out) {
        return;
    }
    // Decode batch index and output channel index
    int n = n_cout / C_out;
    int cout = n_cout % C_out;

    float value = 0.0f;
    // Loop over input channels and kernel elements
    for (int cin = 0; cin < C_in; cin++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw_ = 0; kw_ < KW; kw_++) {
                // Compute the corresponding input spatial location
                int in_h = oh + pad_h - kh;
                int in_w = ow + pad_w - kw_;

                // Check stride alignment
                if ((in_h % stride_h == 0) && (in_w % stride_w == 0)) {
                    int real_in_h = in_h / stride_h;
                    int real_in_w = in_w / stride_w;
                    // Check bounds
                    if (real_in_h >= 0 && real_in_h < H_in &&
                        real_in_w >= 0 && real_in_w < W_in) {
                        int in_offset = ((n * C_in + cin) * H_in + real_in_h) * W_in + real_in_w;
                        int w_offset = ((cin * C_out + cout) * KH + kh) * KW + kw_;
                        value += input[in_offset] * weight[w_offset];
                    }
                }
            }
        }
    }
    if (use_bias) {
        value += bias[cout];
    }
    // Write result
    int out_offset = ((n * C_out + cout) * H_out + oh) * W_out + ow;
    output[out_offset] = value;
}

torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int out_pad_h,
    int out_pad_w
) {
    // Input shape:  [N, C_in, H_in, W_in]
    // Weight shape: [C_in, C_out, KH, KW]
    // Bias shape:   [C_out] or None

    auto N      = input.size(0);
    auto C_in   = input.size(1);
    auto H_in   = input.size(2);
    auto W_in   = input.size(3);

    auto C_out  = weight.size(1);
    auto KH     = weight.size(2);
    auto KW     = weight.size(3);

    // Compute output spatial dimensions
    // TransposedConv2d output formula for each dimension:
    // H_out = (H_in - 1)*stride_h - 2*pad_h + (KH) + out_pad_h
    // W_out = (W_in - 1)*stride_w - 2*pad_w + (KW) + out_pad_w
    int H_out = (H_in - 1) * stride_h - 2 * pad_h + KH + out_pad_h;
    int W_out = (W_in - 1) * stride_w - 2 * pad_w + KW + out_pad_w;

    // Create output tensor
    auto out = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Launch kernel
    const int block_dim_x = 16;
    const int block_dim_y = 16;

    dim3 block(block_dim_x, block_dim_y);
    dim3 grid(
        (W_out + block_dim_x - 1) / block_dim_x,
        (H_out + block_dim_y - 1) / block_dim_y,
        N * C_out
    );

    transposed_conv2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        N,
        C_in,
        C_out,
        H_in,
        W_in,
        KH,
        KW,
        H_out,
        W_out,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        out_pad_h,
        out_pad_w,
        bias.has_value()
    );

    return out;
}
"""

transposed_conv2d_cpp_source = r"""
torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int out_pad_h,
    int out_pad_w
);
"""

transposed_conv2d = load_inline(
    name="transposed_conv2d",
    cpp_sources=transposed_conv2d_cpp_source,
    cuda_sources=transposed_conv2d_source,
    functions=["transposed_conv2d_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized transposed 2D convolution with a custom CUDA kernel.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,   # not used in this custom kernel
        bias: bool = False
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)

        # Weight shape for ConvTranspose2d in PyTorch: [in_channels, out_channels, kernel_h, kernel_w]
        self.weight = nn.Parameter(torch.randn(
            in_channels,
            out_channels,
            kernel_size[0],
            kernel_size[1]
        ))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_h, pad_w = self.padding
        stride_h, stride_w = self.stride
        out_pad_h, out_pad_w = self.output_padding
        return transposed_conv2d.transposed_conv2d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else None,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_pad_h,
            out_pad_w
        )
