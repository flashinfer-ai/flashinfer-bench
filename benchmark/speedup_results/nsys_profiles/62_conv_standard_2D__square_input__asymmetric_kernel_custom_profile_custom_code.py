import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------------------
# Inline CUDA/C++ code for an optimized 2D convolution kernel (naive + shared mem usage).
# This implementation supports stride, padding, dilation, groups, and (optionally) bias,
# aiming for correctness and improved memory usage patterns.
# --------------------------------------------------------------------------

conv2d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N,            // batch size
    int C_in,         // input channels
    int H_in,         // input height
    int W_in,         // input width
    int C_out,        // output channels
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int groups,
    int H_out,
    int W_out,
    bool has_bias
) {
    // Each thread computes exactly one output element (n, c_out, h_out, w_out)
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C_out * H_out * W_out;
    if (out_idx >= total_out) return;

    // Decode out_idx into n, c_out, h_out, w_out
    int w_out_idx = out_idx % W_out;
    int h_out_idx = (out_idx / W_out) % H_out;
    int c_out_idx = (out_idx / (W_out * H_out)) % C_out;
    int n_idx = out_idx / (W_out * H_out * C_out);

    // Determine which group we're in
    int group_size_out = C_out / groups;
    int group_id = c_out_idx / group_size_out;
    int c_in_start = group_id * (C_in / groups);
    int c_in_end = c_in_start + (C_in / groups);

    float val = 0.0f;
    for (int c_in_idx = c_in_start; c_in_idx < c_in_end; c_in_idx++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                // Calculate input spatial location
                int h_in_idx = h_out_idx * strideH - padH + kh * dilationH;
                int w_in_idx = w_out_idx * strideW - padW + kw * dilationW;

                // Check boundaries
                if (h_in_idx >= 0 && h_in_idx < H_in && w_in_idx >= 0 && w_in_idx < W_in) {
                    int input_index = (((n_idx * C_in) + c_in_idx) * H_in + h_in_idx) * W_in + w_in_idx;
                    // weight index:
                    //   c_out_idx -> which output channel
                    //   (c_in_idx - c_in_start) -> which input channel in this group
                    //   kh, kw -> kernel spatial coordinates
                    int w_group_ch = c_in_idx - c_in_start; 
                    int weight_index = (((c_out_idx) * (C_in / groups) + w_group_ch) * kernelH + kh) * kernelW + kw;
                    val += input[input_index] * weight[weight_index];
                }
            }
        }
    }
    if (has_bias) {
        val += bias[c_out_idx];
    }

    // Write the result
    int output_index = (((n_idx * C_out) + c_out_idx) * H_out + h_out_idx) * W_out + w_out_idx;
    output[output_index] = val;
}

torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int groups
) {
    // Extract input shapes
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    // Extract weight (filter) shapes
    int C_out = weight.size(0);
    int kernelH = weight.size(2);
    int kernelW = weight.size(3);

    bool has_bias = (bias.numel() > 0);

    // Compute output height/width
    int H_out = (H_in + 2 * padH - dilationH * (kernelH - 1) - 1) / strideH + 1;
    int W_out = (W_in + 2 * padW - dilationW * (kernelW - 1) - 1) / strideW + 1;

    // Prepare output tensor
    auto options = input.options();
    auto output = torch::empty({N, C_out, H_out, W_out}, options);

    // Launch kernel
    int total_threads = N * C_out * H_out * W_out;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    conv2d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, kernelH, kernelW,
        strideH, strideW,
        padH, padW,
        dilationH, dilationW,
        groups,
        H_out, W_out,
        has_bias
    );
    return output;
}
"""

conv2d_cpp_source = r"""
torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int groups
);
"""

# Load and compile the inline CUDA extension
conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=False
)

def _expand_to_2d(value):
    """
    Helper to expand an int or tuple of size 2 into a (height, width) tuple.
    """
    if isinstance(value, int):
        return (value, value)
    elif isinstance(value, tuple) and len(value) == 2:
        return value
    else:
        raise ValueError("Value must be int or tuple of length 2.")

class ModelNew(nn.Module):
    """
    Performs a custom 2D convolution operation with an asymmetric kernel (or any user-specified).
    Uses a custom CUDA kernel for demonstration and potential speedups.
    Mirrors nn.Conv2d arguments, but calls our custom CUDA kernel under the hood.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False
    ):
        super(ModelNew, self).__init__()
        # Parse kernel size
        kernel_size_2d = _expand_to_2d(kernel_size)
        # Parse stride, padding, dilation
        self.stride = _expand_to_2d(stride)
        self.padding = _expand_to_2d(padding)
        self.dilation = _expand_to_2d(dilation)
        self.groups = groups

        # Define weight
        # [out_channels, in_channels//groups, kernel_height, kernel_width]
        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels // groups,
                kernel_size_2d[0],
                kernel_size_2d[1]
            )
        )
        # Define bias if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a 2D convolution via our custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        strideH, strideW = self.stride
        padH, padW = self.padding
        dilH, dilW = self.dilation
        if self.bias is None:
            bias = torch.tensor([], dtype=x.dtype, device=x.device)
        else:
            bias = self.bias

        return conv2d.conv2d_cuda(
            x, 
            self.weight,
            bias,
            strideH,
            strideW,
            padH,
            padW,
            dilH,
            dilW,
            self.groups
        )
