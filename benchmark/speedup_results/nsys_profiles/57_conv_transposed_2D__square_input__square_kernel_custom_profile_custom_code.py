import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

transposed_conv2d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// A refined reference kernel for 2D transposed convolution (ConvTranspose2d).
// Correctness-focused, with improved indexing for PyTorch ConvTranspose2d layout:
//   weight shape: [in_channels, out_channels, kernel_size, kernel_size].
// Output shape is computed on the Python side and passed in.

__global__ void transposed_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,             // batch_size
    int C_in,          // input channels
    int C_out,         // output channels
    int H_in,          // input height
    int W_in,          // input width
    int H_out,         // output height
    int W_out,         // output width
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    bool has_bias
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C_out * H_out * W_out;
    if (index >= total_threads) return;

    // Decode n, oc, oh, ow from index
    int w_out = index % W_out;
    int tmp = index / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int oc = tmp % C_out;
    int n = tmp / C_out;

    float val = 0.0f;

    // Accumulate over all input channels and kernel positions
    for (int ic = 0; ic < C_in; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Corresponding input position (reverse of forward conv stride/pad logic)
                int h_in = h_out + padding - kh;
                int w_in = w_out + padding - kw;

                if ((h_in % stride) == 0 && (w_in % stride) == 0) {
                    h_in /= stride;
                    w_in /= stride;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int input_idx = ((n * C_in + ic) * H_in + h_in) * W_in + w_in;
                        // weight layout: [ic, oc, kh, kw]
                        int weight_idx = ((ic * C_out + oc) * kernel_size + kh) * kernel_size + kw;
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    if (has_bias) {
        val += bias[oc];
    }

    int out_idx = ((n * C_out + oc) * H_out + h_out) * W_out + w_out;
    output[out_idx] = val;
}

torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding
) {
    // input  shape: [N, C_in, H_in, W_in]
    // weight shape: [C_in, C_out, kH, kW]
    // bias   shape: [C_out] or empty
    // Compute output shape and allocate out tensor
    auto N      = input.size(0);
    auto C_in   = input.size(1);
    auto H_in   = input.size(2);
    auto W_in   = input.size(3);
    auto C_out  = weight.size(1);
    auto kH     = weight.size(2);
    auto kW     = weight.size(3);

    // Output shape calculation for transposed conv
    int H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding;

    auto out_options = input.options();
    auto out = torch::empty({N, C_out, H_out, W_out}, out_options);

    int total_count = N * C_out * H_out * W_out;

    const int block_size = 256;
    int grid_size = (total_count + block_size - 1) / block_size;

    bool has_bias = bias.numel() == C_out;

    transposed_conv2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, H_out, W_out,
        kH,
        stride,
        padding,
        output_padding,
        has_bias
    );

    return out;
}
"""

transposed_conv2d_cpp_source = r"""
torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding
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
    A custom-transposed-convolution module using a refined CUDA kernel for correctness. 
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
        super().__init__()
        # Store parameters needed for transposed conv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Register weight and bias as parameters
        # Weight shape for ConvTranspose2d: [in_channels, out_channels, kernel_size, kernel_size]
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return transposed_conv2d.transposed_conv2d_cuda(
            x, 
            self.weight,
            self.bias if self.bias is not None else torch.tensor([], device=x.device, dtype=x.dtype),
            self.stride,
            self.padding,
            self.output_padding
        )
