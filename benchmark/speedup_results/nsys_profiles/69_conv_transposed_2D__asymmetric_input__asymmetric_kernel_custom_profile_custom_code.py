import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

/*
  A simplified custom transposed convolution kernel supporting:
  - groups=1
  - user-provided stride/padding/output_padding/dilation
  - optional bias

  The weight is assumed to have shape [inC, outC, kH, kW].
  The input shape is [N, inC, inH, inW].
  The output shape is [N, outC, outH, outW], computed in Python.
*/

__global__ void conv2d_transpose_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int N,
    const int inC,
    const int inH,
    const int inW,
    const int outC,
    const int outH,
    const int outW,
    const int kH,
    const int kW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW,
    const int outputPadH,
    const int outputPadW,
    const bool has_bias)
{
    // Each thread computes one element in the output: (n, oc, oh, ow)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * outC * outH * outW;
    if (idx >= total_threads) {
        return;
    }

    // Decompose idx into n, oc, oh, ow
    int ow = idx % outW;
    int temp = idx / outW;
    int oh = temp % outH;
    temp = temp / outH;
    int oc = temp % outC;
    int n = temp / outC;

    // Accumulate convolution
    float val = 0.0f;
    for(int ic = 0; ic < inC; ic++) {
        // Go through each element of the kernel
        for(int kh = 0; kh < kH; kh++) {
            // Compute the corresponding input h index
            int ih = (oh + padH - kh * dilationH) / strideH;
            // Check if divisible (to ensure transpose alignment)
            if(((oh + padH - kh * dilationH) % strideH) != 0) {
                continue;
            }
            if (ih < 0 || ih >= inH) {
                continue;
            }
            for(int kw = 0; kw < kW; kw++) {
                int iw = (ow + padW - kw * dilationW) / strideW;
                if(((ow + padW - kw * dilationW) % strideW) != 0) {
                    continue;
                }
                if (iw < 0 || iw >= inW) {
                    continue;
                }
                // Indices in input/weight
                int in_idx = ((n * inC + ic) * inH + ih) * inW + iw;
                int w_idx  = ((ic * outC + oc) * kH + kh) * kW + kw;
                val += input[in_idx] * weight[w_idx];
            }
        }
    }

    if (has_bias) {
        val += bias[oc];
    }

    // Store result
    int out_idx = ((n * outC + oc) * outH + oh) * outW + ow;
    output[out_idx] = val;
}

torch::Tensor conv2d_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int outputPadH,
    int outputPadW)
{
    // Shapes
    int N = input.size(0);
    int inC = input.size(1);
    int inH = input.size(2);
    int inW = input.size(3);

    int outC = weight.size(1); // weight shape [inC, outC, kH, kW]
    int kH = weight.size(2);
    int kW = weight.size(3);

    // Compute output shape manually
    // outH = (inH - 1)*strideH - 2*padH + (kH - 1)*dilationH + outputPadH + 1
    // outW = (inW - 1)*strideW - 2*padW + (kW - 1)*dilationW + outputPadW + 1
    int outH = (inH - 1) * strideH - 2 * padH + (kH - 1) * dilationH + outputPadH + 1;
    int outW = (inW - 1) * strideW - 2 * padW + (kW - 1) * dilationW + outputPadW + 1;

    auto out_options = input.options().dtype(input.dtype());
    auto output = torch::empty({N, outC, outH, outW}, out_options);

    int total_threads = N * outC * outH * outW;
    int block = 256;
    int grid = (total_threads + block - 1) / block;

    bool has_bias = bias.numel() > 0 ? true : false;

    conv2d_transpose_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N,
        inC,
        inH,
        inW,
        outC,
        outH,
        outW,
        kH,
        kW,
        strideH,
        strideW,
        padH,
        padW,
        dilationH,
        dilationW,
        outputPadH,
        outputPadW,
        has_bias
    );

    return output;
}
""";

conv_transpose2d_cpp_source = r"""
torch::Tensor conv2d_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int outputPadH,
    int outputPadW);
""";

# Build and load the custom CUDA extension
conv_transpose2d_mod = load_inline(
    name="conv_transpose2d_mod",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv2d_transpose_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Custom implementation of a transposed 2D convolution using a CUDA kernel.
    Mirrors the constructor of nn.ConvTranspose2d but overrides forward with
    our custom kernel.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        output_padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,  # Not used in this custom kernel, must be 1
        bias: bool = False
    ):
        super().__init__()
        assert groups == 1, "Custom kernel only supports groups=1 currently."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.bias_flag = bias

        # Weight shape for transposed conv: [in_channels, out_channels, kH, kW]
        kH, kW = kernel_size
        w = torch.empty(in_channels, out_channels, kH, kW)
        nn.init.kaiming_uniform_(w, a=5**0.5)
        self.weight = nn.Parameter(w)

        if bias:
            b = torch.empty(out_channels)
            nn.init.constant_(b, 0.0)
            self.bias = nn.Parameter(b)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose2d_mod.conv2d_transpose_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.tensor([]).to(x.device),
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.output_padding[0],
            self.output_padding[1]
        )
