import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose_3d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Macros for checking tensor properties
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

// A naive transposed 3D convolution kernel with support for groups, stride, padding, and output padding.
// This kernel accumulates contributions from each input pixel/voxel via atomicAdds in output memory.
// For improved performance, additional optimizations such as tiling, shared memory usage, and 
// better memory coalescing patterns can be applied.

__global__ void convTranspose3dKernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N,
    const int Cin,
    const int D_in,
    const int H_in,
    const int W_in,
    const int Cout,
    const int KD,
    const int KH,
    const int KW,
    const int strideD,
    const int strideH,
    const int strideW,
    const int padD,
    const int padH,
    const int padW,
    const int outPadD,
    const int outPadH,
    const int outPadW,
    const int groups,
    const bool has_bias,
    const int D_out,
    const int H_out,
    const int W_out)
{
    // Linear global index for each input element n, c_in, d_in, h_in, w_in
    // index in [0, N*Cin*D_in*H_in*W_in)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_input_size = N * Cin * D_in * H_in * W_in;
    if (idx >= total_input_size) {
        return;
    }

    // Decode (n, c_in, d_in, h_in, w_in) from idx
    int w_in_idx = idx % W_in;
    int tmp1 = idx / W_in;
    int h_in_idx = tmp1 % H_in;
    int tmp2 = tmp1 / H_in;
    int d_in_idx = tmp2 % D_in;
    int tmp3 = tmp2 / D_in;
    int c_in_idx = tmp3 % Cin;
    int n_idx = tmp3 / Cin;

    // Determine which group this input channel belongs to
    // group_size_in is Cin/groups, group_size_out is Cout/groups
    int group_size_in = Cin / groups;
    int group_size_out = Cout / groups;
    int g = c_in_idx / group_size_in; // group index
    int c_in_local = c_in_idx % group_size_in; // local index within the group

    // Read the input value
    float val_in = input[idx];

    // For each output channel in the same group
    int c_out_start = g * group_size_out;
    int c_out_end   = (g + 1) * group_size_out;

    // For each kernel depth/height/width
    for (int kd = 0; kd < KD; kd++) {
        // The output depth index corresponding to d_in_idx
        int d_out = d_in_idx * strideD - padD + kd;
        if (d_out < 0 || d_out >= D_out) {
            continue;
        }

        for (int kh = 0; kh < KH; kh++) {
            int h_out = h_in_idx * strideH - padH + kh;
            if (h_out < 0 || h_out >= H_out) {
                continue;
            }

            for (int kw = 0; kw < KW; kw++) {
                int w_out = w_in_idx * strideW - padW + kw;
                if (w_out < 0 || w_out >= W_out) {
                    continue;
                }

                // For each output channel
                for (int c_out = c_out_start; c_out < c_out_end; c_out++) {
                    // Compute weight index: weight shape = [Cout, Cin/groups, KD, KH, KW]
                    // local c_in index = c_in_local, global c_out index = c_out
                    int weight_idx = c_out * (group_size_in * KD * KH * KW)
                                   + c_in_local * (KD * KH * KW)
                                   + kd * (KH * KW)
                                   + kh * KW
                                   + kw;

                    float w_val = weight[weight_idx];

                    // Output index: [N, Cout, D_out, H_out, W_out]
                    // linearize: n_idx*(Cout*D_out*H_out*W_out) + c_out*(D_out*H_out*W_out) + ...
                    long out_idx = ((long)n_idx * Cout + c_out) * (D_out * H_out * W_out)
                                 + (long)d_out * (H_out * W_out)
                                 + (long)h_out * W_out
                                 + w_out;

                    atomicAdd(&output[out_idx], val_in * w_val);
                }
            }
        }
    }
}

// Host code that wraps the CUDA kernel
torch::Tensor conv_transpose_3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int outPadD, int outPadH, int outPadW,
    int groups)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    // bias can be an empty tensor if has_bias = false
    if (bias.defined()) {
       CHECK_CUDA(bias);
       CHECK_CONTIGUOUS(bias);
    }

    const bool has_bias = bias.defined() && bias.numel() > 0;

    // Input shape: [N, Cin, D_in, H_in, W_in]
    int N    = input.size(0);
    int Cin  = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    // Weight shape: [Cout, Cin/groups, KD, KH, KW]
    int Cout = weight.size(0);
    int KD   = weight.size(2);
    int KH   = weight.size(3);
    int KW   = weight.size(4);

    // Compute expected output size (assuming dilation=1 for simplicity)
    // outDim = (inDim - 1)*stride - 2*pad + (kDim) + outPad
    int D_out = (D_in - 1) * strideD - 2 * padD + KD + outPadD;
    int H_out = (H_in - 1) * strideH - 2 * padH + KH + outPadH;
    int W_out = (W_in - 1) * strideW - 2 * padW + KW + outPadW;

    // Allocate output tensor
    auto output = torch::zeros({N, Cout, D_out, H_out, W_out}, input.options());

    // Launch kernel
    const int threads = 256;
    const int total_input_size = N * Cin * D_in * H_in * W_in;
    const int blocks = (total_input_size + threads - 1) / threads;

    convTranspose3dKernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, Cin, D_in, H_in, W_in,
        Cout, KD, KH, KW,
        strideD, strideH, strideW,
        padD, padH, padW,
        outPadD, outPadH, outPadW,
        groups,
        has_bias,
        D_out, H_out, W_out
    );

    // Add bias if present (outside atomicAdd loops to avoid overhead)
    if (has_bias) {
        // bias shape: [Cout]
        // broadcast over [N, Cout, D_out, H_out, W_out]
        // We can launch a separate kernel or do in-place add:
        // out[n, c, d, h, w] += bias[c]
        // for better performance, a dedicated kernel can be used.
        auto bias_view = bias.view({1, Cout, 1, 1, 1});
        output += bias_view;
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose_3d_cuda", &conv_transpose_3d_cuda, "Naive Transposed 3D Convolution");
}
""".strip()

conv_transpose_3d_cpp_source = r"""
torch::Tensor conv_transpose_3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int outPadD, int outPadH, int outPadW,
    int groups);
""".strip()

conv_transpose_3d = load_inline(
    name="conv_transpose_3d",
    cpp_sources=conv_transpose_3d_cpp_source,
    cuda_sources=conv_transpose_3d_source,
    functions=["conv_transpose_3d_cuda"],
    extra_cflags=["-O3", "-DTORCH_USE_CUDA_DSA"],  # Enable device-side assertions for debugging
    extra_ldflags=[],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Custom transposed 3D convolution module with a naive CUDA kernel.
    Supports groups, stride, padding, and output padding. Dilation is assumed 1.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple=(1,1,1), padding: tuple=(0,0,0),
                 output_padding: tuple=(0,0,0), groups: int=1, bias: bool=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias

        # Register parameters for weight and optional bias
        # Weight shape: [OutChannels, InChannels/groups, kD, kH, kW]
        self.weight = nn.Parameter(torch.empty(
            out_channels,
            in_channels // groups,
            kernel_size[0],
            kernel_size[1],
            kernel_size[2]
        ))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose_3d.conv_transpose_3d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.tensor([]).to(x.device),
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.output_padding[0],
            self.output_padding[1],
            self.output_padding[2],
            self.groups
        )
