import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv3d_code = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv3d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W, int D,
    int outC,
    int kernel_size,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dil_h,
    int dil_w,
    bool has_bias
) {
    // Calculate a flattened index for the (out_h, out_w) plane
    int out_hw = blockIdx.x * blockDim.x + threadIdx.x;
    // out_c is determined by the second dimension of the grid
    int out_c = blockIdx.y;
    // n is determined by the third dimension of the grid
    int n = blockIdx.z;

    // Early return if out_hw is out of bounds
    if (out_hw >= H * W) {
        return;
    }

    // Compute out_h and out_w from the flattened index
    int out_h = out_hw / W;
    int out_w = out_hw % W;

    // Calculate output spatial dimensions
    int outH = (H + 2 * pad_h - (kernel_size - 1) * dil_h - 1) / stride_h + 1;
    int outW = (W + 2 * pad_w - (kernel_size - 1) * dil_w - 1) / stride_w + 1;

    // Discard if outside valid output range
    if (out_h >= outH || out_w >= outW) {
        return;
    }

    for (int d_idx = 0; d_idx < D; d_idx++) {
        float val = 0.0f;

        // Compute the output index in NHWDC format
        int out_index = n * outC * outH * outW * D
                      + out_c * outH * outW * D
                      + out_h * outW * D
                      + out_w * D
                      + d_idx;

        // Accumulate over kernel
        for (int c_in = 0; c_in < C; c_in++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_h = out_h * stride_h - pad_h + kh * dil_h;
                    int in_w = out_w * stride_w - pad_w + kw * dil_w;

                    // Depth is unchanged because kernel depth = 1
                    int in_d = d_idx;

                    // Check boundaries
                    if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                        int in_index = n * C * H * W * D
                                     + c_in * H * W * D
                                     + in_h * W * D
                                     + in_w * D
                                     + in_d;

                        int w_index = out_c * C * kernel_size * kernel_size * 1
                                    + c_in * kernel_size * kernel_size * 1
                                    + kh * kernel_size * 1
                                    + kw * 1;

                        val += input[in_index] * weight[w_index];
                    }
                }
            }
        }

        // Add bias if applicable
        if (has_bias) {
            val += bias[out_c];
        }

        // Write out
        output[out_index] = val;
    }
}

torch::Tensor conv3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dil_h,
    int dil_w
) {
    // Extract input dimensions
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int D = input.size(4);

    // Weight shape: (outC, inC, kH, kW, 1)
    int outC = weight.size(0);
    int kernel_size = weight.size(2);

    bool has_bias = (bias.defined() && bias.numel() > 0);

    int outH = (H + 2 * pad_h - (kernel_size - 1) * dil_h - 1) / stride_h + 1;
    int outW = (W + 2 * pad_w - (kernel_size - 1) * dil_w - 1) / stride_w + 1;

    auto options = input.options();
    auto output = torch::zeros({N, outC, outH, outW, D}, options);

    // Launch config
    dim3 block(256);
    dim3 grid((H * W + block.x - 1) / block.x, outC, N);

    conv3d_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, H, W, D,
        outC,
        kernel_size,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        has_bias
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_forward_cuda", &conv3d_forward_cuda, "Custom 3D Convolution forward");
}
"""

conv3d_module = load_inline(
    name="custom_conv3d",
    cpp_sources=conv3d_code,
    functions=["conv3d_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Custom 3D Convolution with a square kernel (kernel_size x kernel_size, depth=1),
    plus optional bias.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups  # Unused in this custom kernel
        self.use_bias = bias

        # Weight shape: (out_channels, in_channels, kernel_size, kernel_size, 1)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv3d_module.conv3d_forward_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.tensor([], device=x.device, dtype=x.dtype),
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation
        )
