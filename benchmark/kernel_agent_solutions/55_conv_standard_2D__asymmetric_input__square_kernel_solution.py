import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

naive_conv2d_cpp_source = r"""
torch::Tensor naive_conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int groups);
"""

naive_conv2d_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

static __device__ float get_value(const float* data,
                                  int b, int c, int h, int w,
                                  int C, int H, int W) {
    // Returns 0 if out of bounds.
    if (h < 0 || h >= H || w < 0 || w >= W)
        return 0.0f;
    return data[ ((b * C + c) * H + h) * W + w ];
}

__global__ void naive_conv2d_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int batch_size, 
                                    int in_channels,
                                    int in_height,
                                    int in_width,
                                    int out_channels,
                                    int out_height,
                                    int out_width,
                                    int kernel_h,
                                    int kernel_w,
                                    int strideH,
                                    int strideW,
                                    int padH,
                                    int padW,
                                    int dilationH,
                                    int dilationW,
                                    int groups)
{
    // 3D grid: 
    //   blockIdx.x -> out_width
    //   blockIdx.y -> out_height
    //   blockIdx.z -> out_channels * batch_size
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tmp_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_x >= out_width || out_y >= out_height || tmp_z >= out_channels * batch_size) {
        return;
    }
    // Decode batch and out_channel from tmp_z
    int b = tmp_z / out_channels;
    int oc = tmp_z % out_channels;

    float out_val = 0.0f;
    if (bias != nullptr) {
        out_val = bias[oc];
    }

    // Each group has (in_channels / groups) input channels and (out_channels / groups) output channels
    int group_idx = oc / (out_channels / groups);
    int in_ch_begin = group_idx * (in_channels / groups);
    int in_ch_end   = (group_idx + 1) * (in_channels / groups);

    for (int ic = in_ch_begin; ic < in_ch_end; ic++) {
        for (int ky = 0; ky < kernel_h; ky++) {
            for (int kx = 0; kx < kernel_w; kx++) {
                int in_y = out_y * strideH - padH + ky * dilationH;
                int in_x = out_x * strideW - padW + kx * dilationW;
                float val = get_value(input, b, ic, in_y, in_x, in_channels, in_height, in_width);
                // weight shape: (out_channels, in_channels, kernel_h, kernel_w)
                float wval = weight[ ((oc * in_channels + ic) * kernel_h + ky) * kernel_w + kx ];
                out_val += val * wval;
            }
        }
    }

    // store result
    output[ ((b * out_channels + oc) * out_height + out_y) * out_width + out_x ] = out_val;
}

torch::Tensor naive_conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int groups)
{
    // input shape:  (batch_size, in_channels, in_height, in_width)
    // weight shape: (out_channels, in_channels, kernel_h, kernel_w)
    // bias shape:   (out_channels) or None
    int batch_size   = input.size(0);
    int in_channels  = input.size(1);
    int in_height    = input.size(2);
    int in_width     = input.size(3);

    int out_channels = weight.size(0);
    int kernel_h     = weight.size(2);
    int kernel_w     = weight.size(3);

    // compute output height/width
    int out_height = (in_height + 2 * padH - dilationH * (kernel_h - 1) - 1) / strideH + 1;
    int out_width  = (in_width + 2 * padW - dilationW * (kernel_w - 1) - 1) / strideW + 1;

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, options);

    // launch kernel
    const int threads_x = 8;
    const int threads_y = 8;
    const int threads_z = 1;
    dim3 blocks((out_width  + threads_x - 1) / threads_x,
                (out_height + threads_y - 1) / threads_y,
                (out_channels * batch_size + threads_z - 1) / threads_z);
    dim3 threads(threads_x, threads_y, threads_z);

    const float* input_ptr  = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr   = (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    naive_conv2d_kernel<<<blocks, threads>>>(input_ptr,
                                            weight_ptr,
                                            bias_ptr,
                                            output_ptr,
                                            batch_size,
                                            in_channels,
                                            in_height,
                                            in_width,
                                            out_channels,
                                            out_height,
                                            out_width,
                                            kernel_h,
                                            kernel_w,
                                            strideH,
                                            strideW,
                                            padH,
                                            padW,
                                            dilationH,
                                            dilationW,
                                            groups);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_conv2d_cuda_forward", &naive_conv2d_cuda_forward, "Naive Conv2D forward (CUDA)");
}
"""

# Build the extension
naive_conv2d = load_inline(
    name="my_naive_conv2d",
    cpp_sources=naive_conv2d_cpp_source,
    cuda_sources=naive_conv2d_cuda_source,
    functions=["naive_conv2d_cuda_forward"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Replaces the PyTorch Conv2d forward pass with a naive CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super().__init__()
        # Store parameters in a Conv2d so we can reuse weight/bias
        self.conv_ref = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, dilation=dilation,
                                  groups=groups, bias=bias)
        self.stride = stride if isinstance(stride, tuple) else stride
        self.padding = padding if isinstance(padding, tuple) else padding
        self.dilation = dilation if isinstance(dilation, tuple) else dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_ref.weight
        bias = self.conv_ref.bias if self.conv_ref.bias is not None else torch.tensor([], device=x.device)
        return naive_conv2d.naive_conv2d_cuda_forward(
            x,
            weight,
            bias,
            self.stride,
            self.stride,   # assume square stride for simplicity
            self.padding,
            self.padding,  # assume square padding
            self.dilation,
            self.dilation, # assume square dilation
            self.groups
        )
