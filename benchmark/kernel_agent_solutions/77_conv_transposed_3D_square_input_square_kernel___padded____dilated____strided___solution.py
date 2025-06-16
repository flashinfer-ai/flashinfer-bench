import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Utility to convert a linear index to 5D index: (n, oc, od, oh, ow)
__device__ __forceinline__ void offset_to_indices(
    int index, 
    int& n, int& oc, int& od, int& oh, int& ow,
    int batch_size, int out_channels, int out_depth, int out_height, int out_width)
{
    ow = index % out_width;
    index /= out_width;
    oh = index % out_height;
    index /= out_height;
    od = index % out_depth;
    index /= out_depth;
    oc = index % out_channels;
    index /= out_channels;
    n = index;
}

// Kernel for 3D transposed convolution (naive implementation).
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation,
    const bool has_bias)
{
    // Total number of output elements.
    int nthreads = batch_size * out_channels * out_depth * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nthreads) return;

    // Compute (n, oc, od, oh, ow) from thread index
    int n, oc, od, oh, ow;
    offset_to_indices(
        idx, n, oc, od, oh, ow,
        batch_size, out_channels, out_depth, out_height, out_width);

    // Accumulate result for this output position
    float value = 0.0f;
    // If there's a bias term, start from bias
    if (has_bias) {
        value = bias[oc];
    }

    // Loop over input channels and kernel positions
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            // Compute the corresponding input depth index
            int in_d = od + padding - kd * dilation;
            if (in_d % stride != 0) {
                continue;
            }
            in_d /= stride;
            if (in_d < 0 || in_d >= in_depth) {
                continue;
            }

            for (int kh = 0; kh < kernel_h; kh++) {
                int in_h = oh + padding - kh * dilation;
                if (in_h % stride != 0) {
                    continue;
                }
                in_h /= stride;
                if (in_h < 0 || in_h >= in_height) {
                    continue;
                }

                for (int kw = 0; kw < kernel_w; kw++) {
                    int in_w = ow + padding - kw * dilation;
                    if (in_w % stride != 0) {
                        continue;
                    }
                    in_w /= stride;
                    if (in_w < 0 || in_w >= in_width) {
                        continue;
                    }

                    // input offset
                    int input_offset = 
                        n * (in_channels * in_depth * in_height * in_width) +
                        ic * (in_depth * in_height * in_width) +
                        in_d * (in_height * in_width) +
                        in_h * in_width +
                        in_w;

                    // weight offset
                    int weight_offset = 
                        oc * (in_channels * kernel_d * kernel_h * kernel_w) +
                        ic * (kernel_d * kernel_h * kernel_w) +
                        kd * (kernel_h * kernel_w) +
                        kh * kernel_w +
                        kw;

                    value += input[input_offset] * weight[weight_offset];
                }
            }
        }
    }

    // Write the output
    int out_offset = 
        n * (out_channels * out_depth * out_height * out_width) +
        oc * (out_depth * out_height * out_width) +
        od * (out_height * out_width) +
        oh * out_width +
        ow;

    output[out_offset] = value;
}

// Host function to shape output and launch the kernel
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation
) {
    // Input shape: (N, Cin, D, H, W)
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    // Weight shape: (Cout, Cin, kD, kH, kW)
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Compute output dimensions (following PyTorch's ConvTranspose3d formula)
    int out_depth  = (in_depth  - 1) * stride - 2 * padding + dilation * (kernel_d - 1) + 1;
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + 1;
    int out_width  = (in_width  - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + 1;

    auto options = input.options();
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, options);

    bool has_bias = bias.defined() && bias.numel() > 0;

    // Launch kernel
    int nthreads = batch_size * out_channels * out_depth * out_height * out_width;
    int block_size = 256;
    int grid_size = (nthreads + block_size - 1) / block_size;

    conv_transpose3d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        kernel_d,
        kernel_h,
        kernel_w,
        out_depth,
        out_height,
        out_width,
        stride,
        padding,
        dilation,
        has_bias
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));
    }

    return output;
}
'''

conv_transpose3d_cpp_source = r'''
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation
);
'''

# Compile the inline CUDA code
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=[conv_transpose3d_cpp_source],
    cuda_sources=[conv_transpose3d_source],
    extra_cuda_cflags=["-use_fast_math"],
    functions=["conv_transpose3d_cuda"],
    verbose=True
)


class ModelNew(nn.Module):
    """
    Custom 3D transposed convolution using an inline CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register parameters similarly to nn.ConvTranspose3d (OC, IC, kD, kH, kW)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias

        # Weight shape: (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose3d.conv_transpose3d_cuda(
            x, 
            self.weight, 
            self.bias if self.bias is not None else torch.tensor([]).to(x.device), 
            self.stride, 
            self.padding, 
            self.dilation
        )
