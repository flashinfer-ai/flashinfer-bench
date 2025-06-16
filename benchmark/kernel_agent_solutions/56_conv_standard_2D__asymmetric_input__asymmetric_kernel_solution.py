import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>
#include <stdio.h>

// Naive convolution kernel with optional grouping.
// Each thread computes one element of the output.
__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int out_height,
    const int out_width)
{
    // Calculate the output index (n, co, oh, ow) from blockIdx, threadIdx
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * out_channels * out_height * out_width) {
        return;
    }

    // Decompose out_idx to 4D indices
    int ow = out_idx % out_width;
    int temp = out_idx / out_width;
    int oh = temp % out_height;
    temp = temp / out_height;
    int co = temp % out_channels;
    int n = temp / out_channels;

    // Identify which group we're in
    // groups: we split in_channels and out_channels into sub-blocks
    int out_channels_per_group = out_channels / groups;
    int group_id = co / out_channels_per_group;
    int co_in_group = co % out_channels_per_group;

    // For out-of-bounds check
    float val = 0.0f;

    // The offset into weight:
    // weight shape: [out_channels, in_channels/groups, kernel_h, kernel_w]
    // input shape : [batch_size, in_channels, in_height, in_width]
    // output shape: [batch_size, out_channels, out_height, out_width]
    int in_channels_per_group = in_channels / groups;

    // If we have a bias tensor, apply it
    if (bias != nullptr) {
        val = bias[co];
    }

    // Compute convolution
    for(int ci = 0; ci < in_channels_per_group; ci++){
        int in_channel_idx = group_id * in_channels_per_group + ci;

        for(int kh = 0; kh < kernel_h; kh++){
            for(int kw = 0; kw < kernel_w; kw++){
                // Compute the input spatial location
                int ih = oh * stride_h - pad_h + kh * dilation_h;
                int iw = ow * stride_w - pad_w + kw * dilation_w;

                if(ih >= 0 && ih < in_height && iw >= 0 && iw < in_width){
                    // Compute input index
                    int input_index = n * in_channels * in_height * in_width
                                      + in_channel_idx * in_height * in_width
                                      + ih * in_width
                                      + iw;
                    // Compute weight index
                    int weight_index = co * (in_channels_per_group * kernel_h * kernel_w)
                                       + ci * (kernel_h * kernel_w)
                                       + kh * kernel_w
                                       + kw;
                    val += input[input_index] * weight[weight_index];
                }
            }
        }
    }

    // Store result
    output[out_idx] = val;
}

// Forward function that sets up and launches the above kernel.
torch::Tensor conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups)
{
    // Expect float tensors on CUDA
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    }

    // Shapes
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_height = input.size(2);
    const auto in_width = input.size(3);

    const auto out_channels = weight.size(0);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);

    // Calculate output spatial dimensions
    const int out_height = (in_height + 2 * pad_h
                            - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_width = (in_width + 2 * pad_w
                           - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Create output tensor
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, options);

    // Flatten for kernel launch
    const int total_out_elems = batch_size * out_channels * out_height * out_width;

    // Launch kernel
    const int threads = 256;
    const int blocks = (total_out_elems + threads - 1) / threads;

    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        out_height,
        out_width
    );
    // Synchronize to catch any errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv2d_kernel launch failed with error code ", err);
    return output;
}
"""

conv2d_cpp_source = r"""
torch::Tensor conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups);
"""

conv2d = load_inline(
    name="conv2d_naive",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_cuda_forward"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Custom 2D convolution layer using a naive CUDA kernel with support for
    asymmetrical kernel sizes, padding, stride, dilation, and grouping.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False
    ):
        super(ModelNew, self).__init__()
        # Register weight and bias as learnable parameters
        # weight shape: (out_channels, in_channels/groups, kernel_h, kernel_w)
        kernel_h, kernel_w = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_h, kernel_w)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation

        if self.bias is not None:
            return conv2d.conv2d_cuda_forward(
                x, self.weight, self.bias,
                stride_h, stride_w, pad_h, pad_w,
                dilation_h, dilation_w, self.groups
            )
        else:
            # Pass an empty tensor for bias
            empty_bias = torch.tensor([], device=x.device, dtype=x.dtype)
            return conv2d.conv2d_cuda_forward(
                x, self.weight, empty_bias,
                stride_h, stride_w, pad_h, pad_w,
                dilation_h, dilation_w, self.groups
            )
