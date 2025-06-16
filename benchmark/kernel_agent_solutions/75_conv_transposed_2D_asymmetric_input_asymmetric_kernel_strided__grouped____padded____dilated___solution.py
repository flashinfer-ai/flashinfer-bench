import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

transposed_conv2d_cpp = r"""
#include <torch/extension.h>

// Forward declaration of CUDA function
torch::Tensor transposed_conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transposed_conv2d_cuda_forward", &transposed_conv2d_cuda_forward, "transposed_conv2d_cuda_forward");
}
"""

transposed_conv2d_cuda = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// A naive kernel implementing transposed convolution in a straightforward manner.
// Each thread processes exactly one element of the output tensor.
__global__ void transposed_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int64_t batch_size,
    const int64_t in_channels,
    const int64_t in_h,
    const int64_t in_w,
    const int64_t out_channels,
    const int64_t out_h,
    const int64_t out_w,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    const int64_t groups
) {
    int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_elems = batch_size * out_channels * out_h * out_w;
    if (out_index >= total_elems) return;

    // Decode n, oc, oh, ow
    int64_t ow = out_index % out_w;
    int64_t temp = out_index / out_w;
    int64_t oh = temp % out_h;
    temp /= out_h;
    int64_t oc = temp % out_channels;
    int64_t n = temp / out_channels;

    // Compute group info
    int64_t out_channels_per_group = out_channels / groups;
    int64_t group_id = oc / out_channels_per_group;
    int64_t in_channels_per_group = in_channels / groups;

    float val = 0.0f;
    // Weight is [in_channels, out_channels/group, kH, kW]
    // Input is [N, in_channels, in_h, in_w]
    // Output is [N, out_channels, out_h, out_w]
    // Transposed conv: we sum over the in_channels in the same group.
    for (int64_t c = group_id * in_channels_per_group; c < (group_id + 1) * in_channels_per_group; c++) {
        int64_t c_w = c - (group_id * in_channels_per_group); 
        for (int64_t kh = 0; kh < kernel_h; kh++) {
            for (int64_t kw = 0; kw < kernel_w; kw++) {
                // Compute the corresponding input spatial position
                int64_t in_h_pos = oh * stride_h - pad_h + kh * dilation_h;
                int64_t in_w_pos = ow * stride_w - pad_w + kw * dilation_w;

                // Bounds check
                if (in_h_pos >= 0 && in_h_pos < in_h && in_w_pos >= 0 && in_w_pos < in_w) {
                    int64_t in_idx = ((n * in_channels + c) * in_h + in_h_pos) * in_w + in_w_pos;
                    int64_t w_idx = ((c * out_channels_per_group + (oc % out_channels_per_group)) * kernel_h + kh) * kernel_w + kw;
                    val += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    // Add bias if provided (non-empty)
    if (bias.numel() > 0) {
        val += bias[oc];
    }

    output[out_index] = val;
}

torch::Tensor transposed_conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups
) {
    // Ensure CUDA tensors
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda() || !bias.defined(), "bias must be a CUDA tensor or undefined");

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_h = input.size(2);
    const auto in_w = input.size(3);

    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);
    const auto out_channels = weight.size(1) * groups;

    // Calculate output spatial size (transposed conv formula)
    const int64_t out_h = (in_h - 1) * stride_h - 2 * pad_h + (kernel_h - 1) * dilation_h + 1;
    const int64_t out_w = (in_w - 1) * stride_w - 2 * pad_w + (kernel_w - 1) * dilation_w + 1;

    auto options = input.options();
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, options);

    // Launch kernel
    const int threads = 256;
    const int64_t total_elems = batch_size * out_channels * out_h * out_w;
    const int blocks = (total_elems + threads - 1) / threads;

    transposed_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() && bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups
    );

    return output;
}
"""

# Compile the custom transposed convolution
transposed_conv2d_module = load_inline(
    name="transposed_conv2d_v1",
    cpp_sources=transposed_conv2d_cpp,
    cuda_sources=transposed_conv2d_cuda,
    functions=["transposed_conv2d_cuda_forward"],
    extra_cuda_cflags=["-std=c++17"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Custom Transposed Convolution replacement that uses a naive CUDA kernel.
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
        # Register parameters to mimic PyTorch's ConvTranspose2d shape:
        # weight shape: [in_channels, out_channels/groups, kernel_h, kernel_w]
        kernel_h, kernel_w = kernel_size
        w_shape = (in_channels, out_channels // groups, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.randn(w_shape))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

        self.stride_h, self.stride_w = stride
        self.pad_h, self.pad_w = padding
        self.dilation_h, self.dilation_w = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            bias = torch.tensor([], device=x.device)
        else:
            bias = self.bias
        return transposed_conv2d_module.transposed_conv2d_cuda_forward(
            x,
            self.weight.to(x.device),
            bias.to(x.device),
            int(self.stride_h),
            int(self.stride_w),
            int(self.pad_h),
            int(self.pad_w),
            int(self.dilation_h),
            int(self.dilation_w),
            int(self.groups)
        )
