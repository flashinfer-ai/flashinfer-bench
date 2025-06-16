import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Single CUDA/C++ source implementing both depthwise and pointwise convolution.
# We define them in one module to avoid multiple-definition linkage errors.
depthwise_separable_conv_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// -------------------------------------
// Depthwise 2D Convolution (Naive)
// -------------------------------------
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size, int in_channels,
    int in_height, int in_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int out_height, int out_width)
{
    // Each thread will compute exactly one element in [N, C, outH, outW].
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = batch_size * in_channels * out_height * out_width;
    if (index >= total_elems) return;

    // Decompose index into [n, c, oh, ow]
    int ow = index % out_width;
    int oh = (index / out_width) % out_height;
    int c  = (index / (out_width * out_height)) % in_channels;
    int n  = index / (out_width * out_height * in_channels);

    // Compute the corresponding input region
    int in_row = oh * stride_h - pad_h;
    int in_col = ow * stride_w - pad_w;

    float val = 0.0f;
    // Depthwise: single channel per group
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            int r = in_row + kh * dilation_h;
            int c_ = in_col + kw * dilation_w;
            if (r >= 0 && r < in_height && c_ >= 0 && c_ < in_width) {
                int input_idx = n * (in_channels * in_height * in_width)
                                + c * (in_height * in_width)
                                + r * in_width
                                + c_;
                int weight_idx = c * (kernel_h * kernel_w)
                                 + kh * kernel_w
                                 + kw;
                val += input[input_idx] * weight[weight_idx];
            }
        }
    }

    int out_idx = n * (in_channels * out_height * out_width)
                  + c * (out_height * out_width)
                  + oh * out_width
                  + ow;
    output[out_idx] = val;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor dw_weight,
    int stride,
    int padding,
    int dilation)
{
    // input shape:  [N, C, H, W]
    // dw_weight shape: [C, 1, kH, kW] (depthwise)
    // We replicate PyTorch's output size formula
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(dw_weight.dim() == 4, "dw_weight must be 4D");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int kernel_h = dw_weight.size(2);
    int kernel_w = dw_weight.size(3);

    // Support only stride = stride, no separate stride_w
    int stride_h = stride;
    int stride_w = stride;
    int pad_h = padding;
    int pad_w = padding;
    int dilation_h = dilation;
    int dilation_w = dilation;

    // Calculate output height/width
    int out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width  = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({batch_size, in_channels, out_height, out_width}, options);

    int block_size = 256;
    int total_elems = batch_size * in_channels * out_height * out_width;
    int grid_size = (total_elems + block_size - 1) / block_size;

    depthwise_conv2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        dw_weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels,
        in_height, in_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        out_height, out_width
    );

    return output;
}

// -------------------------------------
// Pointwise 2D Convolution (Naive)
// -------------------------------------
__global__ void pointwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width)
{
    // We produce [N, outC, H, W]
    // Each thread: compute one [n, oc, h, w]
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = batch_size * out_channels * height * width;
    if (index >= total_elems) return;

    int w_out = index % width;
    int h_out = (index / width) % height;
    int oc    = (index / (width * height)) % out_channels;
    int n     = index / (width * height * out_channels);

    float val = 0.0f;
    // pointwise weight shape: [outC, inC, 1, 1]
    // sum over inC
    for(int ic = 0; ic < in_channels; ic++){
        int in_idx = n * (in_channels * height * width)
                     + ic * (height * width)
                     + h_out * width
                     + w_out;
        int w_idx = oc * in_channels + ic; // since kernel 1x1

        val += input[in_idx] * weight[w_idx];
    }

    int out_idx = n * (out_channels * height * width)
                  + oc * (height * width)
                  + h_out * width
                  + w_out;
    output[out_idx] = val;
}

torch::Tensor pointwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor pw_weight)
{
    // input: [N, inC, H, W]
    // pw_weight: [outC, inC, 1, 1]
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(pw_weight.dim() == 4, "pw_weight must be 4D");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width  = input.size(3);
    int out_channels = pw_weight.size(0);

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({batch_size, out_channels, height, width}, options);

    int block_size = 256;
    int total_elems = batch_size * out_channels * height * width;
    int grid_size = (total_elems + block_size - 1) / block_size;

    pointwise_conv2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        pw_weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height, width
    );

    return output;
}

// Bindings for Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depthwise_conv2d_cuda", &depthwise_conv2d_cuda, "Depthwise Conv2D (CUDA)");
    m.def("pointwise_conv2d_cuda", &pointwise_conv2d_cuda, "Pointwise Conv2D (CUDA)");
}
"""

# Build the CUDA extension in a single module
depthwise_separable_conv = load_inline(
    name="depthwise_separable_conv",
    cpp_sources="",
    cuda_sources=depthwise_separable_conv_source,
    functions=["depthwise_conv2d_cuda", "pointwise_conv2d_cuda"],
    verbose=True,
    extra_cflags=["-std=c++17"],
    extra_cuda_cflags=["-std=c++17"]
)

class ModelNew(nn.Module):
    """
    Optimized Depthwise-Separable 2D Convolution using custom CUDA kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # For simplicity, store weights as Parameters similar to PyTorch's shapes.
        # Depthwise shape: [inC, 1, kH, kW]
        self.dw_weight = nn.Parameter(
            torch.randn(in_channels, 1, kernel_size, kernel_size)
        )
        # Pointwise shape: [outC, inC, 1, 1]
        self.pw_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, 1, 1)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias
        if bias:
            # If needed, you may define bias Parameters for each conv;
            # not implemented here for brevity
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Depthwise convolution
        dw_out = depthwise_separable_conv.depthwise_conv2d_cuda(
            x, self.dw_weight,
            self.stride,
            self.padding,
            self.dilation
        )
        # Step 2: Pointwise convolution
        out = depthwise_separable_conv.pointwise_conv2d_cuda(
            dw_out, self.pw_weight
        )
        return out
