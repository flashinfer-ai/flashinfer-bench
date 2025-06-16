import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ source for a custom 2D convolution kernel
conv2d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// A naive 2D convolution forward kernel with support for stride, padding, dilation, and optional bias.
// This kernel demonstrates a correct approach to indexing, though further optimization (tiling, shared memory)
// may be required for improved performance.
__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in,
    int H_in, int W_in,
    int C_out, int kernel_h,
    int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int H_out, int W_out)
{
    // 3D grid:
    //   gridDim.x = ceil(W_out / blockDim.x)
    //   gridDim.y = ceil(H_out / blockDim.y)
    //   gridDim.z = B * C_out
    // 2D block:
    //   blockDim.x, blockDim.y

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int bc  = blockIdx.z;      // combined batch & out_channel index
    if (w_out >= W_out || h_out >= H_out) {
        return;
    }
    int n   = bc / C_out;      // batch index
    int co  = bc % C_out;      // out_channel index

    float value = 0.0f;
    // Accumulate over all input channels and the kernel extent
    for(int ci = 0; ci < C_in; ci++) {
        for(int kh = 0; kh < kernel_h; kh++) {
            for(int kw = 0; kw < kernel_w; kw++) {
                int in_h = h_out * stride_h - pad_h + kh * dilation_h;
                int in_w = w_out * stride_w - pad_w + kw * dilation_w;
                if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
                    float inp_val = input[((n * C_in + ci) * H_in + in_h) * W_in + in_w];
                    float wgt_val = weight[(((co * C_in) + ci) * kernel_h + kh) * kernel_w + kw];
                    value += inp_val * wgt_val;
                }
            }
        }
    }

    // Add bias if available
    if (bias != nullptr) {
        value += bias[co];
    }
    // Store result
    output[ ((n * C_out + co) * H_out + h_out) * W_out + w_out ] = value;
}

// Forward function in C++ that launches the above kernel
torch::Tensor conv2d_cuda_forward(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const c10::optional<torch::Tensor> &bias_opt,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt->is_cuda(), "bias must be a CUDA tensor if provided");
    }

    int B       = input.size(0);
    int C_in    = input.size(1);
    int H_in    = input.size(2);
    int W_in    = input.size(3);
    int C_out   = weight.size(0);
    int kernel_h= weight.size(2);
    int kernel_w= weight.size(3);

    // Compute output spatial size
    int H_out = (H_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto options = weight.options().dtype(input.dtype());
    torch::Tensor output = torch::empty({B, C_out, H_out, W_out}, options);

    dim3 block(16, 16);
    dim3 grid(
        (W_out + block.x - 1) / block.x,
        (H_out + block.y - 1) / block.y,
        B * C_out
    );

    conv2d_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_opt.has_value() ? bias_opt->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, C_in, H_in, W_in,
        C_out, kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        H_out, W_out
    );

    return output;
}
"""

# C++ function declaration for calling from Python
conv2d_cpp_source = r"""
torch::Tensor conv2d_cuda_forward(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const c10::optional<torch::Tensor> &bias_opt,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w
);
"""

# Compile the inline CUDA code
conv2d_ext = load_inline(
    name="conv2d_ext",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda_forward"],
    verbose=False
)


class ModelNew(nn.Module):
    """
    Custom 2D convolution module with a fused kernel.
    Replaces PyTorch's nn.Conv2d forward pass with a custom CUDA kernel.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        bias: bool = False
    ):
        super().__init__()
        # Create weight/bias as learnable parameters
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1])
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the custom CUDA forward
        return conv2d_ext.conv2d_cuda_forward(
            x,
            self.weight,
            self.bias if self.bias is not None else None,
            self.stride,
            self.stride,    # For simplicity, use stride_h = stride_w here
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1]
        )
