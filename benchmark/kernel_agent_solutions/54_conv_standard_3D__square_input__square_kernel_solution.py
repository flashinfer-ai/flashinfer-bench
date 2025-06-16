import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ------------------------------
# Inline CUDA/C++ source for a naive 3D convolution (forward only)
# ------------------------------
conv3d_naive_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// A naive GPU kernel for 3D convolution (forward pass). 
// Each thread computes a single element of the output tensor.
__global__ void conv3d_naive_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t KD, int64_t KH, int64_t KW,
    int64_t stride, int64_t padding, int64_t dilation)
{
    // Calculate output dimensions
    int64_t outD = (D + 2 * padding - (dilation * (KD - 1) + 1)) / stride + 1;
    int64_t outH = (H + 2 * padding - (dilation * (KH - 1) + 1)) / stride + 1;
    int64_t outW = (W + 2 * padding - (dilation * (KW - 1) + 1)) / stride + 1;

    // Total number of output elements
    int64_t out_size = N * Cout * outD * outH * outW;
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= out_size) return;

    // Decode the flat output index into n, co, d_out, h_out, w_out
    int64_t w_out = index % outW;
    int64_t tmp = index / outW;
    int64_t h_out = tmp % outH; 
    tmp /= outH;
    int64_t d_out = tmp % outD;
    tmp /= outD;
    int64_t co = tmp % Cout;
    int64_t n  = tmp / Cout;

    // Compute the starting point (focal region) in the input
    int64_t d_in_start = d_out * stride - padding; 
    int64_t h_in_start = h_out * stride - padding; 
    int64_t w_in_start = w_out * stride - padding; 

    // Accumulator for the output value
    float out_val = 0.0f;

    // Perform the convolution sum
    for (int64_t ci = 0; ci < Cin; ci++) {
        for (int64_t kd = 0; kd < KD; kd++) {
            int64_t d_in = d_in_start + kd * dilation;
            if (d_in < 0 || d_in >= D) continue;
            for (int64_t kh = 0; kh < KH; kh++) {
                int64_t h_in = h_in_start + kh * dilation;
                if (h_in < 0 || h_in >= H) continue;
                for (int64_t kw = 0; kw < KW; kw++) {
                    int64_t w_in = w_in_start + kw * dilation;
                    if (w_in < 0 || w_in >= W) continue;

                    int64_t input_idx = n * (Cin * D * H * W) 
                                        + ci * (D * H * W) 
                                        + d_in * (H * W) 
                                        + h_in * W 
                                        + w_in;

                    int64_t weight_idx = co * (Cin * KD * KH * KW)
                                         + ci * (KD * KH * KW)
                                         + kd * (KH * KW)
                                         + kh * KW
                                         + kw;

                    out_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias (if provided)
    if (bias != nullptr) {
        out_val += bias[co];
    }

    // Store result
    output[index] = out_val;
}

// Host function to allocate output, set up kernel dims, and launch the kernel
torch::Tensor conv3d_naive_forward(
    torch::Tensor input,     // [N, Cin, D, H, W]
    torch::Tensor weight,    // [Cout, Cin, KD, KH, KW]
    torch::Tensor bias,      // [Cout] or empty
    int64_t stride,
    int64_t padding,
    int64_t dilation
) {
    // Check for GPU tensors
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda() || !bias.defined(), "bias must be a CUDA tensor or undefined");

    auto N = input.size(0);
    auto Cin = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    auto Cout = weight.size(0);
    auto KD = weight.size(2);
    auto KH = weight.size(3);
    auto KW = weight.size(4);

    int64_t outD = (D + 2 * padding - (dilation * (KD - 1) + 1)) / stride + 1;
    int64_t outH = (H + 2 * padding - (dilation * (KH - 1) + 1)) / stride + 1;
    int64_t outW = (W + 2 * padding - (dilation * (KW - 1) + 1)) / stride + 1;

    // Allocate output
    auto output = torch::zeros({N, Cout, outD, outH, outW}, input.options());

    // Flatten for kernel indexing
    int64_t out_size = N * Cout * outD * outH * outW;

    const int threads = 256;
    const int blocks = (out_size + threads - 1) / threads;

    conv3d_naive_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, Cin, D, H, W,
        Cout, KD, KH, KW,
        stride, padding, dilation
    );

    // Synchronize to catch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed with error: ", cudaGetErrorString(err));

    return output;
}

// Register with the PyTorch dispatcher
TORCH_LIBRARY(my_conv3d, m) {
    m.def("conv3d_naive_forward", conv3d_naive_forward);
}

// Create bindings for Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_naive_forward", &conv3d_naive_forward, "conv3d_naive_forward");
}
"""

# Minimal C++ forward declaration to satisfy load_inline
conv3d_naive_cpp_source = r"""
torch::Tensor conv3d_naive_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation
);
"""

# Build the inline extension
conv3d_naive = load_inline(
    name="conv3d_naive",
    cpp_sources=conv3d_naive_cpp_source,
    cuda_sources=conv3d_naive_source,
    functions=["conv3d_naive_forward"],
    verbose=True
)

# ------------------------------
# PyTorch module that uses the custom CUDA 3D convolution
# ------------------------------
class ModelNew(nn.Module):
    """
    A naive custom 3D convolution layer using our CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We'll store parameters as trainable Variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        # Create weight [out_channels, in_channels, kD, kH, kW]
        # Since original code uses a cube kernel, we'll replicate kernel_size thrice.
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size, kernel_size
        ))

        if self.use_bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias_param', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We call our naive 3D convolution extension
        if self.use_bias:
            return conv3d_naive.conv3d_naive_forward(
                x, self.weight, self.bias_param,
                self.stride, self.padding, self.dilation
            )
        else:
            # Pass a zero bias if bias not used
            return conv3d_naive.conv3d_naive_forward(
                x, self.weight, torch.zeros(
                    self.out_channels, device=x.device, dtype=x.dtype
                ),
                self.stride, self.padding, self.dilation
            )
