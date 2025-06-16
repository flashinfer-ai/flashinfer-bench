import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ code for a 3D MaxPool kernel
# This implementation addresses some optimization considerations by:
#  1) Using each thread to compute exactly one output element, ensuring coalesced writes.
#  2) Initializing max values with the first valid input element (where possible).
#  3) Allowing for stride, padding, and dilation parameters.

maxpool_3d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void maxpool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int outD,
    const int outH,
    const int outW)
{
    // Linear index for the output
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C * outD * outH * outW;
    if (index >= total_out) return;

    // Decompose the linear index into (n, c, od, oh, ow)
    int temp = index;
    int n = temp / (C * outD * outH * outW);
    temp = temp % (C * outD * outH * outW);
    int c = temp / (outD * outH * outW);
    temp = temp % (outD * outH * outW);
    int od = temp / (outH * outW);
    temp = temp % (outH * outW);
    int oh = temp / outW;
    int ow = temp % outW;

    // Compute the start indices in the input tensor (accounting for padding)
    int dstart = od * stride - padding;
    int hstart = oh * stride - padding;
    int wstart = ow * stride - padding;

    float max_val = -FLT_MAX;
    bool found_valid = false;

    // Iterate over the pooling window
    for (int kd = 0; kd < kernel_size; kd++) {
        int di = dstart + kd * dilation;
        if (di < 0 || di >= D) continue;

        for (int kh = 0; kh < kernel_size; kh++) {
            int hi = hstart + kh * dilation;
            if (hi < 0 || hi >= H) continue;

            for (int kw = 0; kw < kernel_size; kw++) {
                int wi = wstart + kw * dilation;
                if (wi < 0 || wi >= W) continue;

                // Compute the input index
                int in_idx = (((n * C + c) * D + di) * H + hi) * W + wi;
                float val = input[in_idx];
                if (!found_valid) {
                    max_val = val;
                    found_valid = true;
                } else {
                    max_val = (val > max_val) ? val : max_val;
                }
            }
        }
    }

    // Store the result
    output[index] = found_valid ? max_val : -FLT_MAX;
}

torch::Tensor custom_maxpool3d(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Infer input size
    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto D = x.size(2);
    const auto H = x.size(3);
    const auto W = x.size(4);

    // If stride is None (as per PyTorch convention), set stride = kernel_size
    const int actual_stride = (stride > 0) ? stride : kernel_size;

    // Compute output dimensions (floor mode)
    const int outD = (D + 2 * padding - dilation * (kernel_size - 1) - 1) / actual_stride + 1;
    const int outH = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / actual_stride + 1;
    const int outW = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / actual_stride + 1;

    auto out = torch::empty({N, C, outD, outH, outW}, x.options());

    // Launch CUDA kernel
    int total_out = N * C * outD * outH * outW;
    const int block_size = 256;
    const int grid_size = (total_out + block_size - 1) / block_size;

    maxpool3d_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, D, H, W,
        kernel_size,
        actual_stride,
        padding,
        dilation,
        outD, outH, outW
    );

    return out;
}
""";

maxpool_3d_cpp_source = r"""
torch::Tensor custom_maxpool3d(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);
""";

# Build/Load the CUDA extension
maxpool_3d_extension = load_inline(
    name="my_3d_maxpool_extension",
    cpp_sources=maxpool_3d_cpp_source,
    cuda_sources=maxpool_3d_source,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    functions=["custom_maxpool3d"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Custom 3D Max Pool model with an inline CUDA kernel for optimized performance.
    """
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        # return_indices and ceil_mode are not implemented in this custom kernel
        # We'll ignore them for demonstration purposes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return maxpool_3d_extension.custom_maxpool3d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )
