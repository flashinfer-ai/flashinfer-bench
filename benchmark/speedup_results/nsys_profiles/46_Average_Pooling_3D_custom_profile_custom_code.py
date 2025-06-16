import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avgpool3d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void average_pool3d_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      const int batch_size,
                                      const int channels,
                                      const int in_depth,
                                      const int in_height,
                                      const int in_width,
                                      const int kernel_size,
                                      const int stride,
                                      const int padding,
                                      const int out_depth,
                                      const int out_height,
                                      const int out_width) {
    // Each thread handles one output element
    // Compute the global linear index
    int out_index = blockIdx.x * blockDim.x + threadIdx.x
                  + blockIdx.y * blockDim.y * gridDim.x * blockDim.x
                  + blockIdx.z * blockDim.z * gridDim.x * blockDim.x * gridDim.y * blockDim.y;
    int total_output_elements = batch_size * channels * out_depth * out_height * out_width;

    if (out_index >= total_output_elements) {
        return;
    }

    // Decompose out_index into (n, c, od, oh, ow)
    int tmp = out_index;
    int ow = tmp % out_width; 
    tmp /= out_width;
    int oh = tmp % out_height; 
    tmp /= out_height;
    int od = tmp % out_depth;
    tmp /= out_depth;
    int c = tmp % channels;
    int n = tmp / channels;

    // Calculate the input region start and end indices
    int d_start = od * stride - padding;
    int h_start = oh * stride - padding;
    int w_start = ow * stride - padding;
    int d_end = d_start + kernel_size;
    int h_end = h_start + kernel_size;
    int w_end = w_start + kernel_size;

    float sum_val = 0.0f;

    // Accumulate values within the kernel
    // Include out-of-bounds as zero (like PyTorch AvgPool3d with padding)
    for(int kd = d_start; kd < d_end; kd++) {
        for(int kh = h_start; kh < h_end; kh++) {
            for(int kw = w_start; kw < w_end; kw++) {
                if ((kd >= 0) && (kd < in_depth) &&
                    (kh >= 0) && (kh < in_height) &&
                    (kw >= 0) && (kw < in_width)) {
                    int in_index = (((n * channels + c) * in_depth + kd)
                                    * in_height + kh) * in_width + kw;
                    sum_val += input[in_index];
                }
            }
        }
    }

    // Average over the entire kernel volume
    float avg_val = sum_val / static_cast<float>(kernel_size * kernel_size * kernel_size);

    // Write to output
    int out_idx = ((((n * channels) + c) * out_depth + od)
                   * out_height + oh) * out_width + ow;
    output[out_idx] = avg_val;
}

torch::Tensor average_pool3d_cuda(torch::Tensor input,
                                  int kernel_size,
                                  int stride,
                                  int padding) {
    // Input shape: (N, C, D, H, W)
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    // Compute output size (same logic as PyTorch)
    int out_depth = (in_depth + 2 * padding - kernel_size) / stride + 1;
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({batch_size, channels, out_depth, out_height, out_width}, options);

    int total_out = batch_size * channels * out_depth * out_height * out_width;

    // Launch configuration
    dim3 block(64, 1, 1);
    dim3 grid((total_out + block.x - 1) / block.x, 1, 1);

    // If the total size is large, distribute across y or z dimensions as well
    // for better coverage
    if(grid.x > 65535) {
        // Approx splitting in y dimension
        int grid_x = 65535;
        int grid_y = (grid.x + 65534) / 65535;
        grid = dim3(grid_x, grid_y, 1);
    }

    average_pool3d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        in_depth, in_height, in_width,
        kernel_size, stride, padding,
        out_depth, out_height, out_width
    );

    return output;
}
"""

avgpool3d_cpp_source = r"""
torch::Tensor average_pool3d_cuda(torch::Tensor input,
                                  int kernel_size,
                                  int stride,
                                  int padding);
"""

# Compile and load the inline extension
avgpool3d = load_inline(
    name="avgpool3d",
    cpp_sources=avgpool3d_cpp_source,
    cuda_sources=avgpool3d_source,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    functions=["average_pool3d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avgpool3d.average_pool3d_cuda(x,
                                             self.kernel_size,
                                             self.stride,
                                             self.padding)
