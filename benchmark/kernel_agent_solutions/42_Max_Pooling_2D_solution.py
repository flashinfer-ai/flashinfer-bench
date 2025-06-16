import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_pool2d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdio>

// Naive kernel for Max Pool 2D
__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernel_size, int stride, int padding, int dilation)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C * outH * outW) return;

    // Determine (n, c, oh, ow) from the linear index
    int ow  = index % outW;
    int oh  = (index / outW) % outH;
    int c   = (index / (outW * outH)) % C;
    int n   = index / (outW * outH * C);

    // Compute the top-left corner of the pooling window
    int hstart = oh * stride - padding;
    int wstart = ow * stride - padding;

    float max_val = -3.402823e+38F; // Smallest float

    // Traverse the pooling region
    for(int i=0; i<kernel_size; i++){
        for(int j=0; j<kernel_size; j++){
            int h = hstart + i * dilation;
            int w = wstart + j * dilation;
            if(h >= 0 && h < H && w >= 0 && w < W){
                float val = input[ ((n * C + c) * H + h) * W + w ];
                if(val > max_val){
                    max_val = val;
                }
            }
        }
    }
    output[index] = max_val;
}

// Interface exposing the kernel to Python
torch::Tensor my_maxpool2d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must have 4 dimensions (N, C, H, W)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Compute output spatial dimensions
    int outH = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int outW = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({N, C, outH, outW}, input.options());

    int total = N * C * outH * outW;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    maxpool2d_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        outH, outW,
        kernel_size, stride, padding, dilation
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_maxpool2d_cuda", &my_maxpool2d_cuda, "Naive MaxPool2D CUDA");
}
"""

max_pool2d_op = load_inline(
    name="max_pool2d_op",
    cpp_sources=max_pool2d_source,
    functions=["my_maxpool2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return max_pool2d_op.my_maxpool2d_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)
