import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

sigmoid_cpp_source = r"""
torch::Tensor sigmoid_cuda(torch::Tensor input);
"""

sigmoid_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sigmoid_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // Use __expf for slightly faster exponential operation
        output[idx] = 1.0f / (1.0f + __expf(-val));
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor input) {
    // Ensure input is contiguous and on CUDA
    input = input.contiguous();
    auto output = torch::empty_like(input);

    const int size = input.numel();
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    sigmoid_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
   );

    return output;
}
"""

sigmoid_ops = load_inline(
    name="sigmoid_ops",
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_cuda_source,
    functions=["sigmoid_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for Sigmoid activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid_ops = sigmoid_ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid_ops.sigmoid_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
