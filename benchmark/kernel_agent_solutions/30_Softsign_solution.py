import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softsign_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float softsign_func(float x) {
    return x / (1.0f + fabsf(x));
}

__global__ void softsign_kernel(const float* __restrict__ in, float* __restrict__ out, int size) {
    int totalThreads = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < size) {
        float val = __ldg(&in[idx]);
        out[idx] = softsign_func(val);
        idx += totalThreads;
    }
}

torch::Tensor softsign_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    softsign_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), 
                                              output.data_ptr<float>(), 
                                              size);
    return output;
}
'''

softsign_cpp_source = "torch::Tensor softsign_cuda(torch::Tensor input);"

# Compile the inline CUDA code
softsign_module = load_inline(
    name="softsign_module",
    cpp_sources=softsign_cpp_source,
    cuda_sources=softsign_source,
    functions=["softsign_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a Softsign activation with a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softsign_fn = softsign_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softsign_fn.softsign_cuda(x)
