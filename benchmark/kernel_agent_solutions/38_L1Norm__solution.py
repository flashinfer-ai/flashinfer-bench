import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_l1_norm_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_l1_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int D
) {
    extern __shared__ float shared_sums[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Phase 1: Partial sum in shared memory
    float sum_val = 0.0f;
    for (int col = tid; col < D; col += blockSize) {
        sum_val += fabsf(input[row * D + col]);
    }
    shared_sums[tid] = sum_val;

    __syncthreads();

    // Phase 2: Reduce in shared memory
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
        }
        __syncthreads();
    }

    // Phase 3: Normalize with error checking
    float row_sum = shared_sums[0];
    if (row_sum != 0.0f) {
        for (int col = tid; col < D; col += blockSize) {
            output[row * D + col] = input[row * D + col] / row_sum;
        }
    } else {
        for (int col = tid; col < D; col += blockSize) {
            output[row * D + col] = 0.0f;
        }
    }
}

torch::Tensor fused_l1_norm_cuda(torch::Tensor input) {
    int N = input.size(0);
    int D = input.size(1);
    auto output = torch::empty_like(input);

    // Configure block size (can be tuned)
    const int blockSize = 256;

    // Launch one block per row, with enough shared memory for partial sums
    dim3 grid(N);
    dim3 block(blockSize);

    size_t shared_mem_bytes = blockSize * sizeof(float);

    fused_l1_norm_kernel<<<grid, block, shared_mem_bytes>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );

    return output;
}
"""

fused_l1_norm_cpp_source = r"""
torch::Tensor fused_l1_norm_cuda(torch::Tensor input);
"""

fused_l1_norm = load_inline(
    name="fused_l1_norm",
    cpp_sources=fused_l1_norm_cpp_source,
    cuda_sources=fused_l1_norm_source,
    functions=["fused_l1_norm_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization via a custom fused CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.fused_l1_norm = fused_l1_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fused_l1_norm.fused_l1_norm_cuda(x)


batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
