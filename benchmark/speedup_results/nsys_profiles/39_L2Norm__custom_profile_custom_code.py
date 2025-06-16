import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA code for computing row-wise L2 norm in two steps:
# 1) sums_of_squares_kernel computes sum of squares per row
# 2) l2norm_kernel performs the division by sqrt of that sum
l2norm_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 1024;

__global__ void sums_of_squares_kernel(const float* __restrict__ input,
                                       float* __restrict__ sums,
                                       const int rows,
                                       const int dim) {
    // Each block processes one row
    int row = blockIdx.x;
    if (row >= rows) return;

    // Global offset for the current row
    int offset = row * dim;

    // Use double for higher precision during accumulation
    double thread_sum = 0.0;

    // Loop over elements in the row
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = input[offset + i];
        thread_sum += (double)val * (double)val;
    }

    // Reduce within the block using shared memory
    __shared__ double sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write result for this block (row) to sums
    if (threadIdx.x == 0) {
        sums[row] = (float)sdata[0];
    }
}

__global__ void l2norm_kernel(const float* __restrict__ input,
                              const float* __restrict__ sums,
                              float* __restrict__ output,
                              const int rows,
                              const int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float sum_val = sums[row];
    // Add a small epsilon to avoid division by zero
    float norm_val = sqrtf(sum_val + 1e-12f);
    int offset = row * dim;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[offset + i] = input[offset + i] / norm_val;
    }
}

torch::Tensor sums_of_squares_cuda(torch::Tensor input) {
    const int rows = input.size(0);
    const int dim = input.size(1);
    auto sums = torch::zeros({rows}, torch::dtype(input.dtype()).device(input.device()));

    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);

    sums_of_squares_kernel<<<grid, block>>>(input.data_ptr<float>(),
                                            sums.data_ptr<float>(),
                                            rows, dim);
    return sums;
}

torch::Tensor l2norm_cuda(torch::Tensor input, torch::Tensor sums) {
    const int rows = input.size(0);
    const int dim = input.size(1);
    auto output = torch::empty_like(input);

    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);

    l2norm_kernel<<<grid, block>>>(input.data_ptr<float>(),
                                   sums.data_ptr<float>(),
                                   output.data_ptr<float>(),
                                   rows, dim);
    return output;
}
""";

l2norm_cpp_source = r"""
torch::Tensor sums_of_squares_cuda(torch::Tensor input);
torch::Tensor l2norm_cuda(torch::Tensor input, torch::Tensor sums);
""";

# Build the custom ops
l2norm_ops = load_inline(
    name="l2norm_ops",
    cpp_sources=l2norm_cpp_source,
    cuda_sources=l2norm_source,
    functions=["sums_of_squares_cuda", "l2norm_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using custom CUDA kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_ops = l2norm_ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure the input is on CUDA
        x = x.cuda()
        sums = self.custom_ops.sums_of_squares_cuda(x)
        return self.custom_ops.l2norm_cuda(x, sums)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
