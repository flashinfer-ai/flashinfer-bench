import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cumprod_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>

/*
  This CUDA implementation handles the cumulative product along dim=1
  for a 2D tensor of shape (nrows, ncols). It uses a three-kernel approach:
  1) partial_prefix_product_kernel: Compute in-block prefix products for chunks.
     Each block processes one row and a chunk of columns; the last partial product
     of each chunk is saved in block_prefixes.
  2) prefix_block_prefixes_kernel: Scan over block_prefixes row by row to obtain
     the prefix products of the chunks.
  3) finalize_prefix_product_kernel: Multiply partial results by the prefix
     from previous chunks to produce the final cumulative product.

  This approach ensures coalesced reads/writes where possible and uses shared memory
  to reduce global memory traffic. Please note that dim=1 is assumed.
*/

__global__ void partial_prefix_product_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ block_prefixes,
    const int nrows,
    const int ncols)
{
    // Each block processes one row, a chunk of columns
    int row = blockIdx.y;
    int chunk_start = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int global_col = chunk_start + tid;

    if (row >= nrows) return;

    extern __shared__ float s[];
    // Load data from global memory into shared memory
    if (global_col < ncols) {
        s[tid] = input[row * ncols + global_col];
    } else {
        // For threads beyond the range, set them to 1 so they don't affect the product
        s[tid] = 1.0f;
    }
    __syncthreads();

    // In-block parallel prefix product (up-sweep phase)
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x) {
            s[idx] *= s[idx - offset];
        }
        __syncthreads();
    }

    // In-block parallel prefix product (down-sweep phase)
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx + offset < blockDim.x) {
            s[idx + offset] *= s[idx];
        }
        __syncthreads();
    }

    // Write results back to global memory
    if (global_col < ncols) {
        output[row * ncols + global_col] = s[tid];
    }

    // The last thread in the block stores the final product of this chunk
    if (tid == blockDim.x - 1) {
        block_prefixes[row * gridDim.x + blockIdx.x] = s[tid];
    }
}

// Kernel to prefix-scan the block_prefixes array for each row
__global__ void prefix_block_prefixes_kernel(
    float* __restrict__ block_prefixes,
    const int nrows,
    const int num_chunks)
{
    // Each row is handled by one block
    int row = blockIdx.x;
    if (row >= nrows) return;

    // We do a simple in-place prefix product on block_prefixes for that row
    // The array for row i is at row * num_chunks, and length num_chunks.
    float running = 1.0f;
    for (int c = 0; c < num_chunks; c++) {
        running *= block_prefixes[row * num_chunks + c];
        block_prefixes[row * num_chunks + c] = running;
    }
}

// Final kernel: multiply each chunk by the prefix from any previous chunks
__global__ void finalize_prefix_product_kernel(
    float* __restrict__ output,
    const float* __restrict__ block_prefixes,
    const int nrows,
    const int ncols)
{
    int row = blockIdx.y;
    int chunk_start = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int global_col = chunk_start + tid;

    if (row >= nrows) return;

    // We retrieve the partial product from all previous chunks
    float prefix_val = 1.0f;
    if (blockIdx.x > 0) {
        int idx_prefix = row * gridDim.x + (blockIdx.x - 1);
        prefix_val = block_prefixes[idx_prefix];
    }

    if (global_col < ncols) {
        output[row * ncols + global_col] *= prefix_val;
    }
}

torch::Tensor cumprod_cuda(torch::Tensor input, int dim) {
    // For simplicity, only handle dim=1 for 2D input
    TORCH_CHECK(input.dim() == 2, "Only 2D tensors are supported by this custom kernel.");
    TORCH_CHECK(dim == 1, "This custom kernel is implemented for dim=1 only.");

    int nrows = input.size(0);
    int ncols = input.size(1);

    auto out = torch::empty_like(input);
    // Decide block and grid sizes
    const int block_size = 256;
    dim3 block(block_size);
    // For each row, we partition columns into chunks of size block_size
    int grid_x = (ncols + block_size - 1) / block_size;
    dim3 grid(grid_x, nrows);

    // Allocate block_prefixes: shape [nrows, grid_x]
    auto block_prefixes = torch::empty({nrows, grid_x}, input.options());

    // 1) partial prefix product
    size_t shared_mem_size = block_size * sizeof(float);
    partial_prefix_product_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        block_prefixes.data_ptr<float>(),
        nrows,
        ncols
    );

    // 2) prefix-scan block_prefixes for each row
    // Use a 1D grid of size nrows, each block handles 1 row
    prefix_block_prefixes_kernel<<<nrows, 1>>>(
        block_prefixes.data_ptr<float>(),
        nrows,
        grid_x
    );

    // 3) finalize: multiply partial results by block_prefix from previous chunks
    finalize_prefix_product_kernel<<<grid, block>>>(
        out.data_ptr<float>(),
        block_prefixes.data_ptr<float>(),
        nrows,
        ncols
    );

    return out;
}
'''

cumprod_cpp_source = r'''
torch::Tensor cumprod_cuda(torch::Tensor input, int dim);
'''

cumprod_extension = load_inline(
    name="cumprod_extension",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_source,
    functions=["cumprod_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a cumulative product operation along dim=1
    using custom CUDA kernels.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return cumprod_extension.cumprod_cuda(x, self.dim)

# Define input dimensions and parameters
batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]
