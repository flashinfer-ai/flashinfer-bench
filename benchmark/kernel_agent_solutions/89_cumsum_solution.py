import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

prefix_sum_source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
 * This kernel computes an inclusive prefix sum (cumsum) along dim=1 for a 2D tensor [N, M].
 *
 * Approach:
 *  - Each row is handled by one block.
 *  - Each thread processes a chunk of the row's data in a local array.
 *  - We then compute the partial sum within each thread, store the last element of each
 *    thread's partial sum into a shared array. We prefix-scan that shared array to get
 *    offsets. Finally, we add the offset to each thread's chunk to get the correct prefix.
 *
 * This approach is often called "block-scan with chunked loading."
 * 
 * Constraints:
 *  - N blocks for N rows.
 *  - blockDim.x <= 1024.
 *  - M can exceed blockDim.x, so each thread handles multiple elements.
 *
 * NOTE: The algorithm below is an inclusive scan (cumsum).
 */

__global__ void prefix_sum_kernel_dim1(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int N, int M)
{
    // row index
    int row = blockIdx.x;
    // number of threads
    int tcount = blockDim.x;
    // global thread index within the block
    int tid = threadIdx.x;

    // pointers to start of this row
    const float* row_in  = input  + row * M;
    float*       row_out = output + row * M;

    // each thread processes CHUNK_SIZE elements
    // so total threads * CHUNK_SIZE = M (or slightly more if leftover)
    int CHUNK_SIZE = (M + tcount - 1) / tcount;

    // start and end index for this thread
    int start_idx = tid * CHUNK_SIZE;
    int end_idx   = (start_idx + CHUNK_SIZE > M) ? M : (start_idx + CHUNK_SIZE);

    // local array to store our thread chunk
    // CHUNK_SIZE can be at most M/tcount + 1
    // We'll do it dynamically if the compiler supports it, else a fixed size max
    // For simplicity, use a static size that covers the maximum we might handle.
    // We assume M <= 65536, for instance. If we have 4000, this is safe.
    __shared__ float thread_offsets[1024];  // to hold partial sums for each thread
    __shared__ float s_data[2048];  // scratch if needed, but we'll store chunk in registers

    // 1) Load data from global, compute partial prefix in local memory
    float accum = 0.0f;
    for(int i = start_idx; i < end_idx; i++){
        accum += row_in[i];
        // we can write directly to row_out here temporarily
        // and add the offset later
        row_out[i] = accum;
    }

    // 2) The last value in 'accum' is the partial sum for this thread
    // store it in shared memory
    thread_offsets[tid] = accum;

    __syncthreads();

    // If tid=0, we handle the prefix sum of each thread's offset in shared memory
    // We'll do a standard in-block prefix sum on thread_offsets
    // Then each thread_offsets[i] is the sum of offsets up to i (inclusive).
    // We'll do an inclusive scan of thread_offsets.

    // -- Up-sweep
    for(int step = 1; step < tcount; step <<= 1){
        int idx = (tid + 1) * step * 2 - 1;
        if(idx < tcount){
            thread_offsets[idx] += thread_offsets[idx - step];
        }
        __syncthreads();
    }
    // -- Down-sweep
    for(int step = tcount/2; step > 0; step >>= 1){
        int idx = (tid + 1) * step * 2 - 1;
        if(idx + step < tcount){
            thread_offsets[idx + step] += thread_offsets[idx];
        }
        __syncthreads();
    }

    // 3) Now thread_offsets[tid] holds the offset for thread tid (the total sum up to that thread).
    //    We must add thread_offsets[tid - 1] to each element in this thread's chunk,
    //    except for tid=0 which has no prior offset. So the offset to add is thread_offsets[tid - 1].
    float add_val = (tid == 0) ? 0.0f : thread_offsets[tid - 1];

    // 4) Add that offset to all elements in this thread's chunk (since we did inclusive partial sums earlier).
    for(int i = start_idx; i < end_idx; i++){
        row_out[i] += add_val;
    }
}

torch::Tensor prefix_sum_dim1_cuda(torch::Tensor x)
{
    // x is [N, M], we do cumsum along dim=1
    TORCH_CHECK(x.dim() == 2, "Input must be 2D (N, M).");
    int N = x.size(0);
    int M = x.size(1);

    auto out = torch::empty_like(x);

    // We'll launch N blocks, each with up to 256 or 512 threads
    // so that we can handle M=4000 in chunk fashion
    int threads = 256;
    dim3 block(threads);
    dim3 grid(N);

    prefix_sum_kernel_dim1<<<grid, block>>>(x.data_ptr<float>(),
                                           out.data_ptr<float>(),
                                           N, M);
    return out;
}
''';

prefix_sum_cpp_source = r'''
torch::Tensor prefix_sum_dim1_cuda(torch::Tensor x);
''';

# Compile the inline CUDA code for prefix sum along dim=1
prefix_sum_module = load_inline(
    name="prefix_sum_dim1",
    cpp_sources=prefix_sum_cpp_source,
    cuda_sources=prefix_sum_source,
    functions=["prefix_sum_dim1_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a cumulative sum (prefix sum) operation along dim=1
    using a custom CUDA kernel.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        # This implementation only optimizes cumsum along dim=1. 
        # If dim != 1, we revert to torch.cumsum for correctness.
    
    def forward(self, x):
        # Check dimension; if dim=1, use custom kernel, else fallback.
        if self.dim == 1 and x.is_cuda and x.dtype == torch.float32:
            return prefix_sum_module.prefix_sum_dim1_cuda(x)
        else:
            return torch.cumsum(x, dim=self.dim)
