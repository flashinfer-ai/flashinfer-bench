import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fused LogSoftmax CUDA kernel (for 2D tensors with log_softmax over dim=1)
log_softmax_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// A warp-level reduction for float max
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// A warp-level reduction for float sum
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for float max
__inline__ __device__ float blockReduceMax(float val) {
    static __shared__ float shared[32]; // 32 partial sums for final warp reduce
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceMax(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // only threads 0..warpSize-1 read and reduce multiple warp results
    float reduced = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -FLT_MAX;
    if (wid == 0) {
        reduced = warpReduceMax(reduced);
    }
    return reduced;
}

// Block-level reduction for float sum
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // 32 partial sums for final warp reduce
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    float reduced = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        reduced = warpReduceSum(reduced);
    }
    return reduced;
}

__global__ void log_softmax_fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int dim
) {
    // Each block handles one batch row:
    //   blockIdx.x = row index
    //   threadIdx.x = thread index
    int row = blockIdx.x;
    // Pointer to the start of the row
    const float* row_input = input + row * dim;
    float* row_output = output + row * dim;

    // 1) Compute max per row (reduce)
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = row_input[i];
        thread_max = fmaxf(thread_max, val);
    }
    // Block reduce for max
    float row_max = blockReduceMax(thread_max);
    __syncthreads();
    if (threadIdx.x == 0) {
        // Broadcast the row_max within the block
        row_output[0] = row_max;  // temporarily store in output to reuse
    }
    __syncthreads();
    row_max = row_output[0];

    // 2) Compute sum of exp(row_input[i] - row_max)
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __expf(row_input[i] - row_max);
        thread_sum += val;
    }
    float row_sum = blockReduceSum(thread_sum);
    __syncthreads();
    if (threadIdx.x == 0) {
        // store row_sum in first element for broadcast
        row_output[0] = row_sum;
    }
    __syncthreads();
    row_sum = row_output[0];

    // 3) Write output = (x - max) - log(sum)
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = row_input[i] - row_max;
        row_output[i] = val - __logf(row_sum);
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor input) {
    // input shape: [batch_size, dim], dim=1 is the second axis
    TORCH_CHECK(input.dim() == 2, "Input must be 2D [batch_size, dim]");
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int dim = sizes[1];

    // Allocate output
    auto output = torch::empty_like(input);

    // We pick a block size. We can adjust based on device properties.
    const int block_size = 256;

    // Each block processes one row (batch index), so we need as many blocks as batch_size
    // This approach assumes that dim can be handled with a block_size threads in a loop,
    // reading multiple elements per thread if dim > block_size.
    dim3 grid(batch_size);
    dim3 block(block_size);

    log_softmax_fused_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );

    // Optional: add GPU error checking in a real system
    // cudaError_t err = cudaGetLastError();
    // TORCH_CHECK(err == cudaSuccess, "Error in log_softmax_fused_kernel: ", cudaGetErrorString(err));

    return output;
}
"""

log_softmax_cpp_source = r"""
torch::Tensor log_softmax_cuda(torch::Tensor input);
"""

# Compile the fused LogSoftmax inline extension
log_softmax_module = load_inline(
    name="log_softmax_fused",
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=["log_softmax_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for log_softmax over dim=1.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.log_softmax_cuda = log_softmax_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assumes input is 2D and log_softmax over dim=1
        # Convert to float if necessary
        x = x.contiguous().float()
        return self.log_softmax_cuda.log_softmax_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
