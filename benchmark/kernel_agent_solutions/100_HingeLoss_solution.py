import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hinge_loss_header = r"""
torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

hinge_loss_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// First kernel: compute hinge losses per element and accumulate partial sums per block
__global__ void hinge_loss_kernel(const float* preds, const float* tgts,
                                  float* block_sums, int size) {
    extern __shared__ __align__(sizeof(float)) unsigned char smem[];
    float* sdata = reinterpret_cast<float*>(smem);

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;

    // Coalesced read (each thread reads one contiguous element)
    if (global_idx < size) {
        float hinge = fmaxf(0.0f, 1.0f - preds[global_idx] * tgts[global_idx]);
        val = hinge;
    }

    // Store per-thread result in shared memory
    sdata[threadIdx.x] = val;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// Second kernel: reduce partial sums from all blocks to a single sum
__global__ void reduce_sum_kernel(float* block_sums, int num_blocks) {
    extern __shared__ __align__(sizeof(float)) unsigned char smem[];
    float* sdata = reinterpret_cast<float*>(smem);

    int tid = threadIdx.x;

    // Load partial sums into shared memory
    if (tid < num_blocks) {
        sdata[tid] = block_sums[tid];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    // Reduce within the single block
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write the total sum to block_sums[0]
    if (tid == 0) {
        block_sums[0] = sdata[0];
    }
}

// Main hinge loss wrapper (returns a scalar)
torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat, "targets must be float32");

    auto size = predictions.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Allocate temporary tensor for block sums
    auto block_sums = torch::empty({blocks}, predictions.options());

    // Launch first kernel to compute block-wise partial sums of hinge losses
    size_t smem_size = threads * sizeof(float);
    hinge_loss_kernel<<<blocks, threads, smem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        size
    );

    // Launch second kernel for final reduction
    size_t smem_size2 = threads * sizeof(float);
    reduce_sum_kernel<<<1, threads, smem_size2>>>(block_sums.data_ptr<float>(), blocks);

    // Retrieve final sum on CPU
    float sum_hinge;
    cudaMemcpy(&sum_hinge, block_sums.data_ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);

    // Compute mean
    float mean_hinge = sum_hinge / static_cast<float>(size);

    // Return a 0-dim tensor with final hinge loss
    auto out = torch::empty({}, predictions.options());
    out.fill_(mean_hinge);

    return out;
}
"""

# Compile the inline extension
hinge_loss_module = load_inline(
    name="hinge_loss_module",
    cpp_sources=hinge_loss_header,
    cuda_sources=hinge_loss_source,
    functions=["hinge_loss_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model computing hinge loss with a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hinge_loss_cuda = hinge_loss_module

    def forward(self, predictions, targets):
        return self.hinge_loss_cuda.hinge_loss_cuda(predictions, targets)


# Keep the same input functions
batch_size = 128
input_shape = (1,)
dim = 1

def get_inputs():
    return [
        torch.randn(batch_size, *input_shape).cuda(),
        (torch.randint(0, 2, (batch_size, 1)).float() * 2 - 1).cuda()
    ]

def get_init_inputs():
    return []
