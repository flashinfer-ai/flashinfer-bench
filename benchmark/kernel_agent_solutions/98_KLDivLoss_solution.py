import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel to compute KL-Divergence with 'batchmean' reduction:
# KL(p || q) = Sum[ p(i) * (log(p(i)) - log(q(i))) ] / batch_size
# where p(i) = targets[i], q(i) = predictions[i].
kl_div_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void kl_div_kernel(const float* preds, const float* targets, float* partial_sums, int totalElements)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float thread_sum = 0.0f;

    // Loop-stride to handle all elements
    while (idx < totalElements) {
        float t = targets[idx];
        float p = preds[idx];
        // Only compute log(t) if t > 0
        if (t > 0.0f && p > 0.0f) {
            thread_sum += t * (logf(t) - logf(p));
        }
        idx += blockDim.x * gridDim.x;
    }

    // Store partial sum in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write per-block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // Ensure float tensors
    auto preds = predictions.contiguous();
    auto targs = targets.contiguous();

    int64_t totalElements = preds.numel();
    const int block_size = 256;
    const int grid_size = (totalElements + block_size - 1) / block_size;

    // One partial sum per block
    auto partial_sums = torch::zeros({grid_size}, preds.options());

    int shmem_size = block_size * sizeof(float);

    // Launch kernel
    kl_div_kernel<<<grid_size, block_size, shmem_size>>>(
        preds.data_ptr<float>(),
        targs.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        totalElements
    );

    // Sum over blocks
    auto total_sum = partial_sums.sum();

    // 'batchmean' reduction: divide by the batch size (first dimension)
    auto batch_size = predictions.size(0);
    auto out = total_sum / (float)batch_size;

    // out is a 0-dimensional tensor (scalar)
    return out;
}
"""

kl_div_cpp_source = r"torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA extension
kl_div_extension = load_inline(
    name="kl_div_extension",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["kl_div_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for KL-Divergence with 'batchmean' reduction.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.kl_div = kl_div_extension

    def forward(self, predictions, targets):
        return self.kl_div.kl_div_cuda(predictions, targets)
