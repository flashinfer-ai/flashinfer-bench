import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cross_entropy_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel to compute per-sample cross entropy values.
// predictions: [B, C]
// targets: [B]
// partial_cost: [B]
// B = batch_size, C = num_classes
__global__ void cross_entropy_forward_kernel(
    const float* __restrict__ predictions,
    const int64_t* __restrict__ targets,
    float* __restrict__ partial_cost,
    const int B, const int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B) {
        int offset = idx * C;

        // 1) Find max value for numerical stability
        float max_val = -1.0e30f;
        for(int j = 0; j < C; j++){
            float val = predictions[offset + j];
            if (val > max_val) {
                max_val = val;
            }
        }

        // 2) Compute sum of exp
        float sum_exp = 0.0f;
        for(int j = 0; j < C; j++){
            sum_exp += expf(predictions[offset + j] - max_val);
        }

        // 3) Compute log of the sum of exponentials, shift back by max_val
        float log_sum_exp = logf(sum_exp) + max_val;

        // 4) Negative log-likelihood for the correct class
        int64_t t = targets[idx];
        float correct_val = predictions[offset + t];
        float nll = log_sum_exp - correct_val;

        // Store per-sample cost
        partial_cost[idx] = nll;
    }
}

// Simple reduction kernel to sum over partial_cost.
// partial_cost: [B]
// total_cost: [1]
__global__ void reduce_sum_kernel(
    const float* __restrict__ partial_cost,
    float* __restrict__ total_cost,
    int B
) {
    __shared__ float buffer[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < B) {
        val = partial_cost[idx];
    }
    buffer[tid] = val;

    __syncthreads();

    // In-block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            buffer[tid] += buffer[tid + s];
        }
        __syncthreads();
    }

    // Accumulate block results atomically
    if (tid == 0) {
        atomicAdd(total_cost, buffer[0]);
    }
}

// Cross-entropy forward pass in CUDA
torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // predictions: [B, C]
    // targets: [B]
    // returns a scalar tensor

    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 1, "targets must be 1D");

    const auto B = predictions.size(0);
    const auto C = predictions.size(1);

    auto partial_cost = torch::empty({B}, predictions.options().dtype(torch::kFloat));
    auto total_cost = torch::zeros({1}, predictions.options().dtype(torch::kFloat));

    // Kernel launch for cross entropy computation
    const int block_size = 256;
    const int grid_size = (B + block_size - 1) / block_size;
    cross_entropy_forward_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        partial_cost.data_ptr<float>(),
        B, C
    );

    // Kernel launch for sum reduction
    reduce_sum_kernel<<<grid_size, block_size>>>(
        partial_cost.data_ptr<float>(),
        total_cost.data_ptr<float>(),
        B
    );

    // Divide by B to get mean
    // (We'll do this on the GPU to keep everything on-device)
    auto mean_ce = total_cost / static_cast<float>(B);

    return mean_ce;
}
"""

cross_entropy_cpp_source = r"""
torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# Build the custom cross entropy extension
cross_entropy = load_inline(
    name="cross_entropy",
    cpp_sources=cross_entropy_cpp_source,
    cuda_sources=cross_entropy_source,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    functions=["cross_entropy_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    An optimized model that computes Cross Entropy Loss for multi-class classification
    using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy = cross_entropy

    def forward(self, predictions, targets):
        return self.cross_entropy.cross_entropy_cuda(predictions, targets)

batch_size = 4096
num_classes = 10
input_shape = (num_classes, )

def get_inputs():
    return [
        torch.randn(batch_size, *input_shape, device="cuda"),
        torch.randint(0, num_classes, (batch_size,), device="cuda")
    ]

def get_init_inputs():
    return []
