import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

layernorm_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel to compute partial sums and partial sums of squares per batch
__global__ void partialSumKernel(
    const float* __restrict__ x,
    float* __restrict__ rowSum,
    float* __restrict__ rowSqSum,
    const int B, const int N)
{
    // b: batch index
    // Each block processes one b (blockIdx.x),
    // and a chunk of N (blockIdx.y).
    int b = blockIdx.x;
    int chunkStart = blockIdx.y * blockDim.x;
    int tid = threadIdx.x;
    int index = chunkStart + tid;
    
    // Shared memory for partial reductions
    __shared__ float sdata[256];
    __shared__ float sdataSq[256];

    float val = 0.0f;
    float valSq = 0.0f;

    if (index < N) {
        val = x[b * N + index];
        valSq = val * val;
    }

    // Store in shared memory
    sdata[tid] = val;
    sdataSq[tid] = valSq;
    __syncthreads();

    // Blockwise reduction
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdataSq[tid] += sdataSq[tid + s];
        }
        __syncthreads();
    }

    // Atomic add to global sums
    if (tid == 0) {
        atomicAdd(&rowSum[b], sdata[0]);
        atomicAdd(&rowSqSum[b], sdataSq[0]);
    }
}

// Kernel to compute mean and var per batch
__global__ void computeMeanVarKernel(
    const float* __restrict__ rowSum,
    const float* __restrict__ rowSqSum,
    float* __restrict__ rowMean,
    float* __restrict__ rowVar,
    const int B, const int N, const float eps)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float sum = rowSum[b];
        float sqSum = rowSqSum[b];
        float mean = sum / (float)N;
        float var = (sqSum / (float)N) - (mean * mean); 
        if (var < 0.f) var = 0.f; // numerical safeguard
        rowMean[b] = mean;
        rowVar[b] = var + eps;
    }
}

// Kernel to apply layer norm
__global__ void layernormForwardKernel(
    const float* __restrict__ x,
    const float* __restrict__ rowMean,
    const float* __restrict__ rowVar,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    const int B, const int N)
{
    int b = blockIdx.x;
    int chunkStart = blockIdx.y * blockDim.x;
    int tid = threadIdx.x;
    int index = chunkStart + tid;

    if (index < N) {
        float mean = rowMean[b];
        float var = rowVar[b];
        float invStd = rsqrtf(var);
        float xVal = x[b * N + index];
        float normVal = (xVal - mean) * invStd;
        normVal = normVal * gamma[index] + beta[index];
        y[b * N + index] = normVal;
    }
}

torch::Tensor layernorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps)
{
    // Ensure inputs are on CUDA
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "beta must be a CUDA tensor");

    // Flatten the last dimensions to (B, N)
    int B = x.size(0);
    int64_t N = 1;
    for (int i = 1; i < x.dim(); i++) {
        N *= x.size(i);
    }

    // Allocate output
    auto y = torch::zeros_like(x);

    // Create temp buffers on GPU
    auto rowSum = torch::zeros({B}, x.options());
    auto rowSqSum = torch::zeros({B}, x.options());
    auto rowMean = torch::zeros({B}, x.options());
    auto rowVar = torch::zeros({B}, x.options());

    // Compute partial sums
    dim3 block(256);
    dim3 gridPartial(B, (N + 255) / 256);
    partialSumKernel<<<gridPartial, block>>>(
        x.data_ptr<float>(),
        rowSum.data_ptr<float>(),
        rowSqSum.data_ptr<float>(),
        B, (int)N
    );

    // Compute mean/var
    dim3 gridMV((B + 255) / 256);
    computeMeanVarKernel<<<gridMV, block>>>(
        rowSum.data_ptr<float>(),
        rowSqSum.data_ptr<float>(),
        rowMean.data_ptr<float>(),
        rowVar.data_ptr<float>(),
        B, (int)N,
        eps
    );

    // Apply LN
    dim3 gridLN(B, (N + 255) / 256);
    layernormForwardKernel<<<gridLN, block>>>(
        x.data_ptr<float>(),
        rowMean.data_ptr<float>(),
        rowVar.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        B, (int)N
    );

    return y;
}
""";

layernorm_cpp_source = r"""
torch::Tensor layernorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps);
""";

# Build the extension
layernorm_mod = load_inline(
    name="layernorm_mod",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_forward_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Custom LayerNorm model using a fused CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        # Parameters for LN
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten gamma/beta to match flattened LN dimension
        gamma_flat = self.gamma.view(-1)
        beta_flat = self.beta.view(-1)
        return layernorm_mod.layernorm_forward_cuda(
            x.contiguous(),
            gamma_flat.contiguous(),
            beta_flat.contiguous(),
            self.eps
        )
        
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2, device='cuda')
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]
