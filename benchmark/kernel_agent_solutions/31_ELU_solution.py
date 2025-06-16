import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA code for ELU
elu_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elu_kernel(const float* __restrict__ x,
                           float* __restrict__ out,
                           const float alpha,
                           const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = (val >= 0.0f) ? val : alpha * (expf(val) - 1.0f);
    }
}

torch::Tensor elementwise_elu_cuda(torch::Tensor x, float alpha) {
    // Ensure contiguous memory for coalesced access
    auto x_contig = x.contiguous();
    auto size = x_contig.numel();
    auto out = torch::empty_like(x_contig);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    elu_kernel<<<num_blocks, block_size>>>(
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        alpha,
        size
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Kernel launch error: ") + cudaGetErrorString(err));
    }

    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Device synchronize error: ") + cudaGetErrorString(err));
    }

    // Reshape output to original input shape
    return out.view(x.sizes());
}
""";

# Corresponding C++ header
elu_header = r"""
torch::Tensor elementwise_elu_cuda(torch::Tensor x, float alpha);
""";

# Compile the inlined CUDA code
elementwise_elu = load_inline(
    name="elementwise_elu",
    cpp_sources=elu_header,
    cuda_sources=elu_source,
    functions=["elementwise_elu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.elu_op = elementwise_elu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu_op.elementwise_elu_cuda(x, self.alpha)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return [1.0]
