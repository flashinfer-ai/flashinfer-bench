import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrix_scalar_mul_cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Optional NVTX support (uncomment to enable)
// #include <nvToolsExt.h>

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__global__ void matrix_scalar_mul_kernel(const float* __restrict__ in, float s, float* __restrict__ out, const long size) {
    // Optional NVTX range push (uncomment to enable)
    // nvtxRangePush("matrix_scalar_mul_kernel");
    long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < size){
        out[idx] = in[idx] * s;
    }
    // Optional NVTX range pop (uncomment to enable)
    // nvtxRangePop();
}

torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat, "Input tensor A must be float32");

    auto output = torch::empty_like(A);
    long total_elems = A.numel();

    const int threads = 256;
    int blocks = (total_elems + threads - 1) / threads;

    matrix_scalar_mul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), 
        s, 
        output.data_ptr<float>(), 
        total_elems
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}
"""

matrix_scalar_mul_cpp_decl = """
torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, float s);
"""

matrix_scalar_mul = load_inline(
    name="matrix_scalar_mul",
    cpp_sources=matrix_scalar_mul_cpp_decl,
    cuda_sources=matrix_scalar_mul_cuda_src,
    functions=["matrix_scalar_mul_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_scalar_mul = matrix_scalar_mul

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.matrix_scalar_mul.matrix_scalar_mul_cuda(A, s)
