import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

M = 8205
K = 2949
N = 5921

matmul_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_DIM 32

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                    int M, int K, int N) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float val = 0.0f;
    // Loop over tiles
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        int tiledAcol = t * TILE_DIM + threadIdx.x;
        int tiledBrow = t * TILE_DIM + threadIdx.y;

        // Load A tile
        if (row < M && tiledAcol < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tiledAcol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile
        if (col < N && tiledBrow < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(tiledBrow * N) + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        for (int i = 0; i < TILE_DIM; i++) {
            val += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Store result
    if (row < M && col < N) {
        C[row * N + col] = val;
    }
}

torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure input tensors are on the same device
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    matmul_tiled_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        auto msg = std::string("CUDA kernel launch error: ") + cudaGetErrorString(err);
        throw std::runtime_error(msg);
    }

    return C;
}
"""

matmul_cpp_source = r"""
torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code for matrix multiplication
matmul_tiled = load_inline(
    name="matmul_tiled",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_tiled_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized tile-based matrix multiplication with shared memory usage.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_fn = matmul_tiled

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_fn.matmul_tiled_cuda(A, B)

def get_inputs():
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    return [A, B]

def get_init_inputs():
    return []
