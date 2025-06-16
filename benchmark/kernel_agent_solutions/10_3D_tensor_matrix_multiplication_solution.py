import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------
# Inline CUDA code for 3D (N,M,K) x (K,L) -> (N,M,L) matmul
# using a tiled (shared-memory) matrix multiplication kernel
# ------------------------------------------------------
matmul_3d_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul3d_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int N, int M, int K, int L)
{
    // Each block is responsible for a TILE_SIZE x TILE_SIZE sub-tile of the MxL result,
    // for one of the N "slices".
    // blockIdx.z = n index
    // blockIdx.y = sub-tile along M dimension
    // blockIdx.x = sub-tile along L dimension
    
    int n = blockIdx.z;  // which 3D slice we are on
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // M dimension
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // L dimension

    // Shared memory tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float val = 0.0f;

    // Loop over tiles along K
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        // Load tile of A into shared memory
        int kA = t * TILE_SIZE + threadIdx.x; 
        if (row < M && kA < K)
            As[threadIdx.y][threadIdx.x] = A[n * (M * K) + row * K + kA];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        int kB = t * TILE_SIZE + threadIdx.y;
        if (kB < K && col < L)
            Bs[threadIdx.y][threadIdx.x] = B[kB * L + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial products
        for (int i = 0; i < TILE_SIZE; i++)
            val += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    // Write result
    if (row < M && col < L)
    {
        C[n * (M * L) + row * L + col] = val;
    }
}

// Host function to launch the kernel
torch::Tensor matmul_3d_cuda(torch::Tensor A, torch::Tensor B)
{
    // Ensure inputs are float tensors on CUDA
    A = A.contiguous();
    B = B.contiguous();

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    // Allocate output tensor
    auto out_options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::empty({N, M, L}, out_options);

    // Grid/Block config
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((L + TILE_SIZE - 1)/TILE_SIZE, (M + TILE_SIZE - 1)/TILE_SIZE, N);

    // Launch kernel
    matmul3d_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M, K, L
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(\"Error in matmul3d_kernel: %s\\n\", cudaGetErrorString(err));
    }

    return C;
}
"""

matmul_3d_cpp_source = r"torch::Tensor matmul_3d_cuda(torch::Tensor A, torch::Tensor B);"

# Compile and load the inline CUDA extension
matmul_3d = load_inline(
    name="matmul_3d",
    cpp_sources=matmul_3d_cpp_source,
    cuda_sources=matmul_3d_source,
    functions=["matmul_3d_cuda"],
    verbose=False,
)

# ------------------------------------------------------
# New optimized model that uses our custom 3D matmul kernel
# ------------------------------------------------------
class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication with a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_3d = matmul_3d

    def forward(self, A, B):
        return self.matmul_3d.matmul_3d_cuda(A, B)
