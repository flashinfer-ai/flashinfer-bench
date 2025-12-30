import sys
import time
from pathlib import Path

import pytest
import torch

from flashinfer_bench.compile.builders import TorchBuilder
from flashinfer_bench.data import (
    AxisConst,
    BuildSpec,
    Definition,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
)


@pytest.fixture(autouse=True)
def _use_tmp_cache_dir(tmp_cache_dir: Path) -> None:
    """Automatically use tmp_cache_dir for all tests in this module."""


def test_cpu_simple():
    builder = TorchBuilder()

    definition = Definition(
        name="cpu_echo_def",
        op_type="op",
        axes={"M": AxisConst(value=1)},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="def run(A):\n    return A\n",
    )

    cpp_source = r"""
#include <torch/extension.h>
namespace py = pybind11;
torch::Tensor echo(torch::Tensor A) { return A; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("echo", &echo); }
"""

    solution = Solution(
        name="cpu_echo_solution",
        definition="cpu_echo_def",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.CPP,
            target_hardware=["cpu"],
            entry_point="bind.cpp::echo",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="bind.cpp", content=cpp_source)],
    )
    runnable = builder.build(definition, solution)

    input_tensor = torch.tensor([1, 2, 3], dtype=torch.float32, device="cpu")
    out: torch.Tensor = runnable(input_tensor)
    assert torch.allclose(out, input_tensor)


def test_cpu_overhead():
    builder = TorchBuilder()

    definition = Definition(
        name="cpu_echo_def",
        op_type="op",
        axes={"M": AxisConst(value=1)},
        inputs={
            "AAAAAA": TensorSpec(shape=["M"], dtype="float32"),
            "BBBBBB": TensorSpec(shape=["M"], dtype="float32"),
            "CCCCCC": TensorSpec(shape=["M"], dtype="float32"),
            "DDDDDD": TensorSpec(shape=["M"], dtype="float32"),
            "EEEEEE": TensorSpec(shape=["M"], dtype="float32"),
            "FFFFFF": TensorSpec(shape=["M"], dtype="float32"),
            "GGGGGG": TensorSpec(shape=["M"], dtype="float32"),
            "HHHHHH": TensorSpec(shape=["M"], dtype="float32"),
            "IIIIII": TensorSpec(shape=["M"], dtype="float32"),
            "JJJJJJ": TensorSpec(shape=["M"], dtype="float32"),
            "KKKKKK": TensorSpec(shape=["M"], dtype="float32"),
            "LLLLLL": TensorSpec(shape=["M"], dtype="float32"),
            "MMMMMM": TensorSpec(shape=["M"], dtype="float32"),
        },
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="""
def run(
    AAAAAA, BBBBBB, CCCCCC, DDDDDD, EEEEEE, FFFFFF, GGGGGG, HHHHHH, IIIIII, JJJJJJ,
    KKKKKK, LLLLLL, MMMMMM):
    return AAAAAA
""",
    )

    cpp_source = r"""
#include <torch/extension.h>
namespace py = pybind11;
torch::Tensor echo(
    torch::Tensor AAAAAA, torch::Tensor BBBBBB, torch::Tensor CCCCCC, torch::Tensor DDDDDD,
    torch::Tensor EEEEEE, torch::Tensor FFFFFF, torch::Tensor GGGGGG, torch::Tensor HHHHHH,
    torch::Tensor IIIIII, torch::Tensor JJJJJJ, torch::Tensor KKKKKK, torch::Tensor LLLLLL,
    torch::Tensor MMMMMM
) { return AAAAAA; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("echo", &echo); }
"""

    solution = Solution(
        name="cpu_echo_solution",
        definition="cpu_echo_def",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.CPP,
            target_hardware=["cpu"],
            entry_point="bind.cpp::echo",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="bind.cpp", content=cpp_source)],
    )
    runnable = builder.build(definition, solution)

    input_tensors = [
        torch.tensor([1, 2, 3], dtype=torch.float32, device="cpu"),
        torch.tensor([4, 5, 6], dtype=torch.float32, device="cpu"),
        torch.tensor([7, 8, 9], dtype=torch.float32, device="cpu"),
        torch.tensor([10, 11, 12], dtype=torch.float32, device="cpu"),
        torch.tensor([13, 14, 15], dtype=torch.float32, device="cpu"),
        torch.tensor([16, 17, 18], dtype=torch.float32, device="cpu"),
        torch.tensor([19, 20, 21], dtype=torch.float32, device="cpu"),
        torch.tensor([22, 23, 24], dtype=torch.float32, device="cpu"),
        torch.tensor([25, 26, 27], dtype=torch.float32, device="cpu"),
        torch.tensor([28, 29, 30], dtype=torch.float32, device="cpu"),
        torch.tensor([31, 32, 33], dtype=torch.float32, device="cpu"),
        torch.tensor([34, 35, 36], dtype=torch.float32, device="cpu"),
        torch.tensor([37, 38, 39], dtype=torch.float32, device="cpu"),
    ]

    # Warmup
    for _ in range(10):
        out = runnable(*input_tensors)
    # Measure time
    begin = time.monotonic()
    for _ in range(100):
        out = runnable(*input_tensors)
    end = time.monotonic()
    print(f"Time taken with runnable: {(end - begin) / 100} seconds")

    assert torch.allclose(out, input_tensors[0])


@pytest.mark.requires_torch_cuda
def test_cuda_vector_add():
    """Test building and running a simple CUDA vector add kernel."""
    definition = Definition(
        name="vec_add_cuda",
        op_type="op",
        axes={"N": AxisConst(value=256)},
        inputs={
            "X": TensorSpec(shape=["N"], dtype="float32"),
            "Y": TensorSpec(shape=["N"], dtype="float32"),
        },
        outputs={"Z": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(X: torch.Tensor, Y: torch.Tensor):\n    return X+Y\n",
    )

    cuda_kernel = r"""
#include <cuda_runtime.h>

extern "C" __global__ void vec_add_kernel(const float* X, const float* Y, float* Z, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Z[i] = X[i] + Y[i];
    }
}

extern "C" void launch_vec_add(const float* X, const float* Y, float* Z, int N) {
    int threads = 128;
    int blocks = (N + threads - 1) / threads;
    vec_add_kernel<<<blocks, threads>>>(X, Y, Z, N);
}
"""

    binding_cpp = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" void launch_vec_add(const float* X, const float* Y, float* Z, int N);

torch::Tensor vec_add(torch::Tensor X, torch::Tensor Y) {
    TORCH_CHECK(X.is_cuda() && Y.is_cuda(), "Inputs must be CUDA");
    TORCH_CHECK(X.is_contiguous() && Y.is_contiguous(), "Inputs must be contiguous");
    auto Z = torch::empty_like(X);
    int N = X.numel();
    launch_vec_add(X.data_ptr<float>(), Y.data_ptr<float>(), Z.data_ptr<float>(), N);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));
    return Z;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vec_add", &vec_add);
}
"""

    solution = Solution(
        name="cuda_vec_add",
        definition="vec_add_cuda",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=["cuda"],
            entry_point="binding.cpp::vec_add",
            destination_passing_style=False,
        ),
        sources=[
            SourceFile(path="kernel.cu", content=cuda_kernel),
            SourceFile(path="binding.cpp", content=binding_cpp),
        ],
    )

    builder = TorchBuilder()
    runnable = builder.build(definition, solution)
    X = torch.arange(256, dtype=torch.float32, device="cuda")
    Y = 2 * torch.ones(256, dtype=torch.float32, device="cuda")
    Z: torch.Tensor = runnable(X, Y)
    assert torch.allclose(Z, X + Y)


@pytest.mark.requires_torch_cuda
def test_cublas_matmul():
    definition = Definition(
        name="touch_cublas_matmul",
        op_type="op",
        axes={"M": AxisConst(value=2), "N": AxisConst(value=2), "K": AxisConst(value=2)},
        inputs={
            "X": TensorSpec(shape=["M", "K"], dtype="float32"),
            "Y": TensorSpec(shape=["N", "K"], dtype="float32"),
        },
        outputs={"Z": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="""
import torch
def run(X: torch.Tensor, Y: torch.Tensor):
    return X @ Y.T
""",
    )

    binding_cpp = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

torch::Tensor cublas_matmul(torch::Tensor X, torch::Tensor Y) {
    TORCH_CHECK(X.is_cuda() && Y.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(X.is_contiguous() && Y.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(X.dim() == 2 && Y.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(X.size(1) == Y.size(1), "For X @ Y.T, X cols must equal Y cols");

    int M = X.size(0);
    int K = X.size(1);
    int N = Y.size(0);

    auto Z = torch::empty({M, N}, X.options());

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS uses column-major, PyTorch uses row-major
    // Y_col = Y^T, we need Y = Y_col^T, so use CUBLAS_OP_T
    // X_col = X^T, we need X^T = X_col, so use CUBLAS_OP_N
    cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        Y.data_ptr<float>(), K,
        X.data_ptr<float>(), K,
        &beta,
        Z.data_ptr<float>(), N);

    cublasDestroy(handle);
    return Z;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cublas_matmul", &cublas_matmul);
}
"""

    solution = Solution(
        name="cublas_matmul",
        definition="touch_cublas_matmul",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=["cuda"],
            entry_point="binding.cpp::cublas_matmul",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="binding.cpp", content=binding_cpp)],
    )

    builder = TorchBuilder()
    runnable = builder.build(definition, solution)

    X = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device="cuda")
    Y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, device="cuda")
    Z: torch.Tensor = runnable(X, Y)
    expected = X @ Y.T
    assert torch.allclose(Z, expected), f"Expected {expected}, got {Z}"


if __name__ == "__main__":
    pytest.main(sys.argv)
