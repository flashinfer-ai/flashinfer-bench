from flashinfer_bench.compile.builder import BuildError
from flashinfer_bench.compile.builders import CUDABuilder
from flashinfer_bench.data.definition import AxisConst, Definition, TensorSpec
from flashinfer_bench.data.solution import BuildSpec, Solution, SourceFile, SupportedLanguages


def test_cuda_builder_minimum(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_DIR", str(cache_dir))

    b = CUDABuilder()
    d = Definition(
        name="d",
        type="op",
        axes={"M": AxisConst(value=1)},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="def run(A):\n    return A\n",
    )
    spec = BuildSpec(
        language=SupportedLanguages.CUDA, target_hardware=["gpu"], entry_point="bind.cpp::echo"
    )
    cpp_source = r"""
        #include <pybind11/pybind11.h>
        namespace py = pybind11;
        py::object echo(py::object A) { return A; }
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("echo", &echo); }
    """
    cu_source = r"""
    extern "C" __global__ void _fi_dummy_kernel_() {}
    """
    srcs = [
        SourceFile(path="bind.cpp", content=cpp_source),
        SourceFile(path="dummy.cu", content=cu_source),
    ]
    s = Solution(name="cuda_ok", definition="d", author="a", spec=spec, sources=srcs)
    r = b.build(d, s)
    out = r(A=[1, 2, 3])
    assert out == [1, 2, 3]


def test_cuda_vector_add(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_DIR", str(cache_dir))

    defn = Definition(
        name="vec_add_cuda",
        type="op",
        axes={"N": AxisConst(value=256)},
        inputs={
            "X": TensorSpec(shape=["N"], dtype="float32"),
            "Y": TensorSpec(shape=["N"], dtype="float32"),
        },
        outputs={"Z": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(X: torch.Tensor, Y: torch.Tensor):\n    return X+Y\n",
    )

    # A simple CUDA kernel and a CUDA-side launcher
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

    spec = BuildSpec(
        language=SupportedLanguages.CUDA,
        target_hardware=["gpu"],
        entry_point="binding.cpp::vec_add",
    )
    srcs = [
        SourceFile(path="kernel.cu", content=cuda_kernel),
        SourceFile(path="binding.cpp", content=binding_cpp),
    ]
    sol = Solution(
        name="cuda_vec_add", definition="vec_add_cuda", author="tester", spec=spec, sources=srcs
    )

    b = CUDABuilder()
    r = b.build(defn, sol)

    import torch

    X = torch.arange(256, dtype=torch.float32, device="cuda")
    Y = 2 * torch.ones(256, dtype=torch.float32, device="cuda")
    Z = r(X=X, Y=Y)
    assert torch.allclose(Z, X + Y)


def test_cuda_references_cublas(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_DIR", str(cache_dir))

    defn = Definition(
        name="touch_cublas",
        type="op",
        axes={"N": AxisConst(value=1)},
        inputs={"X": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"Y": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\n" "def run(X: torch.Tensor):\n    return X\n",
    )

    binding_cpp = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

torch::Tensor touch_cublas(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda(), "Input must be CUDA");
    cublasHandle_t handle;
    cublasStatus_t st = cublasCreate(&handle);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasCreate failed");
    cublasDestroy(handle);
    return X;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("touch_cublas", &touch_cublas);
}
"""

    # Provide at least one CUDA TU so the CUDA toolchain is used
    dummy_cu = r"""
extern "C" __global__ void _fi_dummy_kernel__() {}
"""

    spec = BuildSpec(
        language=SupportedLanguages.CUDA,
        target_hardware=["gpu"],
        entry_point="binding.cpp::touch_cublas",
    )
    srcs = [
        SourceFile(path="binding.cpp", content=binding_cpp),
        SourceFile(path="dummy.cu", content=dummy_cu),
    ]
    sol = Solution(
        name="cuda_touch_cublas",
        definition="touch_cublas",
        author="tester",
        spec=spec,
        sources=srcs,
    )

    b = CUDABuilder()
    r = b.build(defn, sol)

    import torch

    X = torch.ones(1, dtype=torch.float32, device="cuda")
    Y = r(X=X)
    assert Y.is_cuda and Y.numel() == 1 and Y.item() == 1.0
