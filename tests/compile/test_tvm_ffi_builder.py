"""Tests for TVMFFIBuilder."""

import os
import time

import pytest
import torch

from flashinfer_bench.compile.builder import BuildError
from flashinfer_bench.compile.builders.tvm_ffi_builder import TVMFFIBuilder
from flashinfer_bench.data import BuildSpec, Definition, Solution, SourceFile, SupportedLanguages

# ============================================================================
# CPU Tests
# ============================================================================


def test_build_cpp_cpu() -> None:
    """Test building and running a simple CPU kernel."""
    # CPU kernel source - destination passing style
    cpp_source = """
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView output) {
    TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
    TVM_FFI_ICHECK(output.ndim() == 1) << "output must be a 1D tensor";
    TVM_FFI_ICHECK(x.size(0) == output.size(0)) << "x and output must have the same size";

    DLDataType f32_dtype{kDLFloat, 32, 1};
    TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be float32";
    TVM_FFI_ICHECK(output.dtype() == f32_dtype) << "output must be float32";

    // Compute
    for (int i = 0; i < x.size(0); ++i) {
        static_cast<float*>(output.data_ptr())[i] =
            static_cast<float*>(x.data_ptr())[i] + 1.0f;
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cpu, add_one_cpu);
"""

    # Create definition and solution
    n = 5
    definition = Definition(
        name="test_add_one_cpu",
        op_type="test",
        description="Test CPU kernel that adds 1",
        axes={"n": {"type": "const", "value": n}},
        constraints=[],
        inputs={"x": {"shape": ["n"], "dtype": "float32"}},
        outputs={"output": {"shape": ["n"], "dtype": "float32"}},
        reference="def run(x): return x + 1",
    )

    solution = Solution(
        name="test_add_one_cpu_impl",
        definition="test_add_one_cpu",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,  # TVMFFIBuilder accepts CUDA language
            target_hardware=["cpu"],
            entry_point="kernel.cpp::add_one_cpu",
        ),
        sources=[SourceFile(path="kernel.cpp", content=cpp_source)],
        description="Simple CPU add kernel",
    )

    # Build and run
    builder = TVMFFIBuilder()
    runnable = builder.build(definition, solution)

    # Test execution with torch tensors - runnable returns output
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cpu", dtype=torch.float32)
    output_tensor = runnable(x=input_tensor)

    # Verify result
    expected = input_tensor + 1.0
    torch.testing.assert_close(output_tensor, expected, rtol=1e-5, atol=1e-5)


# ============================================================================
# CUDA Tests
# ============================================================================


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip GPU tests in CI")
def test_build_cuda_gpu() -> None:
    """Test building and running a simple CUDA kernel."""
    import torch

    # CUDA kernel source - destination passing style
    cuda_source = """
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

__global__ void add_one_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + 1.0f;
    }
}

void add_one_cuda(tvm::ffi::TensorView x, tvm::ffi::TensorView output) {
    TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
    TVM_FFI_ICHECK(output.ndim() == 1) << "output must be a 1D tensor";
    TVM_FFI_ICHECK(x.size(0) == output.size(0)) << "x and output must have the same size";

    int n = x.size(0);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    add_one_kernel<<<blocks, threads>>>(
        static_cast<const float*>(x.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        n
    );
    cudaDeviceSynchronize();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cuda, add_one_cuda);
"""

    definition = Definition(
        name="test_add_one_cuda",
        op_type="test",
        description="Test CUDA kernel that adds 1",
        axes={"n": {"type": "const", "value": 1024}},
        constraints=[],
        inputs={"x": {"shape": ["n"], "dtype": "float32"}},
        outputs={"output": {"shape": ["n"], "dtype": "float32"}},
        reference="def run(x): return x + 1",
    )

    solution = Solution(
        name="test_add_one_cuda_impl",
        definition="test_add_one_cuda",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=["gpu"],
            entry_point="kernel.cu::add_one_cuda",
        ),
        sources=[SourceFile(path="kernel.cu", content=cuda_source)],
        description="Simple CUDA add kernel",
    )

    # Build and run
    builder = TVMFFIBuilder()
    runnable = builder.build(definition, solution)

    # Test execution with torch tensors - runnable returns output
    n = 1024
    input_tensor = torch.randn(n, device="cuda", dtype=torch.float32)
    output_tensor = runnable(x=input_tensor)

    # Verify result
    expected = input_tensor + 1.0
    torch.testing.assert_close(output_tensor, expected, rtol=1e-5, atol=1e-5)


# ============================================================================
# Basic Functionality Tests
# ============================================================================


def test_can_build_cuda() -> None:
    """Test that TVMFFIBuilder can build CUDA solutions."""
    builder = TVMFFIBuilder()

    cuda_solution = Solution(
        name="test_cuda",
        definition="test",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=["gpu"],
            entry_point="kernel.cu::test_func",
        ),
        sources=[SourceFile(path="kernel.cu", content="// dummy")],
        description="CUDA solution",
    )

    assert builder.can_build(cuda_solution)


def test_can_build_non_cuda() -> None:
    """Test that TVMFFIBuilder rejects non-CUDA solutions."""
    builder = TVMFFIBuilder()

    python_solution = Solution(
        name="test_python",
        definition="test",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["gpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run(): pass")],
        description="Python solution",
    )

    assert not builder.can_build(python_solution)


def test_caching_cpu() -> None:
    """Test that compiled .so is cached and reused for CPU kernels."""
    cpp_source = """
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/function.h>

void add_two_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView output) {
    for (int i = 0; i < x.size(0); ++i) {
        static_cast<float*>(output.data_ptr())[i] =
            static_cast<float*>(x.data_ptr())[i] + 2.0f;
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_two_cpu, add_two_cpu);
"""

    definition = Definition(
        name="test_add_two_cpu",
        op_type="test",
        description="Test CPU caching",
        axes={"n": {"type": "const", "value": 5}},
        constraints=[],
        inputs={"x": {"shape": ["n"], "dtype": "float32"}},
        outputs={"output": {"shape": ["n"], "dtype": "float32"}},
        reference="def run(x): return x + 2",
    )

    solution = Solution(
        name="test_add_two_cpu_cached",
        definition="test_add_two_cpu",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=["cpu"],
            entry_point="kernel.cpp::add_two_cpu",
        ),
        sources=[SourceFile(path="kernel.cpp", content=cpp_source)],
        description="CPU caching test",
    )

    # First build
    builder = TVMFFIBuilder()
    time_start = time.monotonic()
    runnable1 = builder.build(definition, solution)
    time_end = time.monotonic()
    print(f"Time taken to build: {(time_end - time_start) * 1000} ms")

    # Second build should load from cache
    time_start = time.monotonic()
    runnable2 = builder.build(definition, solution)
    time_end = time.monotonic()
    print(f"Time taken to load from cache: {(time_end - time_start) * 1000} ms")

    # Both should produce the same result
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cpu", dtype=torch.float32)

    output1 = runnable1(x=input_tensor)
    output2 = runnable2(x=input_tensor)

    torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(output1, input_tensor + 2.0, rtol=1e-5, atol=1e-5)


def test_call_dest_cpu() -> None:
    """Test calling call_dest directly with pre-allocated output tensors."""
    cpp_source = """
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/function.h>

void multiply_by_two(tvm::ffi::TensorView x, tvm::ffi::TensorView output) {
    for (int i = 0; i < x.size(0); ++i) {
        static_cast<float*>(output.data_ptr())[i] =
            static_cast<float*>(x.data_ptr())[i] * 2.0f;
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(multiply_by_two, multiply_by_two);
"""

    n = 4
    definition = Definition(
        name="test_multiply_by_two",
        op_type="test",
        description="Test multiply by two",
        axes={"n": {"type": "const", "value": n}},
        constraints=[],
        inputs={"x": {"shape": ["n"], "dtype": "float32"}},
        outputs={"output": {"shape": ["n"], "dtype": "float32"}},
        reference="def run(x): return x * 2",
    )

    solution = Solution(
        name="test_multiply_by_two_impl",
        definition="test_multiply_by_two",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=["cpu"],
            entry_point="kernel.cpp::multiply_by_two",
        ),
        sources=[SourceFile(path="kernel.cpp", content=cpp_source)],
        description="Multiply by two kernel",
    )

    # Build
    builder = TVMFFIBuilder()
    runnable = builder.build(definition, solution)

    # Manually allocate input and output tensors
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cpu", dtype=torch.float32)
    output_tensor = torch.empty(n, device="cpu", dtype=torch.float32)

    # Call call_dest directly
    runnable.call_dest(x=input_tensor, output=output_tensor)

    # Verify the output tensor was filled correctly
    expected = input_tensor * 2.0
    torch.testing.assert_close(output_tensor, expected, rtol=1e-5, atol=1e-5)


def test_invalid_entry_point() -> None:
    """Test error handling for missing entry point symbol."""
    cpp_source = """
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

void actual_function(tvm::ffi::TensorView x) {}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(actual_function, actual_function);
"""

    definition = Definition(
        name="test_invalid",
        op_type="test",
        description="Test invalid entry point",
        axes={"n": {"type": "const", "value": 1}},
        constraints=[],
        inputs={"x": {"shape": ["n"], "dtype": "float32"}},
        outputs={},
        reference="def run(x): return x",
    )

    invalid_solution = Solution(
        name="test_invalid_entry",
        definition="test_invalid",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=["cpu"],
            entry_point="kernel.cpp::nonexistent_function",
        ),
        sources=[SourceFile(path="kernel.cpp", content=cpp_source)],
        description="Invalid entry point test",
    )

    builder = TVMFFIBuilder()
    with pytest.raises(BuildError):
        builder.build(definition, invalid_solution)


def test_no_sources() -> None:
    """Test error handling when no sources provided."""
    definition = Definition(
        name="test_no_sources",
        op_type="test",
        description="Test no sources",
        axes={"n": {"type": "const", "value": 1}},
        constraints=[],
        inputs={"x": {"shape": ["n"], "dtype": "float32"}},
        outputs={},
        reference="def run(x): return x",
    )

    # Create a dummy source file to pass validation, but it will fail at build time
    no_sources_solution = Solution(
        name="test_no_sources_impl",
        definition="test_no_sources",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=["cpu"],
            entry_point="kernel.txt::func",  # Invalid extension
        ),
        sources=[SourceFile(path="kernel.txt", content="// not a valid C++ file")],
        description="No sources test",
    )

    builder = TVMFFIBuilder()
    with pytest.raises(BuildError, match="No sources"):
        builder.build(definition, no_sources_solution)


if __name__ == "__main__":
    pytest.main([__file__])
