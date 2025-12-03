"""Tests for TVMFFIBuilder."""

import time
from pathlib import Path

import pytest
import torch

from flashinfer_bench.compile.builder import BuildError
from flashinfer_bench.compile.builders import TVMFFIBuilder
from flashinfer_bench.data import (
    AxisVar,
    BuildSpec,
    Definition,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
)

ADD_ONE_DEFINITION = Definition(
    name="add_one",
    op_type="test",
    description="Test CPU kernel that adds 1",
    axes={"n": AxisVar()},
    constraints=[],
    inputs={"x": TensorSpec(shape=["n"], dtype="float32")},
    outputs={"output": TensorSpec(shape=["n"], dtype="float32")},
    reference="def run(x): return x + 1",
)

CPP_ADD_ONE_SOURCE = """
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

CUDA_ADD_ONE_SOURCE = """
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


@pytest.fixture(autouse=True)
def _use_tmp_cache_dir(tmp_cache_dir: Path) -> None:
    """Automatically use tmp_cache_dir for all tests in this module."""


def test_cpu_add_one() -> None:
    """Test building and running a simple CPU kernel."""
    solution = Solution(
        name="test_add_one_cpu_impl",
        definition=ADD_ONE_DEFINITION.name,
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,  # TVMFFIBuilder accepts CUDA language
            target_hardware=["cpu"],
            entry_point="kernel.cpp::add_one_cpu",
        ),
        sources=[SourceFile(path="kernel.cpp", content=CPP_ADD_ONE_SOURCE)],
        description="Simple CPU add kernel",
    )

    # Build and run
    builder = TVMFFIBuilder()
    runnable = builder.build(ADD_ONE_DEFINITION, solution)

    # Test execution with torch tensors - TVM FFI functions use positional args and DPS style
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cpu", dtype=torch.float32)
    output_tensor = torch.empty_like(input_tensor)
    runnable(x=input_tensor, output=output_tensor)

    # Verify result
    expected = input_tensor + 1.0
    torch.testing.assert_close(output_tensor, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.requires_torch_cuda
def test_cuda_add_one() -> None:
    """Test building and running a simple CUDA kernel."""
    solution = Solution(
        name="test_add_one_cuda_impl",
        definition="test_add_one_cuda",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=["cuda"],
            entry_point="kernel.cu::add_one_cuda",
        ),
        sources=[SourceFile(path="kernel.cu", content=CUDA_ADD_ONE_SOURCE)],
        description="Simple CUDA add kernel",
    )

    # Build and run
    builder = TVMFFIBuilder()
    runnable = builder.build(ADD_ONE_DEFINITION, solution)

    # Test execution with torch tensors - TVM FFI functions use positional args and DPS style
    n = 1024
    input_tensor = torch.randn(n, device="cuda", dtype=torch.float32)
    output_tensor = torch.empty_like(input_tensor)
    runnable(x=input_tensor, output=output_tensor)

    # Verify result
    expected = input_tensor + 1.0
    torch.testing.assert_close(output_tensor, expected, rtol=1e-5, atol=1e-5)


def test_can_build() -> None:
    """Test that TVMFFIBuilder can build CUDA solutions."""
    builder = TVMFFIBuilder()

    cuda_solution = Solution(
        name="test_cuda",
        definition="test",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=["cuda"],
            entry_point="kernel.cu::test_func",
        ),
        sources=[SourceFile(path="kernel.cu", content="// dummy")],
        description="CUDA solution",
    )

    assert builder.can_build(cuda_solution)

    cpp_solution = Solution(
        name="test_cpp",
        definition="test",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CPP,
            target_hardware=["cpu"],
            entry_point="kernel.cpp::test_func",
        ),
        sources=[SourceFile(path="kernel.cpp", content="// dummy")],
        description="C++ solution",
    )
    assert builder.can_build(cpp_solution)

    python_solution = Solution(
        name="test_python",
        definition="test",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run(): pass")],
        description="Python solution",
    )

    assert not builder.can_build(python_solution)


def test_caching_cpu() -> None:
    """Test that compiled .so is cached and reused for CPU kernels."""
    solution = Solution(
        name="test_add_one_cpu_impl",
        definition="test_add_one_cpu_cached",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=["cpu"],
            entry_point="kernel.cpp::add_one_cpu",
        ),
        sources=[SourceFile(path="kernel.cpp", content=CPP_ADD_ONE_SOURCE)],
        description="CPU caching test",
    )

    # First build
    builder = TVMFFIBuilder()
    time_start = time.monotonic()
    runnable1 = builder.build(ADD_ONE_DEFINITION, solution)
    time_end = time.monotonic()
    print(f"Time taken to build: {(time_end - time_start) * 1000} ms")

    # Second build should load from cache
    time_start = time.monotonic()
    runnable2 = builder.build(ADD_ONE_DEFINITION, solution)
    time_end = time.monotonic()
    print(f"Time taken to load from cache: {(time_end - time_start) * 1000} ms")

    # Both should produce the same result - TVM FFI functions use positional args and DPS style
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cpu", dtype=torch.float32)

    output1 = torch.empty_like(input_tensor)
    output2 = torch.empty_like(input_tensor)
    runnable1(x=input_tensor, output=output1)
    runnable2(x=input_tensor, output=output2)

    torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(output1, input_tensor + 1.0, rtol=1e-5, atol=1e-5)


def test_caching_cross_builder() -> None:
    """Test that compiled .so is cached and reused for CPU kernels."""
    solution = Solution(
        name="test_add_one_cpu_cached",
        definition="test_add_one_cpu_cached",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=["cpu"],
            entry_point="kernel.cpp::add_one_cpu",
        ),
        sources=[SourceFile(path="kernel.cpp", content=CPP_ADD_ONE_SOURCE)],
        description="CPU caching test",
    )

    # First build
    builder1 = TVMFFIBuilder()
    time_start = time.monotonic()
    runnable1 = builder1.build(ADD_ONE_DEFINITION, solution)
    time_end = time.monotonic()
    print(f"Time taken to build: {(time_end - time_start) * 1000} ms")

    # Second build should load from cache
    builder2 = TVMFFIBuilder()
    time_start = time.monotonic()
    runnable2 = builder2.build(ADD_ONE_DEFINITION, solution)
    time_end = time.monotonic()
    print(f"Time taken to load from cache: {(time_end - time_start) * 1000} ms")

    # Both should produce the same result - TVM FFI functions use positional args and DPS style
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cpu", dtype=torch.float32)

    output1 = torch.empty_like(input_tensor)
    output2 = torch.empty_like(input_tensor)
    runnable1(x=input_tensor, output=output1)
    runnable2(x=input_tensor, output=output2)

    torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(output1, input_tensor + 1.0, rtol=1e-5, atol=1e-5)


def test_call_value_returning() -> None:
    """Test calling value-returning style with call_value_returning."""
    solution = Solution(
        name="test_add_one_cpu_impl",
        definition=ADD_ONE_DEFINITION.name,
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CPP,
            target_hardware=["cpu"],
            entry_point="kernel.cpp::add_one_cpu",
        ),
        sources=[SourceFile(path="kernel.cpp", content=CPP_ADD_ONE_SOURCE)],
        description="Add one kernel",
    )

    # Build
    builder = TVMFFIBuilder()
    runnable = builder.build(ADD_ONE_DEFINITION, solution)

    # Manually allocate input and output tensors
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cpu", dtype=torch.float32)
    output_tensor = torch.empty_like(input_tensor)

    # Call DPS style directly via the runnable (passes both input and output)
    output_tensor = runnable.call_value_returning(x=input_tensor)

    # Verify the output tensor was filled correctly
    expected = input_tensor + 1.0
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
            language=SupportedLanguages.CPP,
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
    # Create a dummy source file to pass validation, but it will fail at build time
    no_sources_solution = Solution(
        name="test_no_sources_impl",
        definition="test_no_sources",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CPP,
            target_hardware=["cpu"],
            entry_point="kernel.txt::func",  # Invalid extension
        ),
        sources=[SourceFile(path="kernel.txt", content="// not a valid C++ file")],
        description="No sources test",
    )

    builder = TVMFFIBuilder()
    with pytest.raises(BuildError, match="Either cpp_files or cuda_files must be provided"):
        builder.build(ADD_ONE_DEFINITION, no_sources_solution)


def test_source_in_subdirectory() -> None:
    """Test that source files in subdirectories are handled correctly."""
    # Place kernel in a subdirectory
    solution = Solution(
        name="test_subdirectory_impl",
        definition="test_subdirectory",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CPP,
            target_hardware=["cpu"],
            entry_point="subdir/kernel.cpp::add_one_cpu",
        ),
        sources=[SourceFile(path="subdir/kernel.cpp", content=CPP_ADD_ONE_SOURCE)],
        description="Test subdirectory handling",
    )

    # Build and run
    builder = TVMFFIBuilder()
    runnable = builder.build(ADD_ONE_DEFINITION, solution)

    # Test execution - TVM FFI functions use positional args and DPS style
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cpu", dtype=torch.float32)
    output_tensor = torch.empty_like(input_tensor)
    runnable(x=input_tensor, output=output_tensor)

    # Verify result
    expected = input_tensor + 1.0
    torch.testing.assert_close(output_tensor, expected, rtol=1e-5, atol=1e-5)


def test_rebuild_after_cleanup() -> None:
    """Test that rebuilding after cleanup takes longer than loading from cache."""
    solution = Solution(
        name="add_one_impl",
        definition="add_one",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CPP,
            target_hardware=["cpu"],
            entry_point="kernel.cpp::add_one_cpu",
        ),
        sources=[SourceFile(path="kernel.cpp", content=CPP_ADD_ONE_SOURCE)],
        description="Rebuild after cleanup test",
    )

    builder = TVMFFIBuilder()

    # First build (compile from scratch)
    t0 = time.monotonic()
    runnable1 = builder.build(ADD_ONE_DEFINITION, solution)
    first_build_time = time.monotonic() - t0
    print(f"First build time: {first_build_time * 1000:.2f} ms")

    # Second build (load from cache)
    t0 = time.monotonic()
    _ = builder.build(ADD_ONE_DEFINITION, solution)
    cached_load_time = time.monotonic() - t0
    print(f"Cached load time: {cached_load_time * 1000:.2f} ms")

    # Cleanup (removes the .so file)
    runnable1.cleanup()

    # Third build (recompile after cleanup)
    t0 = time.monotonic()
    runnable3 = builder.build(ADD_ONE_DEFINITION, solution)
    rebuild_time = time.monotonic() - t0
    print(f"Rebuild time after cleanup: {rebuild_time * 1000:.2f} ms")

    # Verify: rebuild should be slower than cached load
    assert rebuild_time > cached_load_time, (
        f"Rebuild time ({rebuild_time * 1000:.2f} ms) should be greater than "
        f"cached load time ({cached_load_time * 1000:.2f} ms)"
    )

    # Verify functionality still works
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cpu", dtype=torch.float32)
    output_tensor = torch.empty_like(input_tensor)
    runnable3(x=input_tensor, output=output_tensor)
    torch.testing.assert_close(output_tensor, input_tensor + 1.0, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
