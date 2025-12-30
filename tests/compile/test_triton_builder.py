import sys
from pathlib import Path

import pytest
import torch

from flashinfer_bench.compile.builders import TritonBuilder
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


@pytest.mark.requires_torch_cuda
def test_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mock the import to make triton unavailable
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "triton":
            raise ImportError("Mocked: triton not available")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    assert not TritonBuilder.is_available()


@pytest.mark.requires_torch_cuda
def test_vector_add():
    definition = Definition(
        name="vec_add",
        op_type="op",
        axes={"N": AxisConst(value=256)},
        inputs={
            "X": TensorSpec(shape=["N"], dtype="float32"),
            "Y": TensorSpec(shape=["N"], dtype="float32"),
        },
        outputs={"Z": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(X, Y):\n    return X + Y",
    )

    triton_code = """
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x + y, mask=mask)

def run(X, Y):
    n = X.numel()
    Z = torch.empty_like(X)
    BLOCK = 128
    grid = lambda meta: ( (n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    add_kernel[grid](X, Y, Z, n, BLOCK_SIZE=BLOCK)
    return Z
"""

    solution = Solution(
        name="triton_vec_add",
        definition="vec_add",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.TRITON,
            target_hardware=["cuda"],
            entry_point="module/main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="module/main.py", content=triton_code)],
    )

    builder = TritonBuilder()
    runnable = builder.build(definition, solution)
    x_tensor = torch.arange(256, dtype=torch.float32, device="cuda")
    y_tensor = 2 * torch.ones(256, dtype=torch.float32, device="cuda")
    z_tensor = runnable(x_tensor, y_tensor)
    assert torch.allclose(z_tensor, x_tensor + y_tensor)


if __name__ == "__main__":
    pytest.main(sys.argv)
