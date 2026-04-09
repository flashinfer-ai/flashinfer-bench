import sys

import pytest
import torch

from flashinfer_bench.integration.flashinfer.adapters import linear as linear_mod
from flashinfer_bench.integration.patch_manager import PatchSpec


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
def test_linear_adapter_reentrant_apply_bypasses_nested_dispatch(monkeypatch):
    device = torch.device("cuda")
    inp = torch.randn(2, 3, device=device, dtype=torch.float16)
    weight = torch.randn(4, 3, device=device, dtype=torch.float16)
    calls = {"apply": 0, "orig": 0}

    def orig(input, weight, bias=None):
        assert bias is None
        calls["orig"] += 1
        return torch.matmul(input, weight.t())

    adapter = linear_mod.LinearAdapter()
    wrapper = adapter.make_wrapper(
        PatchSpec(
            path="torch.nn.functional.linear",
            kind="function",
            name="linear",
            ctx_key="gemm_linear",
        ),
        orig,
    )

    def fake_apply(def_name, kwargs=None, fallback=None):
        del def_name, fallback
        calls["apply"] += 1
        return wrapper(kwargs["A"], kwargs["B"])

    monkeypatch.setattr(linear_mod, "apply", fake_apply)

    out = wrapper(inp, weight)
    expected = torch.matmul(inp, weight.t())
    assert torch.allclose(out, expected)
    assert calls == {"apply": 1, "orig": 1}


if __name__ == "__main__":
    pytest.main(sys.argv)
