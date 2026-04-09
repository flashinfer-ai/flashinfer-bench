from __future__ import annotations

import torch

from flashinfer_bench.integration.flashinfer.adapters.rmsnorm import RMSNormAdapter


def test_rmsnorm_adapter_applies_in_place(monkeypatch):
    adapter = RMSNormAdapter()

    def fake_apply(def_name, *, kwargs, fallback):
        assert def_name == "fused_add_rmsnorm_h4096"
        hidden = kwargs["hidden_states"]
        residual = kwargs["residual"]
        weight = kwargs["weight"]
        x = hidden.float() + residual.float()
        out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        return (out * weight.float()).to(hidden.dtype)

    monkeypatch.setattr(
        "flashinfer_bench.integration.flashinfer.adapters.rmsnorm.apply",
        fake_apply,
    )

    def orig(input, residual, weight, eps=1e-5):
        raise AssertionError("fallback should not be used")

    wrapper = adapter.make_wrapper(adapter.targets()[0], orig)
    x = torch.randn(3, 4096, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(4096, dtype=torch.bfloat16)

    x_before = x.clone()
    residual_before = residual.clone()
    expected_residual = residual_before.float() + x_before.float()
    expected_output = expected_residual * torch.rsqrt(expected_residual.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    expected_output = (expected_output * weight.float()).to(torch.bfloat16)

    result = wrapper(x, residual, weight, 1e-5)

    assert result is None
    torch.testing.assert_close(x, expected_output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(residual, expected_residual.to(torch.bfloat16), atol=1e-2, rtol=1e-2)


def test_rmsnorm_adapter_falls_back_for_non_matching_eps():
    adapter = RMSNormAdapter()
    called = {"fallback": False}

    def orig(input, residual, weight, eps=1e-6):
        called["fallback"] = True
        input.add_(1)

    wrapper = adapter.make_wrapper(adapter.targets()[0], orig)
    x = torch.zeros(1, 4096, dtype=torch.bfloat16)
    residual = torch.zeros_like(x)
    weight = torch.ones(4096, dtype=torch.bfloat16)

    wrapper(x, residual, weight, 1e-6)

    assert called["fallback"] is True
    assert x[0, 0].item() == 1


def test_rmsnorm_adapter_bypasses_apply_for_flashinfer_baseline_reentry(monkeypatch):
    adapter = RMSNormAdapter()

    monkeypatch.setattr(
        "flashinfer_bench.integration.flashinfer.adapters.rmsnorm._active_flashinfer_baseline_solution",
        lambda: True,
    )

    def fake_apply(*args, **kwargs):
        raise AssertionError("apply should be bypassed for flashinfer baseline re-entry")

    monkeypatch.setattr(
        "flashinfer_bench.integration.flashinfer.adapters.rmsnorm.apply",
        fake_apply,
    )

    called = {"orig": 0}

    def orig(input, residual, weight, eps=1e-5):
        called["orig"] += 1
        residual.add_(input)
        input.copy_(residual)
        return None

    wrapper = adapter.make_wrapper(adapter.targets()[0], orig)
    x = torch.ones(1, 4096, dtype=torch.bfloat16)
    residual = torch.zeros_like(x)
    weight = torch.ones(4096, dtype=torch.bfloat16)

    result = wrapper(x, residual, weight, 1e-5)

    assert result is None
    assert called["orig"] == 1
    assert x[0, 0].item() == 1
    assert residual[0, 0].item() == 1


def test_rmsnorm_adapter_accepts_in_place_value_returning_solution(monkeypatch):
    adapter = RMSNormAdapter()

    def fake_apply(def_name, *, kwargs, fallback):
        hidden = kwargs["hidden_states"]
        residual = kwargs["residual"]
        hidden_before = hidden.clone()
        residual.add_(hidden_before)
        hidden.fill_(7)
        return hidden

    monkeypatch.setattr(
        "flashinfer_bench.integration.flashinfer.adapters.rmsnorm.apply",
        fake_apply,
    )

    def orig(input, residual, weight, eps=1e-5):
        raise AssertionError("fallback should not be used")

    wrapper = adapter.make_wrapper(adapter.targets()[0], orig)
    x = torch.ones(1, 4096, dtype=torch.bfloat16)
    residual = torch.full_like(x, 2)
    weight = torch.ones(4096, dtype=torch.bfloat16)

    result = wrapper(x, residual, weight, 1e-5)

    assert result is None
    assert x[0, 0].item() == 7
    assert residual[0, 0].item() == 3
