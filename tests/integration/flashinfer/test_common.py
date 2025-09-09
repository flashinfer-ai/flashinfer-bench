import math

import torch

from flashinfer_bench.integration.flashinfer.common import (
    infer_kv_layout_from_args,
    infer_paged_kv_layout_from_tensors,
    infer_ragged_kv_layout_from_tensors,
    normalize_paged_kv_to_nhd,
    normalize_ragged_kv_to_nhd,
    pick_sm_scale_gqa,
    pick_sm_scale_mla,
    write_back_outputs,
)


class Dummy:
    pass


def test_infer_kv_layout_from_args_default_and_explicit():
    obj = Dummy()
    # default should be None (not inferred)
    assert infer_kv_layout_from_args(obj) is None
    obj.kv_layout = "HND"
    assert infer_kv_layout_from_args(obj) == "HND"


def test_infer_paged_kv_layout_from_tensors_tuple_and_5d():
    P, S, H, D = 2, 1, 3, 4
    k = torch.randn(P, S, H, D)
    v = torch.randn(P, S, H, D)
    assert infer_paged_kv_layout_from_tensors((k, v), num_kv_heads=H) == "NHD"
    # HND layout via tuple with [P, H, S, D]
    k_hnd = k.permute(0, 2, 1, 3)
    v_hnd = v.permute(0, 2, 1, 3)
    assert infer_paged_kv_layout_from_tensors((k_hnd, v_hnd), num_kv_heads=H) == "HND"

    x = torch.randn(P, 2, S, H, D)
    assert infer_paged_kv_layout_from_tensors(x, num_kv_heads=H) == "NHD"
    x_hnd = torch.randn(P, 2, H, S, D)
    assert infer_paged_kv_layout_from_tensors(x_hnd, num_kv_heads=H) == "HND"


def test_infer_ragged_kv_layout_and_normalize():
    total, H, D = 5, 3, 4
    k_nhd = torch.randn(total, H, D)
    assert infer_ragged_kv_layout_from_tensors(k_nhd, H) == "NHD"
    # HND layout via [H, total, D]
    k_hnd = k_nhd.permute(1, 0, 2)
    assert infer_ragged_kv_layout_from_tensors(k_hnd, H) == "HND"
    # normalize back to NHD
    k_norm = normalize_ragged_kv_to_nhd(k_hnd, "HND")
    assert k_norm.shape == (total, H, D)


def test_normalize_paged_kv_to_nhd_for_tuple_and_5d():
    P, S, H, D = 2, 1, 3, 4
    k = torch.randn(P, H, S, D)
    v = torch.randn(P, H, S, D)
    k_n, v_n = normalize_paged_kv_to_nhd((k, v), "HND")
    assert k_n.shape == (P, S, H, D) and v_n.shape == (P, S, H, D)

    x = torch.randn(P, 2, S, H, D)
    k2, v2 = normalize_paged_kv_to_nhd(x, "NHD")
    assert k2.shape == (P, S, H, D) and v2.shape == (P, S, H, D)


def test_pick_scales():
    # GQA
    assert math.isclose(pick_sm_scale_gqa(128, None), 1.0 / math.sqrt(128.0))
    assert math.isclose(pick_sm_scale_gqa(64, 0.5), 0.5)
    import torch as _t

    assert math.isclose(pick_sm_scale_gqa(64, _t.tensor(0.25)), 0.25)

    # MLA
    d = pick_sm_scale_mla(128, 64, None)
    assert isinstance(d, float) and math.isclose(d, 1.0 / math.sqrt(128 + 64))
    assert math.isclose(pick_sm_scale_mla(128, 64, 0.5), 0.5)
    import torch as _t

    assert math.isclose(pick_sm_scale_mla(128, 64, _t.tensor(0.25)), 0.25)


def test_write_back_outputs():
    # write_back_outputs behavior
    out = torch.randn(2, 3)
    lse = torch.randn(2, 3)
    out_buf = torch.zeros_like(out)
    lse_buf = torch.zeros_like(lse)

    # No lse requested
    ret = write_back_outputs(output=out, lse=lse, want_lse=False, out_buf=out_buf, lse_buf=lse_buf)
    assert torch.allclose(ret, out_buf) and torch.allclose(out_buf, out)

    # lse requested
    out2 = torch.randn(2, 3)
    lse2 = torch.randn(2, 3)
    ob2 = torch.zeros_like(out2)
    lb2 = torch.zeros_like(lse2)
    o3, l3 = write_back_outputs(output=out2, lse=lse2, want_lse=True, out_buf=ob2, lse_buf=lb2)
    assert torch.allclose(o3, ob2) and torch.allclose(l3, lb2)
    assert torch.allclose(ob2, out2) and torch.allclose(lb2, lse2)
