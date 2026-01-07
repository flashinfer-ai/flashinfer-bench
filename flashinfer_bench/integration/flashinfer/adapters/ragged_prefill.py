from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.flashinfer.common import (
    infer_kv_layout_from_args,
    infer_ragged_kv_layout_from_tensors,
    normalize_ragged_kv_to_nhd,
    pick_sm_scale_gqa,
    write_back_outputs,
)
from flashinfer_bench.integration.patch_manager import PatchSpec
from flashinfer_bench.integration.utils import ArgBinder, ContextStore


def _def_name_resolver(q, k, v, qo_indptr, kv_indptr, sm_scale):
    return f"gqa_ragged_prefill_causal_h{q.shape[1]}_kv{k.shape[1]}_d{q.shape[2]}"


class RaggedPrefillAdapter:
    """Adapter for flashinfer BatchPrefillWithRaggedKVCacheWrapper(plan+run).
    Only covers causal=True. Used by both GQA and MLA ragged prefill.
    """

    def __init__(self) -> None:
        self._store = ContextStore()

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.plan",
                kind="method",
                name="ragged_prefill_plan",
                ctx_key="prefill_ragged",
            ),
            PatchSpec(
                path="flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.run",
                kind="method",
                name="ragged_prefill_run",
                ctx_key="prefill_ragged",
            ),
        ]

    def make_wrapper(self, spec: PatchSpec, orig: Callable[..., Any]) -> Callable[..., Any]:
        if spec.name == "ragged_prefill_plan":
            binder = ArgBinder.from_callable(orig)

            def plan_wrapper(inst, *args, **kwargs):
                bound = binder.bind((inst, *args), kwargs)
                ctx = self._store.get(inst)

                ctx["qo_indptr"] = bound["qo_indptr"]
                ctx["kv_indptr"] = bound["kv_indptr"]
                ctx["num_qo_heads"] = int(bound["num_qo_heads"])
                ctx["num_kv_heads"] = int(bound["num_kv_heads"])
                ctx["head_dim"] = int(bound["head_dim_qk"])
                ctx["causal"] = bool(bound.get("causal", False))
                ctx["kv_layout"] = infer_kv_layout_from_args(inst)
                ctx["sm_scale"] = pick_sm_scale_gqa(ctx["head_dim"], bound.get("sm_scale", None))

                # Needs to call original anyways in case of run fallback
                return orig(inst, *args, **kwargs)

            return plan_wrapper

        elif spec.name == "ragged_prefill_run":
            binder = ArgBinder.from_callable(orig)

            def run_wrapper(inst, *args, **kwargs):
                ctx = self._store.get(inst)
                if not ctx:
                    return orig(inst, *args, **kwargs)

                bound = binder.bind((inst, *args), kwargs)
                q: torch.Tensor = bound["q"]
                k: torch.Tensor = bound["k"]
                v: torch.Tensor = bound["v"]

                return_lse: bool = bool(bound.get("return_lse", False))
                out_buf = bound.get("out", None)
                lse_buf = bound.get("lse", None)

                # Only causal supported by captured kernels
                if not ctx.get("causal", False):
                    return orig(inst, *args, **kwargs)

                num_qo_heads = ctx.get("num_qo_heads", None)
                num_kv_heads = ctx.get("num_kv_heads", None)
                head_dim = ctx.get("head_dim", None)

                # Validate shapes
                if q.dim() != 3 or q.shape[1] != num_qo_heads or q.shape[2] != head_dim:
                    return orig(inst, *args, **kwargs)

                kv_layout = ctx.get("kv_layout", None)
                if not kv_layout:
                    kv_layout = infer_ragged_kv_layout_from_tensors(k, num_kv_heads)
                if not kv_layout:
                    return orig(inst, *args, **kwargs)

                k_nhd = normalize_ragged_kv_to_nhd(k, kv_layout)
                v_nhd = normalize_ragged_kv_to_nhd(v, kv_layout)

                sm_scale = ctx.get("sm_scale")

                rk: Dict[str, Any] = {
                    "q": q,
                    "k": k_nhd,
                    "v": v_nhd,
                    "qo_indptr": ctx["qo_indptr"],
                    "kv_indptr": ctx["kv_indptr"],
                    "sm_scale": sm_scale,
                }

                def _fallback(**_rk):
                    return orig(inst, *args, **kwargs)

                ret = apply(_def_name_resolver, kwargs=rk, fallback=_fallback)

                output = None
                lse = None
                if isinstance(ret, tuple):
                    if len(ret) == 2:
                        output, lse = ret
                    elif len(ret) == 1:
                        output = ret[0]
                else:
                    output = ret

                return write_back_outputs(
                    output=output, lse=lse, want_lse=return_lse, out_buf=out_buf, lse_buf=lse_buf
                )

            return run_wrapper
        else:
            return orig
