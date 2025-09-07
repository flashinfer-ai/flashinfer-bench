from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch

from flashinfer_bench.apply.api import apply
from flashinfer_bench.apply.runtime import get_runtime

from ...patch_manager import PatchSpec
from ...utils import (
    ArgBinder,
    ContextStore,
    infer_kv_layout,
    pick_sm_scale,
    split_paged_kv_to_nhd,
    write_back_outputs,
)

def_name_resolver = (
    lambda q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale: f"gqa_paged_prefill_causal_h{q.shape[1]}_kv{k_cache.shape[2]}_d{q.shape[2]}_ps1"
)


class PrefillGqaPagedAdapter:
    """
    Adapter for flashinfer BatchPrefillWithPagedKVCacheWrapper(plan+run).
    Only covers causal=True and page_size=1.
    """

    def __init__(self) -> None:
        self._store = ContextStore()

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.plan",
                kind="method",
                name="prefill_plan",
                ctx_key="prefill",
            ),
            PatchSpec(
                path="flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run",
                kind="method",
                name="prefill_run",
                ctx_key="prefill",
            ),
        ]

    def make_wrapper(self, spec: PatchSpec, orig: Callable[..., Any]) -> Callable[..., Any]:
        if spec.name == "prefill_plan":
            binder = ArgBinder.from_callable(orig)

            def plan_wrapper(inst, *args, **kwargs):
                if get_runtime() is None:
                    return orig(inst, *args, **kwargs)

                bound = binder.bind((inst, *args), kwargs)
                ctx = self._store.get(inst)

                ctx["qo_indptr"] = bound["qo_indptr"]
                ctx["kv_indptr"] = bound["paged_kv_indptr"]
                ctx["kv_indices"] = bound["paged_kv_indices"]
                ctx["num_qo_heads"] = int(bound["num_qo_heads"])
                ctx["num_kv_heads"] = int(bound["num_kv_heads"])
                ctx["head_dim"] = int(bound["head_dim_qk"])
                ctx["page_size"] = int(bound["page_size"])
                ctx["causal"] = bool(bound.get("causal", False))
                ctx["kv_layout"] = infer_kv_layout(inst)
                ctx["sm_scale"] = bound.get("sm_scale", None)

                # Needs to call original anyways in case of run fallback
                return orig(inst, *args, **kwargs)

            return plan_wrapper

        elif spec.name == "prefill_run":
            # run
            binder = ArgBinder.from_callable(orig)

            def run_wrapper(inst, *args, **kwargs):
                if get_runtime() is None:
                    return orig(inst, *args, **kwargs)

                ctx = self._store.get(inst)
                # No plan context; fall back immediately
                if not ctx:
                    return orig(inst, *args, **kwargs)

                bound = binder.bind((inst, *args), kwargs)
                q: torch.Tensor = bound["q"]
                paged_kv_cache = bound["paged_kv_cache"]
                return_lse: bool = bool(bound.get("return_lse", False))
                out_buf = bound.get("out", None)
                lse_buf = bound.get("lse", None)

                # Compatibility checks (const axes & causal & page_size=1)
                if not ctx.get("causal", False):
                    return orig(inst, *args, **kwargs)
                if ctx.get("page_size", None) != 1:
                    return orig(inst, *args, **kwargs)

                num_qo_heads = ctx.get("num_qo_heads", None)
                num_kv_heads = ctx.get("num_kv_heads", None)
                head_dim = ctx.get("head_dim", None)
                if (num_qo_heads, num_kv_heads, head_dim) not in {(32, 8, 128), (32, 4, 128)}:
                    return orig(inst, *args, **kwargs)
                if q.dim() != 3 or q.shape[1] != num_qo_heads or q.shape[2] != head_dim:
                    return orig(inst, *args, **kwargs)

                # Normalize KV layout to NHD 4D views (no copy)
                kv_layout = ctx.get("kv_layout", "NHD")
                k_cache, v_cache = split_paged_kv_to_nhd(paged_kv_cache, kv_layout)

                # Assemble runtime kwargs (explicit keys)
                sm_scale = pick_sm_scale(head_dim, ctx.get("sm_scale"))
                rk: Dict[str, Any] = {
                    "q": q,
                    "k_cache": k_cache,
                    "v_cache": v_cache,
                    "qo_indptr": ctx["qo_indptr"],
                    "kv_indptr": ctx["kv_indptr"],
                    "kv_indices": ctx["kv_indices"],
                    "sm_scale": sm_scale,
                }

                # Fallback
                def _fb(**_rk):
                    return orig(inst, *args, **kwargs)

                ret = apply(def_name_resolver, runtime_kwargs=rk, fallback=_fb)

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
