from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.flashinfer.common import (
    infer_kv_layout_from_args,
    infer_paged_kv_layout_from_tensors,
    normalize_paged_kv_to_nhd,
    pick_sm_scale_gqa,
    write_back_outputs,
)
from flashinfer_bench.integration.patch_manager import PatchSpec
from flashinfer_bench.integration.utils import ArgBinder, ContextStore


def _def_name_resolver(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, page_size):
    return f"gqa_paged_decode_h{q.shape[1]}_kv{k_cache.shape[2]}_d{q.shape[2]}_ps{page_size}"


class GQAPagedDecodeAdapter:
    """Adapter for flashinfer BatchDecodeWithPagedKVCacheWrapper(plan+run).
    Covers page_size=1 and 64 only for now.
    """

    def __init__(self) -> None:
        self._store = ContextStore()

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.plan",
                kind="method",
                name="gqa_paged_decode_plan",
                ctx_key="decode_gqa",
            ),
            PatchSpec(
                path="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
                kind="method",
                name="gqa_paged_decode_run",
                ctx_key="decode_gqa",
            ),
        ]

    def make_wrapper(self, spec: PatchSpec, orig: Callable[..., Any]) -> Callable[..., Any]:
        if spec.name == "gqa_paged_decode_plan":
            binder = ArgBinder.from_callable(orig)

            def plan_wrapper(inst, *args, **kwargs):
                bound = binder.bind((inst, *args), kwargs)
                ctx = self._store.get(inst)

                ctx["kv_indptr"] = bound["indptr"]
                ctx["kv_indices"] = bound["indices"]
                ctx["page_size"] = int(bound["page_size"])
                ctx["num_qo_heads"] = int(bound["num_qo_heads"])
                ctx["num_kv_heads"] = int(bound["num_kv_heads"])
                ctx["head_dim"] = int(bound["head_dim"])
                ctx["kv_layout"] = infer_kv_layout_from_args(inst)
                ctx["sm_scale"] = pick_sm_scale_gqa(ctx["head_dim"], bound.get("sm_scale", None))

                # Needs to call original anyways in case of run fallback
                return orig(inst, *args, **kwargs)

            return plan_wrapper

        elif spec.name == "gqa_paged_decode_run":
            binder = ArgBinder.from_callable(orig)

            def run_wrapper(inst, *args, **kwargs):
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

                # Compatibility checks
                page_size = ctx.get("page_size", None)
                if page_size not in (1, 64):
                    return orig(inst, *args, **kwargs)

                num_qo_heads = ctx.get("num_qo_heads", None)
                num_kv_heads = ctx.get("num_kv_heads", None)
                head_dim = ctx.get("head_dim", None)
                if q.dim() != 3 or q.shape[1] != num_qo_heads or q.shape[2] != head_dim:
                    return orig(inst, *args, **kwargs)

                # Best-effort KV layout detection if not exposed by wrapper
                kv_layout = ctx.get("kv_layout", None)
                if not kv_layout:
                    kv_layout = infer_paged_kv_layout_from_tensors(paged_kv_cache, num_kv_heads)
                if not kv_layout:
                    # Don't know kv layout; fall back
                    return orig(inst, *args, **kwargs)
                ctx["kv_layout"] = kv_layout

                k_cache, v_cache = normalize_paged_kv_to_nhd(paged_kv_cache, kv_layout)

                sm_scale = ctx.get("sm_scale")

                rk: Dict[str, Any] = {
                    "q": q,
                    "k_cache": k_cache,
                    "v_cache": v_cache,
                    "kv_indptr": ctx["kv_indptr"],
                    "kv_indices": ctx["kv_indices"],
                    "sm_scale": sm_scale,
                    "page_size": page_size,
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
