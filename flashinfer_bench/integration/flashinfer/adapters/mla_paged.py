from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch

from flashinfer_bench.apply import apply
from flashinfer_bench.integration.flashinfer.common import pick_sm_scale_mla, write_back_outputs
from flashinfer_bench.integration.patch_manager import PatchSpec
from flashinfer_bench.integration.utils import ArgBinder, ContextStore


def _decode_def_name(
    q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, page_size
):
    # h16_ckv512_kpe64_ps1
    return (
        f"mla_paged_decode_h{q_nope.shape[1]}_ckv{q_nope.shape[2]}_kpe{q_pe.shape[2]}_ps{page_size}"
    )


def _prefill_def_name(
    q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale, page_size
):
    return f"mla_paged_prefill_causal_h{q_nope.shape[1]}_ckv{q_nope.shape[2]}_kpe{q_pe.shape[2]}_ps{page_size}"


class MLAPagedAdapter:
    """Adapter for flashinfer.mla.BatchMLAPagedAttentionWrapper(plan+run).
    - Detects decode vs (incremental) prefill by comparing qo_indptr vs q_nope batch size.
    - Covers page_size=1 only.
    """

    def __init__(self) -> None:
        self._store = ContextStore()

    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="flashinfer.mla.BatchMLAPagedAttentionWrapper.plan",
                kind="method",
                name="mla_paged_plan",
                ctx_key="mla_paged",
            ),
            PatchSpec(
                path="flashinfer.mla.BatchMLAPagedAttentionWrapper.run",
                kind="method",
                name="mla_paged_run",
                ctx_key="mla_paged",
            ),
        ]

    def make_wrapper(self, spec: PatchSpec, orig: Callable[..., Any]) -> Callable[..., Any]:
        if spec.name == "mla_paged_plan":
            binder = ArgBinder.from_callable(orig)

            def plan_wrapper(inst, *args, **kwargs):
                bound = binder.bind((inst, *args), kwargs)
                ctx = self._store.get(inst)

                ctx["qo_indptr"] = bound.get("qo_indptr", None)
                ctx["kv_indptr"] = bound.get("kv_indptr", None)
                ctx["kv_indices"] = bound.get("kv_indices", None)
                ctx["kv_len_arr"] = bound.get("kv_len_arr", None)
                ctx["num_heads"] = int(bound.get("num_heads"))
                ctx["head_dim_ckv"] = int(bound.get("head_dim_ckv"))
                ctx["head_dim_kpe"] = int(bound.get("head_dim_kpe"))
                ctx["page_size"] = int(bound.get("page_size"))
                ctx["causal"] = bool(bound.get("causal", False))
                ctx["sm_scale"] = pick_sm_scale_mla(
                    128, ctx["head_dim_kpe"], bound.get("sm_scale", None)
                )

                # Needs to call original anyways in case of run fallback
                return orig(inst, *args, **kwargs)

            return plan_wrapper

        elif spec.name == "mla_paged_run":
            binder = ArgBinder.from_callable(orig)

            def run_wrapper(inst, *args, **kwargs):
                ctx = self._store.get(inst)
                if not ctx:
                    return orig(inst, *args, **kwargs)

                bound = binder.bind((inst, *args), kwargs)
                q_nope: torch.Tensor = bound["q_nope"]
                q_pe: torch.Tensor = bound["q_pe"]
                ckv_cache: torch.Tensor = bound["ckv_cache"]
                kpe_cache: torch.Tensor = bound["kpe_cache"]

                return_lse: bool = bool(bound.get("return_lse", False))
                out_buf = bound.get("out", None)
                lse_buf = bound.get("lse", None)

                # Compatibility checks
                if not ctx.get("causal", False):
                    return orig(inst, *args, **kwargs)
                page_size = ctx.get("page_size", None)
                if page_size not in (1, 64):
                    return orig(inst, *args, **kwargs)

                H = ctx.get("num_heads", None)
                D_ckv = ctx.get("head_dim_ckv", None)
                D_kpe = ctx.get("head_dim_kpe", None)
                if (H, D_ckv, D_kpe) != (16, 512, 64):
                    return orig(inst, *args, **kwargs)
                if (
                    q_nope.dim() != 3
                    or q_nope.shape[1] != H
                    or q_nope.shape[2] != D_ckv
                    or q_pe.shape[1] != H
                    or q_pe.shape[2] != D_kpe
                ):
                    return orig(inst, *args, **kwargs)

                # Determine decode vs prefill using qo_indptr
                qo_indptr = ctx.get("qo_indptr", None)
                kv_indptr = ctx.get("kv_indptr", None)
                kv_indices = ctx.get("kv_indices", None)
                if qo_indptr is None or kv_indptr is None or kv_indices is None:
                    return orig(inst, *args, **kwargs)

                len_indptr = int(qo_indptr.shape[0])
                batch_size = len_indptr - 1
                # decode if q batch equals batch_size
                is_decode = q_nope.shape[0] == batch_size

                sm_scale = ctx.get("sm_scale")

                if is_decode:
                    def_name = _decode_def_name(
                        q_nope,
                        q_pe,
                        ckv_cache,
                        kpe_cache,
                        kv_indptr,
                        kv_indices,
                        sm_scale,
                        page_size,
                    )
                else:
                    def_name = _prefill_def_name(
                        q_nope,
                        q_pe,
                        ckv_cache,
                        kpe_cache,
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        sm_scale,
                        page_size,
                    )

                if is_decode:
                    kwargs: Dict[str, Any] = {
                        "q_nope": q_nope,
                        "q_pe": q_pe,
                        "ckv_cache": ckv_cache,
                        "kpe_cache": kpe_cache,
                        "kv_indptr": kv_indptr,
                        "kv_indices": kv_indices,
                        "sm_scale": sm_scale,
                    }
                else:
                    kwargs = {
                        "q_nope": q_nope,
                        "q_pe": q_pe,
                        "ckv_cache": ckv_cache,
                        "kpe_cache": kpe_cache,
                        "qo_indptr": qo_indptr,
                        "kv_indptr": kv_indptr,
                        "kv_indices": kv_indices,
                        "sm_scale": sm_scale,
                    }

                def _fallback(**_rk):
                    return orig(inst, *args, **kwargs)

                ret = apply(def_name, kwargs=kwargs, fallback=_fallback)

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
