import torch

from flashinfer_bench.apply import ApplyConfig, ApplyRuntime, set_apply_runtime
from flashinfer_bench.data import (
    AxisConst,
    AxisVar,
    BuildSpec,
    Correctness,
    Definition,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
)


def test_rmsnorm_adapter_substitution(tmp_path, monkeypatch):
    """
    Integration-style test that mirrors test_ragged_prefill_adapter_substitution.

    It constructs a minimal TraceSet containing a python solution for a
    definition whose name matches the RMSNorm adapter resolver
    (fused_add_rmsnorm_h{hidden}). Then it installs an ApplyRuntime and calls
    flashinfer.norm.fused_add_rmsnorm(...) to ensure the adapter dispatches to
    the python solution returning the sentinel string.
    """

    import flashinfer  # type: ignore

    # small shapes
    B = 2
    H = 8

    device = torch.device("cpu")
    dtype = torch.bfloat16

    # build tensors matching adapter's expected arg names: input, residual, weight
    inp = torch.zeros((B, H), dtype=dtype, device=device)
    res = torch.zeros((B, H), dtype=dtype, device=device)
    w = torch.zeros((H,), dtype=dtype, device=device)

    def_name = f"fused_add_rmsnorm_h{H}"

    definition = Definition(
        name=def_name,
        op_type="rmsnorm",
        axes={"M": AxisVar(), "H": AxisConst(value=H)},
        inputs={
            "hidden_states": TensorSpec(shape=["M", "H"], dtype="bfloat16"),
            "residual": TensorSpec(shape=["M", "H"], dtype="bfloat16"),
            "weight": TensorSpec(shape=["H"], dtype="bfloat16"),
        },
        outputs={"output": TensorSpec(shape=["M", "H"], dtype="bfloat16")},
        reference=("def run(hidden_states, residual, weight):\n    return hidden_states\n"),
    )

    sol_src = SourceFile(
        path="main.py",
        content=(
            "import torch\n"
            "def run(hidden_states, residual, weight):\n"
            "    return '__SUB__rmsnorm__'\n"
        ),
    )

    solution = Solution(
        name=f"{def_name}__python_direct_call",
        definition=def_name,
        author="ut",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[sol_src],
        description="Tests",
    )

    workload = Workload(
        axes={"M": B}, inputs={"hidden_states": RandomInput(), "residual": RandomInput()}, uuid="w0"
    )
    trace = Trace(
        definition=def_name,
        workload=workload,
        solution=solution.name,
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            log="/dev/null",
            environment=Environment(hardware="cpu", libs={}),
            timestamp="now",
            correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
            performance=Performance(latency_ms=1.0, reference_latency_ms=2.0, speedup_factor=2.0),
        ),
    )

    trace_set = TraceSet(
        root=tmp_path,
        definitions={def_name: definition},
        solutions={def_name: [solution]},
        traces={def_name: [trace]},
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))
    rt = ApplyRuntime(trace_set, ApplyConfig())
    set_apply_runtime(rt)

    # Call the function in the real flashinfer package; adapter should patch it
    out = flashinfer.norm.fused_add_rmsnorm(inp, res, w)
    assert out == "__SUB__rmsnorm__"
