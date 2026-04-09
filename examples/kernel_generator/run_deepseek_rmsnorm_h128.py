"""Generate and benchmark one DeepSeek solution on flashinfer-trace rmsnorm_h128."""

import os
import random
import sys
from pathlib import Path


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    repo_root = Path("/data1/workspace/airulan/bench/flashinfer-bench")
    trace_root = Path(
        os.getenv("FLASHINFER_TRACE_PATH", "/data1/workspace/airulan/bench/flashinfer-trace")
    )

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "examples" / "kernel_generator"))

    from kernel_generator import KernelGenerator
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet
    from flashinfer_bench.data import save_json_file

    _require_env("LLM_API_KEY")
    _require_env("BASE_URL")

    model_name = os.getenv("MODEL_NAME", "deepseek-coder")
    language = os.getenv("FIB_LANGUAGE", "cuda")
    target_gpu = os.getenv("FIB_TARGET_GPU", "A800")
    definition_name = "rmsnorm_h128"
    gen_rounds = int(os.getenv("FIB_GEN_ROUNDS", "4"))

    print(f"Trace root: {trace_root}")
    print(f"Model: {model_name}")
    print(f"Language: {language}")
    print(f"Target GPU: {target_gpu}")
    print(f"Definition: {definition_name}")

    trace_set = TraceSet.from_path(trace_root)
    definition = trace_set.definitions[definition_name]

    generator = KernelGenerator(
        model_name=model_name,
        language=language,
        target_gpu=target_gpu,
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["BASE_URL"],
        use_ffi=False,
    )

    random.seed(0)
    solution = generator.generate(
        trace_set=trace_set,
        definition=definition,
        gen_rounds=gen_rounds,
        beam=False,
    )

    solution_dir = trace_root / "solutions" / solution.author / definition.op_type / definition_name
    solution_dir.mkdir(parents=True, exist_ok=True)
    solution_path = solution_dir / f"{solution.name}.json"
    save_json_file(solution, solution_path)

    print(f"Saved solution: {solution_path}")
    print(f"Solution name: {solution.name}")

    trace_set = TraceSet.from_path(trace_root)
    benchmark = Benchmark(
        trace_set,
        BenchmarkConfig(
            definitions=[definition_name],
            solutions=[solution.name],
            warmup_runs=5,
            iterations=20,
            num_trials=3,
            timeout_seconds=300,
        ),
    )

    try:
        benchmark.run_all(dump_traces=True, resume=False)
    finally:
        benchmark.close()

    print("Benchmark complete.")
    print(f"Trace file: {trace_root / 'traces' / definition.op_type / f'{definition_name}.jsonl'}")


if __name__ == "__main__":
    main()
