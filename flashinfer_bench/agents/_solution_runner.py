"""Solution runner - standalone script for profiling solutions.

This module is used by profiling tools (NCU, compute-sanitizer, etc.).
It builds and executes a solution so the external tool can observe it.

Invocation:
    python -m flashinfer_bench.agents._solution_runner --data-dir <dir> --device <device>
"""

import argparse
from pathlib import Path

import torch

from flashinfer_bench.bench.evaluators.utils import allocate_outputs
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Definition, Solution, Workload


def main():
    parser = argparse.ArgumentParser(description="Run solution for profiling")
    parser.add_argument("--data-dir", required=True, help="Path to data directory")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to run on")
    parser.add_argument("--trace-set-path", help="Path to trace set")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = args.device
    trace_set_path = Path(args.trace_set_path) if args.trace_set_path else None

    # Load data from JSON files
    definition = Definition.model_validate_json((data_dir / "definition.json").read_text())
    solution = Solution.model_validate_json((data_dir / "solution.json").read_text())
    workload = Workload.model_validate_json((data_dir / "workload.json").read_text())

    # Build the solution
    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)

    # Load safetensors if needed
    safe_tensors = None
    if any(inp.type == "safetensors" for inp in workload.inputs.values()):
        safe_tensors = load_safetensors(definition, workload, trace_set_path)

    # Generate inputs
    inputs = gen_inputs(definition, workload, device, safe_tensors)

    # Allocate output tensors
    outputs = allocate_outputs(definition, inputs, device)

    # Warmup run to trigger JIT compilation
    with torch.no_grad():
        runnable.call_destination_passing(*inputs, *outputs)
    torch.cuda.synchronize()

    # Actual run for profiling (marked with NVTX for NCU filtering)
    with torch.cuda.nvtx.range("flashinfer_bench_ncu_profile"):
        with torch.no_grad():
            runnable.call_destination_passing(*inputs, *outputs)
        torch.cuda.synchronize()

    # Cleanup
    runnable.cleanup()


if __name__ == "__main__":
    main()
