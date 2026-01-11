"""NCU runner - standalone script for profiling solutions."""

import argparse
import json
from pathlib import Path

import torch

from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Definition, Solution, Workload


def main():
    parser = argparse.ArgumentParser(description="Run a solution for NCU profiling")
    parser.add_argument("--data", required=True, help="Path to JSON data file")
    args = parser.parse_args()

    # Load data from JSON file
    data = json.loads(Path(args.data).read_text())

    definition = Definition.model_validate(data["definition"])
    solution = Solution.model_validate(data["solution"])
    workload = Workload.model_validate(data["workload"])
    trace_set_root = Path(data["trace_set_root"]) if data["trace_set_root"] else None
    device = data["device"]

    # Build the solution
    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)

    # Load safetensors if needed
    safe_tensors = None
    if any(inp.type == "safetensors" for inp in workload.inputs.values()):
        safe_tensors = load_safetensors(definition, workload, trace_set_root)

    # Generate inputs
    inputs = gen_inputs(definition, workload, device, safe_tensors)

    # Warmup run to trigger JIT compilation
    with torch.no_grad():
        runnable.call_value_returning(*inputs)
    torch.cuda.synchronize()

    # Actual run for profiling
    with torch.no_grad():
        result = runnable.call_value_returning(*inputs)
    torch.cuda.synchronize()

    print(result)


if __name__ == "__main__":
    main()
