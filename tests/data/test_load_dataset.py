#!/usr/bin/env python3
"""Test script to load and validate data files from the dataset folder."""

import json
from pathlib import Path

from flashinfer_bench import BenchmarkConfig, Definition, Solution, Trace
from flashinfer_bench.utils.json_utils import load_json, load_jsonl


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}\n")


def test_load_definition():
    """Load and test the GEMM definition."""
    print_section("Loading Definition from dataset/definitions/gemm.json")

    # Load directly from JSON file
    definition_path = Path("dataset/definitions/gemm.json")
    definition = load_json(definition_path, Definition)

    print(f"✓ Loaded definition: {definition.name}")
    print(f"  Type: {definition.type}")
    print(f"  Description: {definition.description}")
    print(f"  Axes: {list(definition.axes.keys())}")
    print(f"    Variable axes: {definition.get_var_axes()}")
    print(f"    Constant axes: {definition.get_const_axes()}")
    print(f"  Inputs: {list(definition.inputs.keys())}")
    print(f"  Outputs: {list(definition.outputs.keys())}")

    # Test getting concrete shapes
    shapes = definition.get_input_shapes({"M": 128})
    print(f"  Input shapes with M=128: {shapes}")

    # Validate the reference code is present
    print(f"  Reference code length: {len(definition.reference)} chars")
    print(f"  Reference starts with: {definition.reference[:30]}...")

    return definition


def test_load_solution():
    """Load and test the GEMM Triton solution."""
    print_section("Loading Solution from dataset/solutions/gemm_triton_gemini.json")

    # Load directly from JSON file
    solution_path = Path("dataset/solutions/gemm_triton_gemini.json")
    solution = load_json(solution_path, Solution)

    print(f"✓ Loaded solution: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Description: {solution.description[:60] if solution.description else 'N/A'}...")
    print(f"  Language: {solution.spec['language']}")
    print(f"  Target hardware: {solution.spec['target_hardware']}")
    print(f"  Dependencies: {solution.spec['dependencies']}")
    print(f"  Entry point: {solution.spec['entry_point']}")
    print(f"  Is JIT compiled: {solution.is_jit_compiled()}")
    print(f"  Requires build: {solution.requires_build()}")

    # Check source files
    print(f"  Number of source files: {len(solution.sources)}")
    for source in solution.sources:
        print(f"    - {source['path']} ({len(source['content'])} chars)")

    # Get main source
    main_source = solution.get_main_source()
    if main_source:
        print(f"  Main source file: {main_source['path']}")
        print(f"  Main source preview: {main_source['content'][:50]}...")

    return solution


def test_load_traces():
    """Load and test trace files."""
    print_section("Loading Traces from dataset/traces/")

    # Load main trace file
    trace_path = Path("dataset/traces/gemm.jsonl")
    traces = load_jsonl(trace_path, Trace)

    print(f"✓ Loaded {len(traces)} trace(s) from gemm.jsonl")

    for i, trace in enumerate(traces):
        print(f"\nTrace {i+1}:")
        print(f"  Definition: {trace.definition}")
        print(f"  Solution: {trace.solution}")
        print(f"  Workload axes: {trace.workload['axes']}")
        print(f"  Status: {trace.evaluation['status']}")
        print(f"  Device: {trace.evaluation['environment']['device']}")
        print(f"  Latency: {trace.get_latency_ms():.3f} ms")
        print(
            f"  Reference latency: {trace.evaluation['performance']['reference_latency_ms']:.3f} ms"
        )
        print(f"  Speedup: {trace.get_speedup():.2f}x")
        print(f"  Max error: {trace.get_max_error():.2e}")
        print(f"  Is successful: {trace.is_successful()}")

    # Load workload-specific trace files
    workload_dir = Path("dataset/traces/workloads")
    workload_files = list(workload_dir.glob("*.jsonl"))

    print(f"\n✓ Found {len(workload_files)} workload trace files:")

    all_traces = []
    for wf in sorted(workload_files):
        wf_traces = load_jsonl(wf, Trace)
        all_traces.extend(wf_traces)
        print(f"  - {wf.name}: {len(wf_traces)} traces")

        # Show first trace from each file
        if wf_traces:
            t = wf_traces[0]
            print(f"    First trace: M={t.workload['axes']['M']}")

    return all_traces


def test_cross_validation(definition: Definition, solution: Solution, traces: list):
    """Validate that the loaded data is consistent."""
    print_section("Cross-Validation")

    # Check that solution references the correct definition
    if solution.definition == definition.name:
        print(f"✓ Solution '{solution.name}' correctly references definition '{definition.name}'")
    else:
        print(f"✗ Solution references '{solution.definition}' but we loaded '{definition.name}'")

    # Check that all traces reference correct definition and solution
    trace_defs = set(t.definition for t in traces)
    trace_sols = set(t.solution for t in traces)

    print(f"\n✓ Traces reference {len(trace_defs)} definition(s): {trace_defs}")
    print(f"✓ Traces reference {len(trace_sols)} solution(s): {trace_sols}")

    # Validate workload axes match definition
    print("\n✓ Validating workload axes:")
    var_axes = definition.get_var_axes()
    const_axes = definition.get_const_axes()

    for i, trace in enumerate(traces[:3]):  # Check first 3 traces
        workload_axes = trace.workload["axes"]
        print(f"  Trace {i+1} workload axes: {workload_axes}")

        # Check that all variable axes are provided
        for var_axis in var_axes:
            if var_axis in workload_axes:
                print(f"    - Variable axis '{var_axis}' = {workload_axes[var_axis]} ✓")
            else:
                print(f"    - Variable axis '{var_axis}' missing! ✗")

        # Constant axes should not be in workload
        for const_axis, value in const_axes.items():
            if const_axis in workload_axes:
                print(f"    - Constant axis '{const_axis}' should not be in workload! ✗")


def test_serialization():
    """Test JSON serialization round-trip."""
    print_section("Testing Serialization")

    # Load original
    definition_path = Path("dataset/definitions/gemm.json")
    original_def = load_json(definition_path, Definition)

    # Serialize to JSON and back
    json_str = original_def.to_json()
    restored_def = Definition.from_json(json_str)

    # Compare key fields
    print("✓ Definition serialization round-trip:")
    print(f"  Name matches: {original_def.name == restored_def.name}")
    print(f"  Type matches: {original_def.type == restored_def.type}")
    print(f"  Axes match: {original_def.axes == restored_def.axes}")
    print(f"  Inputs match: {original_def.inputs == restored_def.inputs}")
    print(f"  Outputs match: {original_def.outputs == restored_def.outputs}")

    # Do the same for solution
    solution_path = Path("dataset/solutions/gemm_triton_gemini.json")
    original_sol = load_json(solution_path, Solution)

    json_str = original_sol.to_json()
    restored_sol = Solution.from_json(json_str)

    print("\n✓ Solution serialization round-trip:")
    print(f"  Name matches: {original_sol.name == restored_sol.name}")
    print(f"  Spec matches: {original_sol.spec == restored_sol.spec}")
    print(f"  Sources count matches: {len(original_sol.sources) == len(restored_sol.sources)}")


def test_benchmark_config():
    """Test creating and using BenchmarkConfig."""
    print_section("Testing BenchmarkConfig")

    # Create a config
    config = BenchmarkConfig(
        warmup_runs=5, iterations=20, max_diff_limit=1e-4, device="cuda:0", log_level="INFO"
    )

    print(f"✓ Created BenchmarkConfig:")
    print(f"  Warmup runs: {config.warmup_runs}")
    print(f"  Iterations: {config.iterations}")
    print(f"  Max diff limit: {config.max_diff_limit}")
    print(f"  Device: {config.device}")
    print(f"  Log level: {config.log_level}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("FlashInfer Bench Dataset Loading Test".center(60))
    print("=" * 60)

    try:
        # Load all data
        definition = test_load_definition()
        solution = test_load_solution()
        traces = test_load_traces()

        # Cross-validate
        test_cross_validation(definition, solution, traces)

        # Test serialization
        test_serialization()

        # Test benchmark config
        test_benchmark_config()

        print_section("All Tests Completed Successfully! ✓")

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
