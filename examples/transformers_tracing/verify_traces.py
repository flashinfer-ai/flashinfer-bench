#!/usr/bin/env python3
"""
Verify that tracing properly captured all expected operators for a model.

This script analyzes the traces collected by trace_models.py and verifies
that all expected operators were traced for the target models.

Usage:
    python verify_traces.py --traces ./traces --model llama-3.1-70b
    python verify_traces.py --traces ./traces --model qwen3-30b-moe
    python verify_traces.py --traces ./traces  # Check all traces
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


# Base operator configurations - all operators are required
# (Linear/GEMM is excluded as it's not currently traced)
def _make_operator_expectations(has_moe: bool = False, activation: str = "silu"):
    """Create operator expectations for a model.
    
    Parameters
    ----------
    has_moe : bool
        Whether the model uses MoE (Mixture of Experts)
    activation : str
        Activation type: "silu", "gelu", or "none" (for custom activations)
    """
    ops = {
        "attention": {
            "pattern": r"gqa_ragged_prefill_.*_h\d+_kv\d+_d\d+",
            "description": "GQA attention operations",
        },
        "rmsnorm": {
            "pattern": r"rmsnorm_h\d+|fused_add_rmsnorm_h\d+",
            "description": "RMS normalization",
        },
        "rope": {
            "pattern": r"rope_h\d+_d\d+",
            "description": "Rotary position embedding",
        },
        "embedding": {
            "pattern": r"embedding_v\d+_d\d+",
            "description": "Token embedding lookup",
        },
        "softmax": {
            "pattern": r"softmax_d\d+",
            "description": "Softmax for sampling/attention",
        },
        "topk": {
            "pattern": r"topk_d\d+_k\d+",
            "description": "Top-k sampling/routing",
        },
        "multinomial": {
            "pattern": r"sampling_multinomial_v\d+",
            "description": "Multinomial sampling",
        },
    }
    
    # Add activation based on model type
    # "none" means model uses custom activation not traced by standard adapters
    if activation == "silu":
        ops["silu"] = {
            "pattern": r"silu_h\d+",
            "description": "SiLU activation (SwiGLU)",
        }
    elif activation == "gelu":
        ops["gelu"] = {
            "pattern": r"gelu(_tanh)?_h\d+",
            "description": "GELU activation",
        }
    # "none" - don't add any activation requirement
    
    # Add MoE for MoE models
    if has_moe:
        ops["moe"] = {
            "pattern": r"moe_(batched|grouped|fp8).*",
            "description": "Mixture of Experts",
        }
    
    return ops


# Expected operators for each model type
MODEL_OPERATOR_EXPECTATIONS = {
    "qwen3-30b-moe": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "expected_operators": _make_operator_expectations(has_moe=True, activation="silu"),
    },
    "llama-3.1-70b": {
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "expected_operators": _make_operator_expectations(has_moe=False, activation="silu"),
    },
    "gpt-oss-120b": {
        "model_id": "openai/gpt-oss-120b",
        # gpt-oss-120b uses custom GLU activation (not SiLU/GELU), so we don't check for activation
        # It has MoE with 128 experts, top-4 (uses placeholder tensors for MXFP4 quantized experts)
        "expected_operators": _make_operator_expectations(has_moe=True, activation="none"),
    },
    "llama-3.1-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "expected_operators": _make_operator_expectations(has_moe=False, activation="silu"),
    },
}

# Add FP8 variants (same expectations)
MODEL_OPERATOR_EXPECTATIONS["qwen3-30b-moe-fp8"] = {
    **MODEL_OPERATOR_EXPECTATIONS["qwen3-30b-moe"],
    "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
}
MODEL_OPERATOR_EXPECTATIONS["llama-3.1-70b-fp8"] = {
    **MODEL_OPERATOR_EXPECTATIONS["llama-3.1-70b"],
    "model_id": "RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8",
}


def load_traces(traces_path: Path) -> dict[str, list[dict]]:
    """Load all traces from a directory."""
    traces = defaultdict(list)
    
    workloads_dir = traces_path / "workloads"
    if not workloads_dir.exists():
        print(f"Warning: No workloads directory found at {workloads_dir}")
        return traces
    
    # Traces are stored in subdirectories by op_type: workloads/<op_type>/<def_name>.jsonl
    for trace_file in workloads_dir.glob("**/*.jsonl"):
        def_name = trace_file.stem
        with open(trace_file, "r") as f:
            for line in f:
                if line.strip():
                    trace = json.loads(line)
                    traces[def_name].append(trace)
    
    return traces


def analyze_traces(traces: dict[str, list[dict]]) -> dict[str, dict[str, Any]]:
    """Analyze traces and categorize them by operator type."""
    analysis = defaultdict(lambda: {"count": 0, "definitions": set(), "examples": []})
    
    # Patterns to categorize definitions
    patterns = {
        "attention": r"gqa_ragged_prefill_.*_h\d+_kv\d+_d\d+|gqa_paged_.*",
        "rmsnorm": r"rmsnorm_h\d+|fused_add_rmsnorm_h\d+",
        "rope": r"rope_h\d+_d\d+",
        "embedding": r"embedding_v\d+_d\d+",
        "silu": r"silu_h\d+",
        "gelu": r"gelu(_tanh)?_h\d+",
        "moe": r"moe_(batched|grouped)_e\d+_h\d+_i\d+_topk\d+",
        "linear": r"gemm_n\d+_k\d+",
        "softmax": r"softmax_d\d+",
        "topk": r"topk_d\d+_k\d+",
        "multinomial": r"sampling_multinomial_v\d+",
    }
    
    for def_name, trace_list in traces.items():
        categorized = False
        for op_type, pattern in patterns.items():
            if re.match(pattern, def_name):
                analysis[op_type]["count"] += len(trace_list)
                analysis[op_type]["definitions"].add(def_name)
                if len(analysis[op_type]["examples"]) < 3:
                    analysis[op_type]["examples"].append(def_name)
                categorized = True
                break
        
        if not categorized:
            analysis["unknown"]["count"] += len(trace_list)
            analysis["unknown"]["definitions"].add(def_name)
    
    return dict(analysis)


def verify_model_traces(
    traces_path: Path,
    model_key: str,
) -> tuple[bool, dict[str, Any]]:
    """
    Verify traces for a specific model configuration.
    
    Returns (success, report_dict)
    """
    if model_key not in MODEL_OPERATOR_EXPECTATIONS:
        return False, {"error": f"Unknown model key: {model_key}"}
    
    expectations = MODEL_OPERATOR_EXPECTATIONS[model_key]
    traces = load_traces(traces_path)
    analysis = analyze_traces(traces)
    
    report = {
        "model_key": model_key,
        "model_id": expectations["model_id"],
        "traces_path": str(traces_path),
        "total_definitions": len(traces),
        "total_traces": sum(len(t) for t in traces.values()),
        "operator_coverage": {},
        "missing_required": [],
        "missing_optional": [],
        "success": True,
    }
    
    for op_name, op_config in expectations["expected_operators"].items():
        pattern = op_config["pattern"]
        description = op_config["description"]
        
        # Check if any definition matches this pattern
        matching_defs = []
        for def_name in traces.keys():
            if re.match(pattern, def_name):
                matching_defs.append(def_name)
        
        found = len(matching_defs) > 0
        trace_count = sum(len(traces[d]) for d in matching_defs)
        
        report["operator_coverage"][op_name] = {
            "found": found,
            "description": description,
            "matching_definitions": matching_defs[:5],  # Limit to 5 examples
            "trace_count": trace_count,
        }
        
        if not found:
            report["missing_required"].append(op_name)
            report["success"] = False
    
    return report["success"], report


def print_report(report: dict[str, Any], verbose: bool = False):
    """Print a verification report."""
    print("\n" + "=" * 70)
    print(f"TRACE VERIFICATION REPORT")
    print(f"Model: {report.get('model_id', 'Unknown')}")
    print(f"Traces path: {report.get('traces_path', 'Unknown')}")
    print("=" * 70)
    
    print(f"\nTotal definitions traced: {report.get('total_definitions', 0)}")
    print(f"Total trace entries: {report.get('total_traces', 0)}")
    
    print("\n" + "-" * 70)
    print("OPERATOR COVERAGE:")
    print("-" * 70)
    
    coverage = report.get("operator_coverage", {})
    for op_name, op_info in coverage.items():
        status = "✓" if op_info["found"] else "✗"
        count_str = f"[{op_info['trace_count']} traces]" if op_info["found"] else ""
        
        print(f"  {status} {op_name:15} {count_str}")
        
        if verbose and op_info["found"] and op_info["matching_definitions"]:
            for def_name in op_info["matching_definitions"][:3]:
                print(f"      - {def_name}")
    
    print("\n" + "-" * 70)
    
    if report.get("missing_required"):
        print("MISSING OPERATORS:")
        for op in report["missing_required"]:
            print(f"  ✗ {op}")
        print()
    
    success = report.get("success", False)
    if success:
        print("STATUS: ✓ ALL OPERATORS COVERED")
    else:
        print("STATUS: ✗ MISSING OPERATORS")
    
    print("=" * 70 + "\n")
    
    return success


def list_all_traces(traces_path: Path):
    """List all traced definitions and their counts."""
    traces = load_traces(traces_path)
    analysis = analyze_traces(traces)
    
    print("\n" + "=" * 70)
    print("ALL TRACED OPERATORS")
    print(f"Traces path: {traces_path}")
    print("=" * 70)
    
    for op_type, info in sorted(analysis.items()):
        print(f"\n{op_type.upper()} ({info['count']} traces):")
        for def_name in sorted(info["definitions"]):
            count = len(traces[def_name])
            print(f"  - {def_name} [{count} traces]")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Verify traced operators for transformers models"
    )
    parser.add_argument(
        "--traces",
        type=str,
        required=True,
        help="Path to traces directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_OPERATOR_EXPECTATIONS.keys()) + ["all"],
        default=None,
        help="Model configuration to verify (or 'all' to list all traces)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all traced definitions without verification",
    )
    
    args = parser.parse_args()
    traces_path = Path(args.traces)
    
    if not traces_path.exists():
        print(f"Error: Traces path does not exist: {traces_path}")
        return 1
    
    if args.list or args.model == "all":
        list_all_traces(traces_path)
        return 0
    
    if args.model is None:
        print("Available model configurations:")
        for key, config in MODEL_OPERATOR_EXPECTATIONS.items():
            print(f"  {key}: {config['model_id']}")
        print("\nUse --model <key> to verify a specific model")
        print("Use --list to see all traced operators")
        return 0
    
    success, report = verify_model_traces(traces_path, args.model)
    print_report(report, verbose=args.verbose)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
