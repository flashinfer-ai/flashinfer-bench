"""
Utility functions for reference implementation testing.

Provides reusable tensor comparison, error reporting, and definition loading functions.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from flashinfer_bench.data import Definition, load_json_file

# Path to definitions directory (relative to flashinfer_trace/)
DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"


def load_definition(name: str) -> Definition:
    """
    Load a definition by name from definitions directory.

    Searches through all op_type subdirectories to find the matching definition file.

    Args:
        name: The definition name (e.g., "rmsnorm_h128", "gqa_paged_decode_h32_kv8_d128_ps1")

    Returns:
        Definition object loaded from the JSON file

    Raises:
        FileNotFoundError: If no definition with the given name is found
    """
    for op_dir in DEFINITIONS_DIR.iterdir():
        if op_dir.is_dir():
            def_file = op_dir / f"{name}.json"
            if def_file.exists():
                return load_json_file(Definition, def_file)
    raise FileNotFoundError(f"Definition {name} not found in {DEFINITIONS_DIR}")


def compile_reference(reference_code: str) -> Callable:
    """
    Compile reference implementation code to a callable function.

    The reference code is expected to define a `run()` function that takes
    the input tensors and returns the output tensors.

    Args:
        reference_code: Python source code containing the run() function definition

    Returns:
        The compiled run() function

    Example:
        >>> definition = load_definition("rmsnorm_h128")
        >>> run = compile_reference(definition.reference)
        >>> output = run(hidden_states, weight)
    """
    namespace = {"torch": torch, "math": math, "F": F}
    exec(reference_code, namespace)
    return namespace["run"]


def get_reference_run(definition_name: str) -> Callable:
    """
    Convenience function to load a definition and compile its reference implementation.

    Args:
        definition_name: The definition name (e.g., "rmsnorm_h128")

    Returns:
        The compiled run() function from the definition's reference code
    """
    definition = load_definition(definition_name)
    return compile_reference(definition.reference)


@dataclass
class TensorComparisonMetrics:
    """Metrics for comparing two tensors."""

    max_abs_diff: float
    max_rel_diff: float
    mean_abs_diff: float
    mean_rel_diff: float
    cosine_similarity: float
    mse: float
    all_close: bool


def compare_tensors(
    ref: torch.Tensor,
    actual: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 5e-2,
    eps: float = 1e-8,
) -> TensorComparisonMetrics:
    """
    Compare two tensors and compute various error metrics.

    Args:
        ref: Reference tensor
        actual: Actual tensor to compare against reference
        atol: Absolute tolerance for allclose check
        rtol: Relative tolerance for allclose check
        eps: Small epsilon value for numerical stability in relative difference

    Returns:
        TensorComparisonMetrics object containing all comparison metrics
    """
    # Convert to float32 for comparison
    ref_f32 = ref.float()
    actual_f32 = actual.float()

    # Compute absolute and relative differences
    abs_diff = torch.abs(ref_f32 - actual_f32)
    rel_diff = abs_diff / (torch.abs(actual_f32) + eps)

    # Compute error metrics
    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    mean_rel_diff = rel_diff.mean().item()

    # Compute cosine similarity
    ref_flat = ref_f32.flatten()
    actual_flat = actual_f32.flatten()
    cosine_sim = F.cosine_similarity(ref_flat.unsqueeze(0), actual_flat.unsqueeze(0), dim=1).item()

    # Compute MSE
    mse = ((ref_f32 - actual_f32) ** 2).mean().item()

    # Check if tensors are close
    all_close = torch.allclose(ref_f32, actual_f32, atol=atol, rtol=rtol)

    return TensorComparisonMetrics(
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        mean_abs_diff=mean_abs_diff,
        mean_rel_diff=mean_rel_diff,
        cosine_similarity=cosine_sim,
        mse=mse,
        all_close=all_close,
    )


def print_comparison_metrics(
    metrics: TensorComparisonMetrics, tensor_name: str = "Tensor", indent: str = ""
):
    """
    Print tensor comparison metrics in a formatted way.

    Args:
        metrics: TensorComparisonMetrics object
        tensor_name: Name of the tensor for display
        indent: String to prepend to each line for indentation
    """
    print(f"{indent}{tensor_name} comparison:")
    print(f"{indent}  Max absolute difference: {metrics.max_abs_diff:.6e}")
    print(f"{indent}  Max relative difference: {metrics.max_rel_diff:.6e}")
    print(f"{indent}  Mean absolute difference: {metrics.mean_abs_diff:.6e}")
    print(f"{indent}  Mean relative difference: {metrics.mean_rel_diff:.6e}")
    print(f"{indent}  Cosine similarity: {metrics.cosine_similarity:.6f}")
    print(f"{indent}  MSE: {metrics.mse:.6e}")


def find_and_print_top_errors(
    ref: torch.Tensor,
    actual: torch.Tensor,
    shape_names: Optional[tuple[str, ...]] = None,
    top_k: int = 5,
    tensor_name: str = "Tensor",
):
    """
    Find and print top error locations for debugging.

    Args:
        ref: Reference tensor
        actual: Actual tensor to compare against reference
        shape_names: Names for each dimension (e.g., ("batch", "heads", "dim"))
        top_k: Number of top errors to print
        tensor_name: Name of the tensor for display
    """
    ref_f32 = ref.float()
    actual_f32 = actual.float()

    abs_diff = torch.abs(ref_f32 - actual_f32)
    flat_abs_diff = abs_diff.flatten()

    k = min(top_k, flat_abs_diff.numel())
    if k == 0:
        return

    top_errors, top_indices = torch.topk(flat_abs_diff, k)

    print(f"\nTop {k} {tensor_name} error locations:")
    for i in range(k):
        idx = top_indices[i].item()

        # Convert flat index to multi-dimensional indices
        indices = []
        remaining = idx
        for dim_size in reversed(ref.shape):
            indices.append(remaining % dim_size)
            remaining //= dim_size
        indices = list(reversed(indices))

        # Format indices with names if provided
        if shape_names and len(shape_names) == len(indices):
            index_str = ", ".join(f"{name}={val}" for name, val in zip(shape_names, indices))
        else:
            index_str = ", ".join(str(i) for i in indices)

        ref_val = ref_f32.flatten()[idx].item()
        actual_val = actual_f32.flatten()[idx].item()

        print(
            f"  [{index_str}]: "
            f"ref={ref_val:.6f}, actual={actual_val:.6f}, diff={top_errors[i].item():.6e}"
        )


def compare_and_report(
    ref: torch.Tensor,
    actual: torch.Tensor,
    tensor_name: str = "Output",
    shape_names: Optional[tuple[str, ...]] = None,
    atol: float = 1e-2,
    rtol: float = 5e-2,
    show_top_errors: bool = True,
    top_k: int = 5,
) -> bool:
    """
    Compare two tensors, print metrics, and optionally show top errors.

    This is a convenience function that combines compare_tensors, print_comparison_metrics,
    and find_and_print_top_errors.

    Args:
        ref: Reference tensor
        actual: Actual tensor to compare against reference
        tensor_name: Name of the tensor for display
        shape_names: Names for each dimension (e.g., ("batch", "heads", "dim"))
        atol: Absolute tolerance for allclose check
        rtol: Relative tolerance for allclose check
        show_top_errors: Whether to print top error locations if tensors don't match
        top_k: Number of top errors to print

    Returns:
        True if tensors match within tolerance, False otherwise
    """
    metrics = compare_tensors(ref, actual, atol=atol, rtol=rtol)
    print(f"\n{tensor_name} comparison:")
    print_comparison_metrics(metrics, tensor_name="", indent="  ")

    if not metrics.all_close and show_top_errors:
        find_and_print_top_errors(ref, actual, shape_names, top_k, tensor_name)

    return metrics.all_close
