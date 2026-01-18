"""Comparators for comparing reference and baseline outputs. Used by DefinitionTest."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class CompareResult:
    """Result of comparing reference and baseline outputs."""

    passed: bool
    """Whether the comparison passed."""
    stats: Dict[str, Any] = field(default_factory=dict)
    """Statistics from the comparison (e.g., max_abs_diff, mse)."""
    details: Optional[str] = None
    """Detailed error information if comparison failed."""


class Comparator(ABC):
    """Base class for output comparators."""

    @abstractmethod
    def compare(self, ref_output: Any, baseline_output: Any) -> CompareResult:
        """Compare reference output with baseline output.

        Parameters
        ----------
        ref_output : Any
            Output from reference implementation.
        baseline_output : Any
            Output from baseline implementation.

        Returns
        -------
        CompareResult
            Comparison result with pass/fail status and statistics.
        """
        ...


class TensorComparator(Comparator):
    """Comparator for single tensor outputs using torch.allclose."""

    def __init__(self, atol: float = 1e-2, rtol: float = 5e-2):
        """Initialize with tolerance values.

        Parameters
        ----------
        atol : float
            Absolute tolerance for comparison.
        rtol : float
            Relative tolerance for comparison.
        """
        self.atol = atol
        self.rtol = rtol

    def compare(self, ref_output: torch.Tensor, baseline_output: torch.Tensor) -> CompareResult:
        """Compare two tensors using torch.allclose.

        Parameters
        ----------
        ref_output : torch.Tensor
            Reference tensor output.
        baseline_output : torch.Tensor
            Baseline tensor output.

        Returns
        -------
        CompareResult
            Comparison result with pass/fail status and statistics.
        """
        ref = ref_output.float()
        base = baseline_output.float()

        abs_diff = (ref - base).abs()
        rel_diff = abs_diff / (base.abs() + 1e-8)

        stats = {
            "max_abs_diff": abs_diff.max().item(),
            "mean_abs_diff": abs_diff.mean().item(),
            "max_rel_diff": rel_diff.max().item(),
            "mean_rel_diff": rel_diff.mean().item(),
            "mse": torch.mean((ref - base) ** 2).item(),
        }

        passed = torch.allclose(ref, base, atol=self.atol, rtol=self.rtol)

        details = None
        if not passed:
            details = self._get_error_details(ref, base, abs_diff)

        return CompareResult(passed=passed, stats=stats, details=details)

    def _get_error_details(
        self, ref: torch.Tensor, base: torch.Tensor, abs_diff: torch.Tensor, top_k: int = 5
    ) -> str:
        """Get detailed error information for failed comparisons."""
        flat_diff = abs_diff.flatten()
        k = min(top_k, flat_diff.numel())
        top_errors, top_indices = torch.topk(flat_diff, k)

        lines = [f"Top {k} error locations:"]
        for i in range(k):
            idx = top_indices[i].item()
            ref_val = ref.flatten()[idx].item()
            base_val = base.flatten()[idx].item()
            lines.append(
                f"  [{idx}]: ref={ref_val:.6e}, base={base_val:.6e}, "
                f"diff={top_errors[i].item():.6e}"
            )

        return "\n".join(lines)


class MultiOutputComparator(Comparator):
    """Comparator for multiple outputs (tuple or dict)."""

    def __init__(
        self,
        output_names: List[str],
        comparators: Optional[Dict[str, Comparator]] = None,
        atol: float = 1e-2,
        rtol: float = 5e-2,
    ):
        """Initialize with output names and optional per-output comparators.

        Parameters
        ----------
        output_names : List[str]
            Names of outputs in order (for tuple unpacking).
        comparators : Dict[str, Comparator], optional
            Dict mapping output names to specific comparators.
        atol : float
            Default absolute tolerance.
        rtol : float
            Default relative tolerance.
        """
        self.output_names = output_names
        comparators = comparators or {}
        self.comparators = {
            name: comparators[name] if name in comparators else TensorComparator(atol, rtol)
            for name in output_names
        }

    def compare(
        self, ref_output: Union[tuple, dict], baseline_output: Union[tuple, dict]
    ) -> CompareResult:
        """Compare multiple outputs.

        Parameters
        ----------
        ref_output : Union[tuple, dict]
            Reference outputs as tuple or dict.
        baseline_output : Union[tuple, dict]
            Baseline outputs as tuple or dict.

        Returns
        -------
        CompareResult
            Comparison result with pass/fail status and per-output statistics.
        """
        # Convert tuples to dicts
        if isinstance(ref_output, tuple):
            ref_dict = dict(zip(self.output_names, ref_output, strict=True))
        else:
            ref_dict = ref_output

        if isinstance(baseline_output, tuple):
            base_dict = dict(zip(self.output_names, baseline_output, strict=True))
        else:
            base_dict = baseline_output

        all_passed = True
        all_stats: Dict[str, Any] = {}
        all_details: List[str] = []

        for name in self.output_names:
            if name not in ref_dict or name not in base_dict:
                continue

            comparator = self.comparators[name]
            result = comparator.compare(ref_dict[name], base_dict[name])

            all_stats[name] = result.stats
            if not result.passed:
                all_passed = False
                if result.details:
                    all_details.append(f"[{name}] {result.details}")

        details = "\n".join(all_details) if all_details else None
        return CompareResult(passed=all_passed, stats=all_stats, details=details)


class HitRatioComparator(Comparator):
    """Comparator that allows a percentage of values to exceed tolerance.

    Useful for low-precision kernels (e.g., FP8) where strict allclose may fail.
    """

    def __init__(self, atol: float = 1e-1, rtol: float = 2e-1, min_hit_ratio: float = 0.85):
        """Initialize with tolerance values and minimum hit ratio.

        Parameters
        ----------
        atol : float
            Absolute tolerance for per-element comparison.
        rtol : float
            Relative tolerance for per-element comparison.
        min_hit_ratio : float
            Minimum fraction of elements that must pass (0.0-1.0).
        """
        self.atol = atol
        self.rtol = rtol
        self.min_hit_ratio = min_hit_ratio

    def compare(self, ref_output: torch.Tensor, baseline_output: torch.Tensor) -> CompareResult:
        """Compare tensors with hit ratio criterion.

        Parameters
        ----------
        ref_output : torch.Tensor
            Reference tensor output.
        baseline_output : torch.Tensor
            Baseline tensor output.

        Returns
        -------
        CompareResult
            Comparison result with hit ratio statistics.
        """
        ref = ref_output.float()
        base = baseline_output.float()

        abs_diff = (ref - base).abs()
        threshold = self.atol + self.rtol * base.abs()
        ok = abs_diff <= threshold
        hit_ratio = ok.float().mean().item()

        stats = {
            "hit_ratio": hit_ratio,
            "required_ratio": self.min_hit_ratio,
            "max_abs_diff": abs_diff.max().item(),
            "mean_abs_diff": abs_diff.mean().item(),
        }

        passed = hit_ratio >= self.min_hit_ratio

        details = None
        if not passed:
            details = f"Hit ratio {hit_ratio:.2%} < required {self.min_hit_ratio:.2%}"

        return CompareResult(passed=passed, stats=stats, details=details)
