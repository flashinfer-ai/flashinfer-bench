from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    All fields have default values to make configuration optional.
    """

    warmup_runs: int = field(default=10)
    iterations: int = field(default=50)
    num_trials: int = field(default=3)
    rtol: float = field(default=1e-2)
    atol: float = field(default=1e-2)
    log_dir: str = field(default="/tmp/flashinfer_bench")
    use_isolated_runner: bool = field(default=False)
    required_matched_ratio: Optional[float] = field(default=None)
    sampling_validation_trials: int = field(default=100)
    sampling_tvd_threshold: float = field(default=0.2)
    definitions: Optional[List[str]] = field(default=None)
    solutions: Optional[List[str]] = field(default=None)
    timeout_seconds: int = field(default=300)
    profile_baseline: bool = field(default=True)

    def __post_init__(self):
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be >= 0")
        if self.iterations <= 0:
            raise ValueError("iterations must be > 0")
        if self.num_trials <= 0:
            raise ValueError("num_trials must be > 0")
        if self.rtol <= 0 or self.atol <= 0:
            raise ValueError("rtol/atol must be > 0")
        if not isinstance(self.rtol, float):
            raise ValueError("rtol must be a float")
        if not isinstance(self.atol, float):
            raise ValueError("atol must be a float")
        if self.required_matched_ratio is not None and not (
            0.0 < self.required_matched_ratio <= 1.0
        ):
            raise ValueError("required_matched_ratio must be between 0 and 1")
        if self.required_matched_ratio is not None and not isinstance(
            self.required_matched_ratio, float
        ):
            raise ValueError("required_matched_ratio must be a float")
        if self.sampling_validation_trials <= 0:
            raise ValueError("sampling_validation_trials must be > 0")
        if not isinstance(self.sampling_validation_trials, int):
            raise ValueError("sampling_validation_trials must be an int")
        if not (0.0 <= self.sampling_tvd_threshold <= 1.0):
            raise ValueError("sampling_tvd_threshold must be between 0 and 1")
        if not isinstance(self.sampling_tvd_threshold, float):
            raise ValueError("sampling_tvd_threshold must be a float")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if not isinstance(self.timeout_seconds, int):
            raise ValueError("timeout_seconds must be an int")
        if self.definitions is not None and not isinstance(self.definitions, list):
            raise ValueError("definitions must be a list or None")
        if self.solutions is not None and not isinstance(self.solutions, list):
            raise ValueError("solutions must be a list or None")
