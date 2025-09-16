from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


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
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = field(default="INFO")
    auto_flush: bool = field(default=False) # if enabled, benchmark will store results per definition

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
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")
