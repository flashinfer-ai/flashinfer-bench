from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    All fields have default values to make configuration optional.
    """

    warmup_runs: int = field(default=10)
    iterations: int = field(default=50)
    correctness_trials: int = field(default=3)
    performance_trials: int = field(default=5)
    rtol: float = field(default=1e-2)
    atol: float = field(default=1e-2)
    device: str = field(default="cuda:0")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = field(default="INFO")

    def __post_init__(self):
        """Validate configuration values."""
        if self.warmup_runs < 0:
            raise ValueError(f"warmup_runs must be non-negative, got {self.warmup_runs}")

        if self.iterations <= 0:
            raise ValueError(f"iterations must be positive, got {self.iterations}")

        if self.correctness_trials <= 0:
            raise ValueError(f"correctness_trials must be positive, got {self.correctness_trials}")

        if self.performance_trials <= 0:
            raise ValueError(f"performance_trials must be positive, got {self.performance_trials}")

        if self.rtol <= 0:
            raise ValueError(f"rtol must be positive, got {self.rtol}")

        if self.atol <= 0:
            raise ValueError(f"atol must be positive, got {self.atol}")

        if not self.device:
            raise ValueError("device cannot be empty")

        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")
