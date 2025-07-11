from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class BenchmarkConfig:
    warmup_runs: int = 10
    iterations: int = 50
    max_diff_limit: float = 1e-5
    device: str = "cuda:0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"