from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ApplyConfig:
    # The maximum absolute difference allowed between the reference and the candidate
    max_atol: float = 1e-2
    # The maximum relative difference allowed between the reference and the candidate
    max_rtol: float = 1e-5
    # The ratio of the top solutions to AOT build for each definition
    aot_ratio: float = 1.0
    # The policy when a runtime ApplyKey misses the table
    on_miss_policy: Literal["fallback_only", "use_def_best"] = "fallback_only"

    def __post_init__(self) -> None:
        if not isinstance(self.max_atol, float):
            raise ValueError("max_atol must be a float")
        if not isinstance(self.max_rtol, float):
            raise ValueError("max_rtol must be a float")
        if not isinstance(self.aot_ratio, float):
            raise ValueError("aot_ratio must be a float")

        if self.aot_ratio < 0 or self.aot_ratio > 1:
            raise ValueError("aot_ratio must be between 0 and 1")
        if self.on_miss_policy not in ["fallback_only", "use_def_best"]:
            raise ValueError("on_miss_policy must be either 'fallback_only' or 'use_def_best'")
        if self.max_atol <= 0:
            raise ValueError("max_atol must be positive")
        if self.max_rtol <= 0:
            raise ValueError("max_rtol must be positive")
