"""Configuration for benchmark execution."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class EvalConfig(BaseModel):
    """Per-definition eval parameters. All fields Optional; None means inherit from parent layer."""

    warmup_runs: Optional[int] = None
    iterations: Optional[int] = None
    num_trials: Optional[int] = None
    rtol: Optional[float] = None
    atol: Optional[float] = None
    required_matched_ratio: Optional[float] = None
    extra: Dict[str, Any] = {}


class ResolvedEvalConfig(BaseModel):
    """Resolved eval parameters with all fields populated. This is what evaluators consume."""

    warmup_runs: int = 10
    iterations: int = 50
    num_trials: int = 3
    rtol: float = 1e-2
    atol: float = 1e-2
    required_matched_ratio: Optional[float] = None
    profile_baseline: bool = True
    extra: Dict[str, Any] = {}


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark runs. Mirrors the YAML structure exactly.

    All fields have default values to make configuration optional.
    """

    # System-level
    use_isolated_runner: bool = False
    definitions: Optional[List[str]] = None
    solutions: Optional[List[str]] = None
    timeout_seconds: int = 300
    profile_baseline: bool = True
    log_dir: Optional[str] = None

    # Per-definition defaults
    warmup_runs: int = 10
    iterations: int = 50
    num_trials: int = 3
    rtol: float = 1e-2
    atol: float = 1e-2
    required_matched_ratio: Optional[float] = None
    # Deprecated: use op_type_config/definition_config extra instead
    sampling_validation_trials: int = 100
    sampling_tvd_threshold: float = 0.2

    # Per op_type / per definition overrides
    op_type_config: Dict[str, EvalConfig] = {}
    definition_config: Dict[str, EvalConfig] = {}

    @model_validator(mode="after")
    def _validate_fields(self) -> BenchmarkConfig:
        if self.log_dir is not None:
            warnings.warn(
                "log_dir is deprecated and ignored; logs are embedded in trace evaluations",
                DeprecationWarning,
                stacklevel=2,
            )
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
        return self

    @classmethod
    def from_yaml(cls, path: str, **overrides: Any) -> BenchmarkConfig:
        """Load config from a YAML file, with optional field overrides."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        data.update(overrides)
        return cls.model_validate(data)

    @classmethod
    def default(cls, **overrides: Any) -> BenchmarkConfig:
        """Load the bundled eval_config.yaml if it exists, otherwise use defaults."""
        yaml_path = Path(__file__).parent / "eval_config.yaml"
        if yaml_path.exists():
            return cls.from_yaml(str(yaml_path), **overrides)
        return cls(**overrides)

    def resolve_eval_config(self, definition: Any) -> ResolvedEvalConfig:
        """Merge: per-def defaults -> op_type_config -> definition_config."""
        resolved = ResolvedEvalConfig(
            warmup_runs=self.warmup_runs,
            iterations=self.iterations,
            num_trials=self.num_trials,
            rtol=self.rtol,
            atol=self.atol,
            required_matched_ratio=self.required_matched_ratio,
            profile_baseline=self.profile_baseline,
            extra={
                "sampling_validation_trials": self.sampling_validation_trials,
                "sampling_tvd_threshold": self.sampling_tvd_threshold,
            },
        )

        layers = [
            self.op_type_config.get(definition.op_type),
            self.definition_config.get(definition.name),
        ]

        for layer in layers:
            if layer is None:
                continue
            for field_name, value in layer.model_dump(exclude={"extra"}).items():
                if value is not None:
                    setattr(resolved, field_name, value)
            resolved.extra.update(layer.extra)

        return resolved
