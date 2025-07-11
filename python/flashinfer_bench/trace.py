"""Trace data class for FlashInfer Bench."""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime

from flashinfer_bench.utils.json_utils import to_json, from_json
from flashinfer_bench.utils.validation import validate_workload_axes


@dataclass
class Trace:
    """An atomic and immutable record of a single benchmark run.
    
    A Trace links a specific Solution to a specific Definition, details the exact
    workload configuration used for the run, and records the complete evaluation result.
    
    Special case: A "workload trace" only contains definition and workload fields,
    with solution and evaluation being empty. This represents a workload configuration
    without an actual benchmark run.
    """
    
    definition: str
    solution: str
    workload: Dict[str, Any]
    evaluation: Dict[str, Any]
    
    def __post_init__(self):
        """Validate the trace after initialization."""
        if not self.definition:
            raise ValueError("Trace must reference a definition")
        
        if not self.workload:
            raise ValueError("Trace must have a workload")
        
        # Check if this is a workload trace (only definition and workload provided)
        is_workload_trace = (
            (not self.solution or self.solution == "") and 
            (not self.evaluation or self.evaluation == {})
        )
        
        if is_workload_trace:
            # For workload traces, only validate the workload
            self._validate_workload()
        else:
            # For regular traces, validate all fields
            if not self.solution:
                raise ValueError("Regular trace must reference a solution")
            
            if not self.evaluation:
                raise ValueError("Regular trace must have an evaluation")
            
            self._validate_workload()
            
            self._validate_evaluation()
    
    def _validate_workload(self):
        """Validate the workload dictionary."""
        if "axes" not in self.workload:
            raise ValueError("Workload must have 'axes' field")
        
        if "inputs" not in self.workload:
            raise ValueError("Workload must have 'inputs' field")
        
        # Validate axes
        if not isinstance(self.workload["axes"], dict):
            raise ValueError("workload.axes must be a dictionary")
        
        for axis_name, value in self.workload["axes"].items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"Workload axis '{axis_name}' must be a positive integer")
        
        # Validate inputs
        if not isinstance(self.workload["inputs"], dict):
            raise ValueError("workload.inputs must be a dictionary")
        
        for input_name, input_desc in self.workload["inputs"].items():
            self._validate_input_descriptor(input_name, input_desc)
    
    def _validate_input_descriptor(self, name: str, descriptor: Dict[str, Any]):
        """Validate an input descriptor."""
        if "type" not in descriptor:
            raise ValueError(f"Input '{name}' must have a 'type' field")
        
        input_type = descriptor["type"]
        
        if input_type == "random":
            # Random inputs don't require additional fields
            pass
        
        elif input_type == "safetensors":
            if "path" not in descriptor:
                raise ValueError(f"Safetensors input '{name}' must have a 'path' field")
            
            if "tensor_key" not in descriptor:
                raise ValueError(f"Safetensors input '{name}' must have a 'tensor_key' field")
            
            if not isinstance(descriptor["path"], str) or not descriptor["path"]:
                raise ValueError(f"Input '{name}' path must be a non-empty string")
            
            if not isinstance(descriptor["tensor_key"], str) or not descriptor["tensor_key"]:
                raise ValueError(f"Input '{name}' tensor_key must be a non-empty string")
        
        else:
            raise ValueError(f"Invalid input type '{input_type}' for input '{name}'")
    
    def _validate_evaluation(self):
        """Validate the evaluation dictionary."""
        required_fields = ["status", "log_file", "correctness", "performance", "environment", "timestamp"]
        
        for field in required_fields:
            if field not in self.evaluation:
                raise ValueError(f"Evaluation missing required field '{field}'")
        
        valid_statuses = ["PASSED", "INCORRECT", "RUNTIME_ERROR", "COMPILE_ERROR"]
        if self.evaluation["status"] not in valid_statuses:
            raise ValueError(f"Invalid evaluation status '{self.evaluation['status']}'. Must be one of: {valid_statuses}")
        
        if not isinstance(self.evaluation["log_file"], str) or not self.evaluation["log_file"]:
            raise ValueError("evaluation.log_file must be a non-empty string")
        
        self._validate_correctness(self.evaluation["correctness"])
        
        self._validate_performance(self.evaluation["performance"])
        
        self._validate_environment(self.evaluation["environment"])
        
        try:
            datetime.fromisoformat(self.evaluation["timestamp"].replace('Z', '+00:00'))
        except Exception:
            raise ValueError("evaluation.timestamp must be a valid ISO 8601 timestamp")
    
    def _validate_correctness(self, correctness: Dict[str, Any]):
        """Validate correctness metrics."""
        if not isinstance(correctness, dict):
            raise ValueError("evaluation.correctness must be a dictionary")
        
        required_fields = ["max_relative_error", "max_absolute_error"]
        
        for field in required_fields:
            if field not in correctness:
                raise ValueError(f"Correctness missing required field '{field}'")
            
            if not isinstance(correctness[field], (int, float)) or correctness[field] < 0:
                raise ValueError(f"correctness.{field} must be a non-negative number")
    
    def _validate_performance(self, performance: Dict[str, Any]):
        """Validate performance metrics."""
        if not isinstance(performance, dict):
            raise ValueError("evaluation.performance must be a dictionary")
        
        required_fields = ["latency_ms", "reference_latency_ms", "speedup_factor"]
        
        for field in required_fields:
            if field not in performance:
                raise ValueError(f"Performance missing required field '{field}'")
            
            if not isinstance(performance[field], (int, float)) or performance[field] < 0:
                raise ValueError(f"performance.{field} must be a non-negative number")
    
    def _validate_environment(self, environment: Dict[str, Any]):
        """Validate environment information."""
        if not isinstance(environment, dict):
            raise ValueError("evaluation.environment must be a dictionary")
        
        if "device" not in environment:
            raise ValueError("Environment missing required field 'device'")
        
        if not isinstance(environment["device"], str) or not environment["device"]:
            raise ValueError("environment.device must be a non-empty string")
        
        if "libs" not in environment:
            raise ValueError("Environment missing required field 'libs'")
        
        if not isinstance(environment["libs"], dict):
            raise ValueError("environment.libs must be a dictionary")
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return to_json(self)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Trace":
        """Create from JSON string."""
        return from_json(json_str, cls)
    
    def is_workload(self) -> bool:
        """Check if this is a workload-only trace."""
        return (not self.solution or self.solution == "") and (not self.evaluation or self.evaluation == {})
    
    def is_successful(self) -> bool:
        """Check if the benchmark run was successful."""
        return self.evaluation["status"] == "PASSED"
    
    def get_speedup(self) -> Optional[float]:
        """Get the speedup factor."""
        return self.evaluation["performance"]["speedup_factor"]
    
    def get_latency_ms(self) -> Optional[float]:
        """Get the implementation latency in milliseconds."""
        return self.evaluation["performance"]["latency_ms"]
    
    def get_max_error(self) -> Optional[float]:
        """Get the maximum error (relative or absolute)."""
        return max(
            self.evaluation["correctness"]["max_relative_error"],
            self.evaluation["correctness"]["max_absolute_error"]
        )