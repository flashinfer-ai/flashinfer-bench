from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Correctness:
    max_relative_error: float
    max_absolute_error: float

@dataclass
class Performance:
    latency_ms: float
    reference_latency_ms: float
    speedup_factor: float

@dataclass
class Environment:
    device: str
    libs: Dict[str, str]

@dataclass
class Evaluation:
    status: str
    log_file: str
    correctness: Correctness
    performance: Performance
    environment: Environment
    timestamp: str

@dataclass
class Trace:
    definition: str
    solution: str
    workload: Dict[str, Any]
    evaluation: Evaluation

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Trace":
        eval_dict = obj["evaluation"]
        evaluation = Evaluation(
            status=eval_dict["status"],
            log_file=eval_dict["log_file"],
            correctness=Correctness(**eval_dict["correctness"]),
            performance=Performance(**eval_dict["performance"]),
            environment=Environment(**eval_dict["environment"]),
            timestamp=eval_dict["timestamp"]
        )
        return cls(
            definition=obj["definition"],
            solution=obj["solution"],
            workload=obj["workload"],
            evaluation=evaluation
        )
