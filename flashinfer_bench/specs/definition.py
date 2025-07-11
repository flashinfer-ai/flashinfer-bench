from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal

@dataclass
class Axis:
    type: Literal["const", "var"]
    value: Optional[int] = None
    parent: Optional[str] = None  # Only used for "var"

@dataclass
class TensorSpec:
    shape: List[str]
    dtype: str = "float16"

@dataclass
class Definition:
    name: str
    type: str
    description: Optional[str] = ""
    axes: Dict[str, Axis] = field(default_factory=dict)
    inputs: Dict[str, TensorSpec] = field(default_factory=dict)
    outputs: Dict[str, TensorSpec] = field(default_factory=dict)
    reference: str = ""  # PyTorch source code
    constraints: Optional[List[str]] = field(default_factory=list)
