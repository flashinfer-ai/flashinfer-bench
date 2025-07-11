from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class SourceFile:
    path: str
    content: str

@dataclass
class SolutionSpec:
    language: str
    target_hardware: List[str]
    entry_point: str
    dependencies: Optional[List[str]] = field(default_factory=list)
    build_steps: Optional[List[str]] = field(default_factory=list)

@dataclass
class Solution:
    name: str
    definition: str  # name of the definition it implements
    author: str
    spec: SolutionSpec
    sources: List[SourceFile]
    description: Optional[str] = ""
