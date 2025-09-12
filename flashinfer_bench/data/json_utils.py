"""Unified JSON encoding/decoding for all dataclasses."""

from pathlib import Path
from typing import List, Type, Union

from pydantic import BaseModel


def save_json_file(obj: BaseModel, path: Union[str, Path]) -> None:
    """Save a dataclass to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(obj.model_dump_json(indent=2))


def load_json_file(path: Union[str, Path], type: Type[BaseModel]) -> BaseModel:
    """Load a dataclass from a JSON file."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return type.model_validate_json(f.read())


def save_jsonl_file(objects: List[BaseModel], path: Union[str, Path]) -> None:
    """Save a list of dataclasses to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in objects:
            f.write(obj.model_dump_json(indent=None))
            f.write("\n")


def load_jsonl_file(path: Union[str, Path], type: Type[BaseModel]) -> List[BaseModel]:
    """Load a list of dataclasses from a JSONL file."""
    out = []
    with open(Path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(type.model_validate_json(line))
    return out


def append_jsonl_lines(path: Union[str, Path], objs: List[BaseModel]) -> None:
    """Append a list of dataclasses to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for obj in objs:
            f.write(obj.model_dump_json(indent=None))
            f.write("\n")
