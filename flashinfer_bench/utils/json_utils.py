import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, List, Type, TypeVar, Union

T = TypeVar("T")


class FlashInferJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for FlashInfer Bench data classes."""

    def default(self, obj):
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def to_json(obj: Any, indent: int = 2) -> str:
    """Convert an object to JSON string."""
    return json.dumps(obj, cls=FlashInferJSONEncoder, indent=indent)


def from_json(json_str: str, cls: Type[T]) -> T:
    """Convert JSON string to object."""
    data = json.loads(json_str)

    return cls(**data)


def save_json(obj: Any, path: Union[str, Path]) -> None:
    """Save object to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj, f, cls=FlashInferJSONEncoder, indent=2)


def load_json(path: Union[str, Path], cls: Type[T]) -> T:
    """Load object from JSON file."""
    path = Path(path)

    with open(path, "r") as f:
        data = json.load(f)

    return cls(**data)


def load_jsonl(path: Union[str, Path], cls: Type[T]) -> List[T]:
    """Load objects from JSONL (JSON Lines) file.

    Each line should be a valid JSON object.
    """
    path = Path(path)
    results = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                results.append(cls(**data))

    return results


def save_jsonl(objects: List[Any], path: Union[str, Path]) -> None:
    """Save list of objects to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for obj in objects:
            f.write(json.dumps(obj, cls=FlashInferJSONEncoder) + "\n")
