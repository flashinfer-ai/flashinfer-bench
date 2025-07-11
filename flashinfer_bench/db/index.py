from typing import Dict, List, TypeVar, Callable

T = TypeVar("T")

def build_index(items: List[T], key_fn: Callable[[T], str]) -> Dict[str, T]:
    return {key_fn(item): item for item in items}
