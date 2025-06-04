from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

from ..definition import Definition
from ..solution import Solution


class BuildError(Exception):
    """Exception raised when build process fails"""

    pass


class BaseBuilder(ABC):
    """Abstract base class for all kernel builders"""

    def __init__(self):
        self.build_cache: Dict[str, Callable] = {}

    @abstractmethod
    def can_build(self, solution: Solution) -> bool:
        """Check if this builder can handle the given solution"""
        pass

    @abstractmethod
    def build_reference(self, definition: Definition) -> Callable:
        """Build a callable from definition reference code"""
        pass

    @abstractmethod
    def build_implementation(self, solution: Solution) -> Callable:
        """Build a callable from solution implementation"""
        pass

    @abstractmethod
    def validate_signature(self, definition: Definition, solution: Solution) -> bool:
        """Validate that solution signature matches definition"""
        pass

    def get_cached_callable(self, key: str) -> Callable:
        """Get cached callable by key"""
        return self.build_cache.get(key)

    def cache_callable(self, key: str, callable_obj: Callable) -> None:
        """Cache a built callable"""
        self.build_cache[key] = callable_obj

    def clear_cache(self) -> None:
        """Clear the build cache"""
        self.build_cache.clear()


class BuilderRegistry:
    """Registry for managing different builder types"""

    def __init__(self):
        self._builders: List[BaseBuilder] = []

    def register(self, builder: BaseBuilder) -> None:
        """Register a new builder"""
        self._builders.append(builder)

    def get_builder(self, solution: Solution) -> BaseBuilder:
        """Get appropriate builder for solution"""
        for builder in self._builders:
            if builder.can_build(solution):
                return builder
        raise BuildError(f"No suitable builder found for solution: {solution.name}")

    def get_reference_builder(self) -> BaseBuilder:
        """Get builder for reference implementations (typically Python)"""
        # For now, return the first builder that can handle Python
        # In practice, this would be more sophisticated
        for builder in self._builders:
            if hasattr(builder, "language") and builder.language == "python":
                return builder
        raise BuildError("No Python reference builder available")

    def list_builders(self) -> List[str]:
        """List all registered builder types"""
        return [builder.__class__.__name__ for builder in self._builders]
