"""Pytest-compatible base class for definition tests."""

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional

import pytest

from flashinfer_bench.testing.pytest_config import requires_torch_cuda

from .comparators import Comparator, CompareResult
from .core import DefinitionTest


class DefinitionTestCase:
    """Pytest-compatible test case base class for Definition-based testing.

    Inherit from this class to create parametrized tests that compare
    a baseline implementation against the reference from a Definition JSON.

    The baseline function must have the same interface as the reference's run().

    Example:
        class TestGQAPagedDecode(DefinitionTestCase):
            definition_path = "flashinfer_trace/definitions/gqa_paged/gqa_paged_decode.json"
            configs = [
                {"batch_size": 1, "num_pages": 100},
                {"batch_size": 4, "num_pages": 200},
            ]

            def baseline_fn(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
                # Wrap FlashInfer to match reference interface
                wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(...)
                wrapper.plan(...)
                return wrapper.run(...)

    Run with pytest:
        pytest tests/definition/test_gqa.py -v
    """

    # Required: subclass must define these
    definition_path: str = ""
    configs: List[Dict[str, Any]] = []

    # Optional: subclass can override these
    input_generator: Optional[Callable[..., Dict[str, Any]]] = None
    comparator: Optional[Comparator] = None
    atol: float = 1e-2
    rtol: float = 5e-2
    device: str = "cuda"

    @abstractmethod
    def baseline_fn(self, **inputs: Any) -> Any:
        """Baseline implementation with same interface as reference run().

        This method must accept the same arguments as the reference
        implementation's run() function defined in the Definition.

        Args:
            **inputs: Input tensors/scalars as defined in Definition

        Returns:
            Output tensor(s) matching the Definition's outputs spec
        """
        raise NotImplementedError("Subclass must implement baseline_fn")

    @pytest.fixture
    def tester(self) -> DefinitionTest:
        """Create a DefinitionTest instance."""
        return DefinitionTest(
            definition=self.definition_path,
            baseline_fn=self.baseline_fn,
            input_generator=self.input_generator,
            comparator=self.comparator,
            device=self.device,
        )

    @pytest.fixture(params=[])
    def config(self, request: pytest.FixtureRequest) -> Dict[str, Any]:
        """Fixture that provides test configurations.

        Note: This is overridden by pytest_generate_tests to use self.configs.
        """
        return request.param

    def pytest_generate_tests(self, metafunc: pytest.Metafunc) -> None:
        """Generate parametrized test cases from configs.

        This is called by pytest to generate test parameters.
        """
        if "config" in metafunc.fixturenames and self.configs:
            # Create readable IDs for each config
            ids = [self._config_to_id(c) for c in self.configs]
            metafunc.parametrize("config", self.configs, ids=ids)

    @staticmethod
    def _config_to_id(config: Dict[str, Any]) -> str:
        """Convert config dict to a readable test ID."""
        parts = [f"{k}={v}" for k, v in config.items()]
        return "-".join(parts)

    def test_definition_schema(self, tester: DefinitionTest) -> None:
        """Test that the definition file is valid JSON and conforms to schema.

        Args:
            tester: DefinitionTest instance (from fixture)
        """
        tester.validate_schema()

    def test_reference_executable(self, tester: DefinitionTest) -> None:
        """Test that the reference code can be executed and has a callable 'run' function.

        Args:
            tester: DefinitionTest instance (from fixture)
        """
        import math

        import torch

        code = tester.definition.get("reference", "")
        namespace: Dict[str, Any] = {"torch": torch, "math": math}

        # Test that code can be executed
        try:
            exec(code, namespace)  # noqa: S102
        except Exception as e:
            raise AssertionError(f"Reference code failed to execute: {e}") from e

        # Test that 'run' exists
        assert "run" in namespace, "Reference must define a 'run' function"

        # Test that 'run' is callable
        assert callable(namespace["run"]), "Reference 'run' must be callable"

    @requires_torch_cuda
    def test_reference_correctness(self, tester: DefinitionTest, config: Dict[str, Any]) -> None:
        """Main test method that compares baseline with reference.

        Args:
            tester: DefinitionTest instance (from fixture)
            config: Test configuration (parametrized)
        """
        result: CompareResult = tester.run_single(**config)

        # Build detailed error message
        error_msg = self._build_error_message(config, result)
        assert result.passed, error_msg

    def _build_error_message(self, config: Dict[str, Any], result: CompareResult) -> str:
        """Build detailed error message for failed tests."""
        lines = [f"Definition test failed for config: {config}", "", "Statistics:"]

        if isinstance(result.stats, dict):
            for key, val in result.stats.items():
                if isinstance(val, dict):
                    lines.append(f"  {key}:")
                    for k, v in val.items():
                        if isinstance(v, float):
                            lines.append(f"    {k}: {v:.6e}")
                        else:
                            lines.append(f"    {k}: {v}")
                else:
                    if isinstance(val, float):
                        lines.append(f"  {key}: {val:.6e}")
                    else:
                        lines.append(f"  {key}: {val}")

        if result.details:
            lines.append("")
            lines.append("Details:")
            lines.append(result.details)

        return "\n".join(lines)
