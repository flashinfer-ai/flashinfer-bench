"""Test definitions: schema correctness, reference correctness, etc."""

import json
import math
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest
import torch

from flashinfer_bench.data import Definition
from flashinfer_bench.env import get_fib_dataset_path
from flashinfer_bench.utils import dtype_str_to_torch_dtype

from .comparators import Comparator, CompareResult, MultiOutputComparator, TensorComparator
from .pytest_config import requires_torch_cuda


class DefinitionRunner:
    """Runs reference and baseline implementations and compares outputs.

    Handles extracting reference implementation from definition, generating inputs,
    executing both implementations, and comparing results.
    """

    def __init__(
        self,
        definition: dict,
        baseline_fn: Callable[..., Any],
        input_generator: Optional[Callable[..., Dict[str, Any]]] = None,
        comparator: Optional[Comparator] = None,
        atol: float = 1e-2,
        rtol: float = 5e-2,
        device: str = "cuda",
    ):
        """Initialize the runner with definition and baseline function.

        Parameters
        ----------
        definition : dict
            Definition dict (already loaded from JSON).
        baseline_fn : Callable[..., Any]
            Baseline function to compare against reference.
        input_generator : Callable[..., Dict[str, Any]], optional
            Custom input generator function. If None, uses default generator.
        comparator : Comparator, optional
            Custom comparator. If None, auto-inferred from definition outputs.
        atol : float
            Absolute tolerance for comparison.
        rtol : float
            Relative tolerance for comparison.
        device : str
            Device to generate tensors on.
        """
        self.definition = definition
        self.reference_fn = self._extract_reference()
        self.baseline_fn = baseline_fn
        self.input_generator = input_generator
        self.atol = atol
        self.rtol = rtol
        self.device = device
        self.comparator = comparator or self._infer_comparator()

    def _extract_reference(self) -> Callable:
        """Extract the run() function from definition's reference field.

        Raises
        ------
        ValueError
            If reference code does not contain a 'run' function.
        """
        code = self.definition["reference"]
        namespace: Dict[str, Any] = {"torch": torch, "math": math}
        exec(code, namespace)  # noqa: S102

        if "run" not in namespace:
            raise ValueError("Definition reference must contain a 'run' function")

        return namespace["run"]

    def _infer_comparator(self) -> Comparator:
        """Infer comparator from definition outputs."""
        outputs = self.definition.get("outputs", {})

        if len(outputs) == 1:
            return TensorComparator(atol=self.atol, rtol=self.rtol)
        else:
            output_names = list(outputs.keys())
            return MultiOutputComparator(output_names, atol=self.atol, rtol=self.rtol)

    def _generate_inputs(self, **config: Any) -> Dict[str, Any]:
        """Generate inputs for the test."""
        if self.input_generator:
            return self.input_generator(**config)
        return self._generate_default_inputs(**config)

    def _generate_default_inputs(self, **axis_values: Any) -> Dict[str, Any]:
        """Generate inputs based on definition schema."""
        inputs: Dict[str, Any] = {}
        input_specs = self.definition.get("inputs", {})
        axes = self.definition.get("axes", {})

        for name, spec in input_specs.items():
            shape = self._resolve_shape(spec.get("shape"), axes, axis_values)
            dtype = dtype_str_to_torch_dtype(spec["dtype"])

            if shape is None:
                # Scalar value
                if dtype in (torch.float32, torch.float16, torch.bfloat16):
                    inputs[name] = 1.0
                elif dtype == torch.bool:
                    inputs[name] = True
                else:
                    inputs[name] = 1
            else:
                # Tensor
                if dtype == torch.bool:
                    tensor = torch.randint(0, 2, shape, dtype=dtype, device=self.device)
                elif dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                    tensor = torch.randint(0, 100, shape, dtype=dtype, device=self.device)
                else:
                    tensor = torch.randn(shape, device=self.device).to(dtype)
                inputs[name] = tensor

        return inputs

    def _resolve_shape(
        self, shape_spec: Optional[List[str]], axes: Dict[str, Any], axis_values: Dict[str, Any]
    ) -> Optional[List[int]]:
        """Resolve shape specification to concrete dimensions.

        Raises
        ------
        ValueError
            If a variable axis is not provided or an unknown axis is referenced.
        """
        if shape_spec is None:
            return None  # scalar

        resolved = []
        for dim in shape_spec:
            if dim in axis_values:
                resolved.append(int(axis_values[dim]))
            elif dim in axes:
                axis = axes[dim]
                if axis["type"] == "const":
                    resolved.append(int(axis["value"]))
                else:
                    raise ValueError(f"Variable axis '{dim}' not provided in config")
            else:
                raise ValueError(f"Unknown axis: {dim}")

        return resolved

    def run(self, **config: Any) -> CompareResult:
        """Generate inputs, execute reference and baseline, then compare outputs.

        Parameters
        ----------
        **config : Any
            Configuration values for input generation (axis values).

        Returns
        -------
        CompareResult
            Comparison result with pass/fail status and statistics.
        """
        inputs = self._generate_inputs(**config)

        # Execute reference and baseline
        with torch.no_grad():
            ref_output = self.reference_fn(**inputs)
            baseline_output = self.baseline_fn(**inputs)

        # Compare outputs
        return self.comparator.compare(ref_output, baseline_output)


class DefinitionTest:
    """Pytest-compatible test case base class for Definition-based testing.

    Inherit from this class to create parametrized tests that compare
    a baseline implementation against the reference from a Definition JSON.
    The baseline function must have the same interface as the reference's run().

    Example::

        class TestGQAPagedDecode(DefinitionTest):
            definition_path = "definitions/gqa_paged/gqa_paged_decode.json"
            configs = [{"batch_size": 1}, {"batch_size": 4}]

            def baseline_fn(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
                # Wrap FlashInfer to match reference interface
                wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(...)
                wrapper.plan(...)
                return wrapper.run(...)
    """

    definition_path: str = ""
    """Path to definition JSON file (relative to FIB_DATASET_PATH)."""
    configs: List[Dict[str, Any]] = []
    """List of test configurations (axis values)."""

    input_generator: Optional[Callable[..., Dict[str, Any]]] = None
    """Custom input generator function."""
    comparator: Optional[Comparator] = None
    """Custom comparator for output comparison."""
    atol: float = 1e-2
    """Absolute tolerance for comparison."""
    rtol: float = 5e-2
    """Relative tolerance for comparison."""
    device: str = "cuda"
    """Device to run tests on."""

    @abstractmethod
    def baseline_fn(self, **inputs: Any) -> Any:
        """Baseline implementation with same interface as reference run().

        Parameters
        ----------
        **inputs : Any
            Input tensors matching the definition's input specification.

        Returns
        -------
        Any
            Output tensor(s) matching the definition's output specification.
        """
        raise NotImplementedError("Subclass must implement baseline_fn")

    @pytest.fixture(scope="class")
    def definition(self) -> dict:
        """Load definition from definition_path.

        Returns
        -------
        dict
            The loaded definition dictionary.
        """
        path = Path(self.definition_path)
        if not path.is_absolute():
            dataset_path = get_fib_dataset_path()
            path = dataset_path / path
        if not path.exists():
            pytest.fail(f"Definition file not found: {path}")
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in definition file {path}: {e}")

    @pytest.fixture(params=[])
    def config(self, request: pytest.FixtureRequest) -> Dict[str, Any]:
        """Fixture that provides test configurations.

        Parameters
        ----------
        request : pytest.FixtureRequest
            Pytest fixture request object.

        Returns
        -------
        Dict[str, Any]
            A single test configuration (axis values).
        """
        return request.param

    def pytest_generate_tests(self, metafunc: pytest.Metafunc) -> None:
        """Generate parametrized test cases from configs.

        Parameters
        ----------
        metafunc : pytest.Metafunc
            Pytest metafunc object for test parametrization.
        """
        if "config" in metafunc.fixturenames and self.configs:
            ids = [self._config_to_id(c) for c in self.configs]
            metafunc.parametrize("config", self.configs, ids=ids)

    @staticmethod
    def _config_to_id(config: Dict[str, Any]) -> str:
        """Convert config dict to a readable test ID."""
        return "-".join(f"{k}={v}" for k, v in config.items())

    def test_definition_schema(self, definition: dict) -> None:
        """Test that the definition conforms to the Definition schema.

        Parameters
        ----------
        definition : dict
            The definition dictionary to validate.
        """
        Definition.model_validate(definition)

    def test_reference_executable(self, definition: dict) -> None:
        """Test that the reference code can be executed and has a callable 'run' function.

        Parameters
        ----------
        definition : dict
            The definition dictionary containing the reference code.
        """
        code = definition.get("reference")
        if code is None:
            name = definition.get("name", "unknown")
            pytest.skip(f"No reference code in definition '{name}'")

        namespace: Dict[str, Any] = {"torch": torch, "math": math}

        try:
            exec(code, namespace)  # noqa: S102
        except Exception as e:
            raise AssertionError(f"Reference code failed to execute: {e}") from e

        assert "run" in namespace, "Reference must define a 'run' function"
        assert callable(namespace["run"]), "Reference 'run' must be callable"

    @requires_torch_cuda
    def test_reference_correctness(self, definition: dict, config: Dict[str, Any]) -> None:
        """Test that baseline output matches reference output.

        Parameters
        ----------
        definition : dict
            The definition dictionary.
        config : Dict[str, Any]
            Test configuration (axis values).
        """
        if definition.get("reference") is None:
            name = definition.get("name", "unknown")
            pytest.skip(f"No reference code in definition '{name}'")

        runner = DefinitionRunner(
            definition=definition,
            baseline_fn=self.baseline_fn,
            input_generator=self.input_generator,
            comparator=self.comparator,
            atol=self.atol,
            rtol=self.rtol,
            device=self.device,
        )
        result = runner.run(**config)
        assert result.passed, self._build_error_message(config, result)

    def _build_error_message(self, config: Dict[str, Any], result: CompareResult) -> str:
        """Build detailed error message for failed tests."""
        lines = [f"Definition test failed for config: {config}", "", "Statistics:"]

        if isinstance(result.stats, dict):
            for key, val in result.stats.items():
                if isinstance(val, dict):
                    lines.append(f"  {key}:")
                    for k, v in val.items():
                        lines.append(
                            f"    {k}: {v:.6e}" if isinstance(v, float) else f"    {k}: {v}"
                        )
                else:
                    lines.append(
                        f"  {key}: {val:.6e}" if isinstance(val, float) else f"  {key}: {val}"
                    )

        if result.details:
            lines.extend(["", "Details:", result.details])

        return "\n".join(lines)
