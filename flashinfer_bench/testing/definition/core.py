"""Core definition testing functionality."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from flashinfer_bench.data import Definition
from flashinfer_bench.env import get_fib_dataset_path
from flashinfer_bench.utils import dtype_str_to_torch_dtype

from .comparators import Comparator, CompareResult, MultiOutputComparator, TensorComparator


@dataclass
class TestResult:
    """Result of running multiple test configurations."""

    passed: int
    total: int
    results: List[CompareResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return self.passed == self.total

    def __repr__(self) -> str:
        return f"TestResult(passed={self.passed}/{self.total})"


class DefinitionTest:
    """Test runner that extracts reference from Definition and compares with baseline.

    This class automatically:
    1. Loads the Definition JSON file
    2. Extracts the reference implementation (run function)
    3. Generates inputs based on Definition schema (or uses custom generator)
    4. Compares reference output with baseline output

    Example:
        def flashinfer_adapter(q, k_cache, v_cache, ...):
            # Wrap FlashInfer to match reference interface
            ...

        test = DefinitionTest(
            definition="path/to/definition.json",
            baseline_fn=flashinfer_adapter,
            configs=[{"batch_size": 4}, {"batch_size": 8}],
        )
        result = test.run_all()
        assert result.all_passed
    """

    def __init__(
        self,
        definition: Union[str, Path, dict],
        baseline_fn: Callable[..., Any],
        input_generator: Optional[Callable[..., Dict[str, Any]]] = None,
        comparator: Optional[Comparator] = None,
        configs: Optional[List[Dict[str, Any]]] = None,
        device: str = "cuda",
    ):
        """Initialize the definition test.

        Args:
            definition: Path to Definition JSON file or dict
            baseline_fn: Baseline function with same interface as reference run()
            input_generator: Optional custom input generator function
            comparator: Optional custom comparator (auto-inferred from outputs if None)
            configs: List of test configurations (axis values)
            device: Device to run tests on
        """
        self.definition = self._load_definition(definition)
        self.reference_fn = self._extract_reference()
        self.baseline_fn = baseline_fn
        self.input_generator = input_generator
        self.comparator = comparator or self._infer_comparator()
        self.configs = configs or [{}]
        self.device = device

    def _load_definition(self, definition: Union[str, Path, dict]) -> dict:
        """Load definition from file or dict.

        If definition is a relative path, it is resolved relative to FIB_DATASET_PATH.
        """
        if isinstance(definition, dict):
            return definition

        path = Path(definition)

        # If path is relative, resolve it relative to FIB_DATASET_PATH
        if not path.is_absolute():
            dataset_path = get_fib_dataset_path()
            path = dataset_path / path

        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def validate_schema(self) -> None:
        """Validate that the definition conforms to the Definition schema.

        Raises:
            pydantic.ValidationError: If the definition does not conform to schema.
            json.JSONDecodeError: If the definition is not valid JSON.
        """
        # This will raise ValidationError if schema is invalid
        Definition.model_validate(self.definition)

    def _extract_reference(self) -> Callable:
        """Extract the run() function from definition's reference field."""
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
            return TensorComparator()
        else:
            output_names = list(outputs.keys())
            return MultiOutputComparator(output_names)

    def _resolve_shape(
        self, shape_spec: Optional[List[str]], axis_values: Dict[str, Any]
    ) -> Optional[List[int]]:
        """Resolve shape specification to concrete dimensions."""
        if shape_spec is None:
            return None  # scalar

        resolved = []
        axes = self.definition.get("axes", {})

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

    def generate_default_inputs(self, **axis_values: Any) -> Dict[str, Any]:
        """Generate inputs based on definition schema.

        Args:
            **axis_values: Values for variable axes

        Returns:
            Dict of input tensors/scalars
        """
        inputs: Dict[str, Any] = {}
        input_specs = self.definition.get("inputs", {})

        for name, spec in input_specs.items():
            shape = self._resolve_shape(spec.get("shape"), axis_values)
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

    def run_single(self, **axis_values: Any) -> CompareResult:
        """Run a single test configuration.

        Args:
            **axis_values: Values for variable axes

        Returns:
            CompareResult with comparison results
        """
        # Generate inputs
        if self.input_generator:
            inputs = self.input_generator(**axis_values)
        else:
            inputs = self.generate_default_inputs(**axis_values)

        # Filter out metadata keys (starting with '_') for reference function
        # These are extra inputs that baseline may need but reference doesn't
        ref_inputs = {k: v for k, v in inputs.items() if not k.startswith("_")}

        # Run reference and baseline with same interface
        with torch.no_grad():
            ref_output = self.reference_fn(**ref_inputs)
            baseline_output = self.baseline_fn(**inputs)

        # Compare outputs
        return self.comparator.compare(ref_output, baseline_output)

    def run_all(self, verbose: bool = True) -> TestResult:
        """Run all test configurations.

        Args:
            verbose: Print progress and results

        Returns:
            TestResult with aggregated results
        """
        results: List[CompareResult] = []
        passed = 0

        if verbose:
            print(f"\nRunning {len(self.configs)} test configurations...")
            print(f"Definition: {self.definition.get('name', 'unknown')}")
            print("=" * 60)

        for i, config in enumerate(self.configs):
            try:
                result = self.run_single(**config)
                results.append(result)

                if result.passed:
                    passed += 1
                    status = "✓ PASSED"
                else:
                    status = "✗ FAILED"

                if verbose:
                    print(f"\n[{i+1}/{len(self.configs)}] {status}")
                    print(f"  Config: {config}")
                    if isinstance(result.stats, dict):
                        for key, val in result.stats.items():
                            if isinstance(val, dict):
                                print(f"  {key}:")
                                for k, v in val.items():
                                    print(
                                        f"    {k}: {v:.6e}"
                                        if isinstance(v, float)
                                        else f"    {k}: {v}"
                                    )
                            else:
                                print(
                                    f"  {key}: {val:.6e}"
                                    if isinstance(val, float)
                                    else f"  {key}: {val}"
                                )
                    if result.details:
                        print(f"  Details: {result.details}")

            except Exception as e:
                if verbose:
                    print(f"\n[{i+1}/{len(self.configs)}] ✗ ERROR")
                    print(f"  Config: {config}")
                    print(f"  Error: {e}")
                results.append(CompareResult(passed=False, stats={}, details=str(e)))

        if verbose:
            print("\n" + "=" * 60)
            print(f"Summary: {passed}/{len(self.configs)} tests passed")

        return TestResult(passed=passed, total=len(self.configs), results=results)
