"""Specification for workloads, which defines the input tensors for a kernel."""

from typing import Dict, Literal, Union

from .utils import BaseModelWithDocstrings, NonEmptyString, NonNegativeInt


class RandomInput(BaseModelWithDocstrings):
    """Random input generation descriptor.

    Represents a specification for generating random tensor input data
    during workload execution and benchmarking.
    """

    type: Literal["random"] = "random"
    """The input type identifier for random data generation."""


class ScalarInput(BaseModelWithDocstrings):
    """Scalar literal input specification.

    Represents a scalar value (integer, float, or boolean) that will be
    used as a direct input parameter to the computational workload.
    """

    type: Literal["scalar"] = "scalar"
    """The input type identifier for scalar values."""
    value: Union[int, float, bool]
    """The scalar value to be used as input. Must be int, float, or bool."""


class SafetensorsInput(BaseModelWithDocstrings):
    """Input specification for data loaded from safetensors files.

    Represents tensor data that will be loaded from a safetensors file
    using a specific tensor key within that file.
    """

    type: Literal["safetensors"] = "safetensors"
    """The input type identifier for safetensors data."""
    path: NonEmptyString
    """Path to the safetensors file containing the tensor data. The path is relative to the root
    path of the TraceSet."""
    tensor_key: NonEmptyString
    """Key identifier for the specific tensor within the safetensors file."""


InputSpec = Union[RandomInput, SafetensorsInput, ScalarInput]
"""Union type representing all possible input specification types."""


class Workload(BaseModelWithDocstrings):
    """Concrete workload configuration for benchmarking.

    Defines a specific instance of a computational workload with concrete
    values for all variable axes and specifications for all input data.
    This represents an executable configuration that can be benchmarked.
    """

    axes: Dict[str, NonNegativeInt]
    """Dictionary mapping axis names to their concrete integer values. All values must be
    positive."""
    inputs: Dict[str, InputSpec]
    """Dictionary mapping input names to their data specifications."""
    uuid: NonEmptyString
    """Unique identifier for this specific workload configuration."""
