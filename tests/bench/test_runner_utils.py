import sys

import pytest
import torch

from flashinfer_bench.bench.runners.runner_utils import (
    gen_inputs,
    load_safetensors,
    normalize_outputs,
    rand_tensor,
)
from flashinfer_bench.data.definition import AxisConst, Definition, TensorSpec
from flashinfer_bench.data.trace import RandomInput, SafetensorsInput, ScalarInput, Workload


def _def2d():
    """Create a simple 2D definition for testing."""
    return Definition(
        name="d",
        type="op",
        axes={"M": AxisConst(value=2), "N": AxisConst(value=3)},
        inputs={
            "X": TensorSpec(shape=["M", "N"], dtype="float32"),
            "Y": TensorSpec(shape=["M", "N"], dtype="int32"),
            "S": TensorSpec(shape=None, dtype="int32"),
        },
        outputs={"O": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="def run(X, Y, S):\n    return X\n",
    )


class TestRunnerUtils:
    """Test utility functions from runner_utils module."""
    
    def test_rand_tensor_and_normalize_cpu(self):
        """Test random tensor generation and output normalization on CPU."""
        dev = torch.device("cpu")

        # Test float tensor generation
        t = rand_tensor([2, 3], torch.float32, dev)
        assert t.shape == (2, 3) and t.dtype == torch.float32 and t.device.type == "cpu"

        # Test boolean tensor generation  
        b = rand_tensor([4], torch.bool, dev)
        assert b.dtype == torch.bool and b.device.type == "cpu"

        # Test integer tensor generation
        i = rand_tensor([2], torch.int32, dev)
        assert i.dtype == torch.int32 and i.device.type == "cpu"

        # Test output normalization with scalar
        out = normalize_outputs(
            {"Z": 3}, device=dev, output_names=["Z"], output_dtypes={"Z": torch.int32}
        )
        assert out["Z"].dtype == torch.int32 and out["Z"].shape == ()

        # Test output normalization with tensor
        y = torch.tensor([1.0, 2.0], dtype=torch.float32)
        out = normalize_outputs(y, device=dev, output_names=["Y"], output_dtypes={"Y": torch.float32})
        assert torch.allclose(out["Y"], y)

    def test_rand_tensor_special_dtypes(self):
        """Test random tensor generation with special dtypes."""
        dev = torch.device("cpu")
        
        # Test float16
        t_f16 = rand_tensor([2, 2], torch.float16, dev)
        assert t_f16.dtype == torch.float16 and t_f16.device.type == "cpu"
        
        # Test bfloat16  
        t_bf16 = rand_tensor([2, 2], torch.bfloat16, dev)
        assert t_bf16.dtype == torch.bfloat16 and t_bf16.device.type == "cpu"
        
        # Test int8
        t_i8 = rand_tensor([3], torch.int8, dev)
        assert t_i8.dtype == torch.int8 and t_i8.device.type == "cpu"
        assert torch.all(t_i8 >= -128) and torch.all(t_i8 <= 127)
        
        # Test int64
        t_i64 = rand_tensor([2], torch.int64, dev)
        assert t_i64.dtype == torch.int64 and t_i64.device.type == "cpu"

    def test_normalize_outputs_various_formats(self):
        """Test output normalization with various input formats."""
        dev = torch.device("cpu")
        
        # Test dict output
        dict_out = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
        normalized = normalize_outputs(
            dict_out, 
            device=dev, 
            output_names=["a", "b"], 
            output_dtypes={"a": torch.float32, "b": torch.float32}
        )
        assert "a" in normalized and "b" in normalized
        assert normalized["a"].dtype == torch.float32
        
        # Test tuple output
        tuple_out = (torch.tensor(3.0), torch.tensor(4.0))
        normalized = normalize_outputs(
            tuple_out,
            device=dev,
            output_names=["x", "y"],
            output_dtypes={"x": torch.float32, "y": torch.float32}
        )
        assert len(normalized) == 2
        assert normalized["x"].item() == 3.0
        assert normalized["y"].item() == 4.0
        
        # Test list output
        list_out = [torch.tensor(5.0), torch.tensor(6.0)]
        normalized = normalize_outputs(
            list_out,
            device=dev,
            output_names=["p", "q"],
            output_dtypes={"p": torch.float32, "q": torch.float32}
        )
        assert len(normalized) == 2
        assert normalized["p"].item() == 5.0
        assert normalized["q"].item() == 6.0
        
        # Test scalar output
        scalar_out = 7.5
        normalized = normalize_outputs(
            scalar_out,
            device=dev,
            output_names=["scalar"],
            output_dtypes={"scalar": torch.float32}
        )
        assert normalized["scalar"].item() == 7.5
        assert normalized["scalar"].shape == ()

    def test_gen_inputs_random_and_scalar_cpu(self):
        """Test input generation with random and scalar inputs on CPU."""
        d = _def2d()
        wl = Workload(
            axes={"M": 2, "N": 3},
            inputs={"X": RandomInput(), "Y": RandomInput(), "S": ScalarInput(value=7)},
            uuid="w1",
        )
        out = gen_inputs(d, wl, device="cpu", stensors={})
        assert out["X"].shape == (2, 3) and out["X"].dtype == torch.float32
        assert out["Y"].shape == (2, 3) and out["Y"].dtype == torch.int32
        assert out["S"] == 7

    def test_gen_inputs_all_random(self):
        """Test input generation with all random inputs."""
        d = _def2d()
        wl = Workload(
            axes={"M": 2, "N": 3},
            inputs={"X": RandomInput(), "Y": RandomInput(), "S": RandomInput()},
            uuid="w_all_random",
        )
        out = gen_inputs(d, wl, device="cpu", stensors={})
        assert out["X"].shape == (2, 3) and out["X"].dtype == torch.float32
        assert out["Y"].shape == (2, 3) and out["Y"].dtype == torch.int32
        # S is a scalar int32, so should be a Python int or 0-d tensor
        assert isinstance(out["S"], (int, torch.Tensor))

    @pytest.mark.skipif(
        __import__("safetensors", fromlist=["torch"]) is None, reason="safetensors not available"
    )
    def test_load_safetensors_and_gen_inputs_cpu(self, tmp_path):
        """Test loading safetensors and generating inputs with them on CPU."""
        import safetensors.torch as st

        d = _def2d()
        data = {"X": torch.zeros((2, 3), dtype=torch.float32)}
        p = tmp_path / "x.safetensors"
        st.save_file(data, str(p))
        wl = Workload(
            axes={"M": 2, "N": 3},
            inputs={
                "X": SafetensorsInput(path=str(p), tensor_key="X"),
                "S": ScalarInput(value=1),
                "Y": RandomInput(),
            },
            uuid="w2",
        )
        stensors = load_safetensors(d, wl)
        out = gen_inputs(d, wl, device="cpu", stensors=stensors)
        assert torch.allclose(out["X"], data["X"]) and out["X"].device.type == "cpu"

    @pytest.mark.skipif(
        __import__("safetensors", fromlist=["torch"]) is None, reason="safetensors not available"
    )
    def test_load_safetensors_with_traceset_root(self, tmp_path):
        """Test loading safetensors with traceset_root parameter."""
        import safetensors.torch as st

        d = _def2d()
        data = {"X": torch.ones((2, 3), dtype=torch.float32)}
        
        # Create subdirectory structure
        subdir = tmp_path / "traces"
        subdir.mkdir()
        p = subdir / "data.safetensors"
        st.save_file(data, str(p))
        
        wl = Workload(
            axes={"M": 2, "N": 3},
            inputs={"X": SafetensorsInput(path="traces/data.safetensors", tensor_key="X")},
            uuid="w_traceset",
        )
        
        # Load with traceset_root
        stensors = load_safetensors(d, wl, traceset_root=tmp_path)
        assert "X" in stensors
        assert torch.allclose(stensors["X"], data["X"])

    def test_unsupported_dtype_error(self):
        """Test that unsupported dtypes raise appropriate errors."""
        dev = torch.device("cpu")
        
        # This should raise an error for unsupported dtype
        with pytest.raises(ValueError, match="Unsupported random dtype"):
            # Using a complex dtype that shouldn't be supported
            rand_tensor([2, 2], torch.complex64, dev)

    def test_normalize_outputs_error_cases(self):
        """Test error cases in normalize_outputs."""
        dev = torch.device("cpu")
        
        # Test single tensor with multiple expected outputs
        with pytest.raises(RuntimeError, match="Single Tensor returned but multiple outputs are defined"):
            normalize_outputs(
                torch.tensor([1.0, 2.0]),
                device=dev,
                output_names=["a", "b"],
                output_dtypes={"a": torch.float32, "b": torch.float32}
            )
        
        # Test scalar with multiple expected outputs
        with pytest.raises(RuntimeError, match="Scalar returned but multiple outputs are defined"):
            normalize_outputs(
                5.0,
                device=dev,
                output_names=["a", "b"],
                output_dtypes={"a": torch.float32, "b": torch.float32}
            )
        
        # Test tuple/list with wrong number of elements
        with pytest.raises(RuntimeError, match="Tuple/list has .* elements but .* outputs expected"):
            normalize_outputs(
                [torch.tensor(1.0)],  # 1 element
                device=dev,
                output_names=["a", "b"],  # 2 expected
                output_dtypes={"a": torch.float32, "b": torch.float32}
            )
        
        # Test unsupported output type
        with pytest.raises(RuntimeError, match="Unexpected return type"):
            normalize_outputs(
                set([1, 2, 3]),  # Unsupported type
                device=dev,
                output_names=["a"],
                output_dtypes={"a": torch.float32}
            )


if __name__ == "__main__":
    pytest.main(sys.argv)
