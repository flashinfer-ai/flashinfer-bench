"""Tests for RMSNorm definitions."""

import sys

import flashinfer
import pytest
import torch

from flashinfer_bench.testing import DefinitionTest


def generate_rmsnorm_inputs(batch_size: int, hidden_size: int, device: str = "cuda"):
    """Generate random inputs for RMSNorm testing.

    Note: Parameter names must match the Definition's inputs spec.
    For rmsnorm_h128.json, this is 'hidden_states' and 'weight'.
    """
    hidden_states = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)

    return {"hidden_states": hidden_states, "weight": weight}


class TestRMSNormH128(DefinitionTest):
    """Test RMSNorm with hidden_size=128."""

    # Relative path to FIB_DATASET_PATH
    definition_path = "definitions/rmsnorm/rmsnorm_h128.json"
    configs = [
        {"batch_size": 1},
        {"batch_size": 4},
        {"batch_size": 8},
        {"batch_size": 16},
        {"batch_size": 32},
    ]
    atol = 8e-3
    rtol = 1e-2

    @staticmethod
    def input_generator(**config):
        return generate_rmsnorm_inputs(batch_size=config["batch_size"], hidden_size=128)

    def baseline_fn(self, hidden_states, weight):
        """FlashInfer baseline implementation.

        Note: Interface must match the Definition's run() function signature.
        """
        # eps is hardcoded in the definition as 1e-6
        return flashinfer.norm.rmsnorm(hidden_states.contiguous(), weight.contiguous(), eps=1e-6)


class TestRMSNormH2048(DefinitionTest):
    """Test RMSNorm with hidden_size=2048."""

    # Relative path to FIB_DATASET_PATH
    definition_path = "definitions/rmsnorm/rmsnorm_h2048.json"
    configs = [{"batch_size": 1}, {"batch_size": 4}, {"batch_size": 8}]
    atol = 8e-3
    rtol = 1e-2

    @staticmethod
    def input_generator(**config):
        return generate_rmsnorm_inputs(batch_size=config["batch_size"], hidden_size=2048)

    def baseline_fn(self, hidden_states, weight):
        """FlashInfer baseline implementation."""
        return flashinfer.norm.rmsnorm(hidden_states.contiguous(), weight.contiguous(), eps=1e-6)


class TestRMSNormH4096(DefinitionTest):
    """Test RMSNorm with hidden_size=4096."""

    definition_path = "definitions/rmsnorm/rmsnorm_h4096.json"
    configs = [{"batch_size": 1}, {"batch_size": 4}, {"batch_size": 8}]
    atol = 8e-3
    rtol = 1e-2

    @staticmethod
    def input_generator(**config):
        return generate_rmsnorm_inputs(batch_size=config["batch_size"], hidden_size=4096)

    def baseline_fn(self, hidden_states, weight):
        """FlashInfer baseline implementation."""
        return flashinfer.norm.rmsnorm(hidden_states.contiguous(), weight.contiguous(), eps=1e-6)


class TestRMSNormH7168(DefinitionTest):
    """Test RMSNorm with hidden_size=7168."""

    definition_path = "definitions/rmsnorm/rmsnorm_h7168.json"
    configs = [{"batch_size": 1}, {"batch_size": 4}, {"batch_size": 8}]
    atol = 8e-3
    rtol = 1e-2

    @staticmethod
    def input_generator(**config):
        return generate_rmsnorm_inputs(batch_size=config["batch_size"], hidden_size=7168)

    def baseline_fn(self, hidden_states, weight):
        """FlashInfer baseline implementation."""
        return flashinfer.norm.rmsnorm(hidden_states.contiguous(), weight.contiguous(), eps=1e-6)


if __name__ == "__main__":
    pytest.main(sys.argv)
