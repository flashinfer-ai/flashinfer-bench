"""Benchmark utility classes for FlashInfer Bench."""

import os
import random
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch


class SeedManager:
    """For reproducibility"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reset_seed()

    def reset_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)


class DeviceManager:
    def __init__(self, device: Union[int, torch.device, str] = None, backend: str = "cuda"):
        self.backend = backend
        self.is_triton = backend == "triton"

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        if device is None:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if self.device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Invalid device id: {self.device.index}. Only {torch.cuda.device_count()} devices available."
            )

        torch.cuda.set_device(self.device)

        if self.is_triton:
            if device is None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device.index)
            os.environ["TORCH_USE_CUDA_DSA"] = "1"

    def get_device_info(self) -> Dict:
        device_id = self.device.index
        return {
            "device_name": torch.cuda.get_device_name(device_id),
            "device_id": device_id,
            "device_str": str(self.device),
            "compute_capability": torch.cuda.get_device_capability(device_id),
            "total_memory": torch.cuda.get_device_properties(device_id).total_memory,
            "cuda": torch.version.cuda,
            "backend": self.backend,
        }


class CorrectnessChecker:
    @staticmethod
    def _tensor_max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.max(torch.abs(a - b)).item()

    @staticmethod
    def _tensor_max_rel(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
        return torch.max(torch.abs(a - b) / (torch.abs(a) + eps)).item()

    @classmethod
    def max_absolute_diff(cls, a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return max(cls.max_absolute_diff(a[k], b[k]) for k in a.keys() & b.keys())
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return cls._tensor_max_abs(a, b)
        raise ValueError(f"Unsupported diff type: {type(a)} and {type(b)}")

    @classmethod
    def max_relative_diff(cls, a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return max(cls.max_relative_diff(a[k], b[k]) for k in a.keys() & b.keys())
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return cls._tensor_max_rel(a, b)
        raise ValueError(f"Unsupported diff type: {type(a)} and {type(b)}")

    @staticmethod
    def validate_shapes(a, b) -> Tuple[bool, str]:
        if isinstance(a, dict) and isinstance(b, dict):
            shared_keys = a.keys() & b.keys()
            missing_1 = b.keys() - a.keys()
            missing_2 = a.keys() - b.keys()
            if missing_1 or missing_2:
                return False, f"Dict keys mismatch, only in a={missing_1}, only in b={missing_2}"

            for k in shared_keys:
                if a[k].shape != b[k].shape:
                    return False, f"Shape mismatch for key '{k}': {a[k].shape} vs {b[k].shape}"

        elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape:
                return False, f"Shape mismatch: {a.shape} vs {b.shape}"
        return True, ""
