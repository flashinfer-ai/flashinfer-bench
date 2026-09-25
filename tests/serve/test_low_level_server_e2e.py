"""End-to-end tests for the low-level GPU server.

These tests require a real GPU and start a real low-level server process.
"""

import hashlib
import json
import socket
import subprocess
import sys
import tempfile
import time
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import numpy as np
import pytest
import requests
import tvm_ffi

pytestmark = pytest.mark.requires_torch_cuda


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(port: int, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if response.status_code == 200:
                return
        except Exception as error:  # pragma: no cover - polling helper
            last_error = error
        time.sleep(0.2)
    raise RuntimeError(f"Server did not start in time: {last_error}")


def _parse_multipart_response(response: requests.Response) -> tuple[dict, dict[str, bytes]]:
    content_type = response.headers["content-type"]
    mime_message = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        + response.content
    )
    message = BytesParser(policy=default).parsebytes(mime_message)
    parts: dict[str, bytes] = {}
    result = None
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        if name == "result":
            result = json.loads(payload.decode("utf-8"))
        else:
            parts[name] = payload
    assert result is not None
    return result, parts


def test_low_level_gpu_server_e2e():
    port = _find_free_port()
    command = [
        sys.executable,
        "-m",
        "flashinfer_bench.serve.low_level.app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--devices",
        "cuda:0",
        "--timeout",
        "30",
        "--log-level",
        "INFO",
    ]
    server_process = subprocess.Popen(command)
    try:
        _wait_for_server(port, timeout_seconds=60.0)

        module_source = b"""
import torch
import tvm_ffi

def _to_torch(tensor):
    return torch.utils.dlpack.from_dlpack(tensor)

@tvm_ffi.register_global_func("e2e_scale")
def e2e_scale(tensor):
    torch_tensor = _to_torch(tensor)
    return torch_tensor + torch_tensor

@tvm_ffi.register_global_func("e2e_sum")
def e2e_sum(tensor):
    torch_tensor = _to_torch(tensor)
    return float(torch_tensor.sum().item())

@tvm_ffi.register_global_func("e2e_fail")
def e2e_fail():
    raise RuntimeError("expected e2e failure")
"""
        input_array = np.arange(8, dtype=np.float16)
        tensor_bytes = input_array.tobytes()

        module_hash = hashlib.sha256(module_source).hexdigest()
        tensor_hash = hashlib.sha256(tensor_bytes).hexdigest()

        success_program = {
            "instructions": [
                {"op": "upload_python_module", "blob": module_hash},
                {
                    "op": "upload_tensor",
                    "dst": 1,
                    "blob": tensor_hash,
                    "shape": [8],
                    "dtype": "float16",
                },
                {"op": "call", "dst": 2, "func": "e2e_scale", "args": [{"r": 1}]},
                {"op": "call", "dst": 3, "func": "e2e_sum", "args": [{"r": 2}]},
                {"op": "return", "reg": 2, "key": "output"},
                {"op": "return", "reg": 3, "key": "total"},
            ],
            "timeout_seconds": 30,
        }

        for _ in range(2):
            response = requests.post(
                f"http://127.0.0.1:{port}/execute",
                files={
                    "program": (
                        "program.json",
                        json.dumps(success_program).encode("utf-8"),
                        "application/json",
                    ),
                    f"blob:{module_hash}": ("module.py", module_source, "application/octet-stream"),
                    f"blob:{tensor_hash}": ("tensor.bin", tensor_bytes, "application/octet-stream"),
                },
                timeout=120.0,
            )
            assert response.status_code == 200, response.text
            result, binary_parts = _parse_multipart_response(response)
            assert result["status"] == "ok"
            assert result["returns"]["total"]["type"] == "json"
            assert result["returns"]["total"]["value"] == pytest.approx(56.0)
            output_part = result["returns"]["output"]["blob"]
            output_array = np.frombuffer(binary_parts[output_part], dtype=np.float16)
            np.testing.assert_allclose(output_array, input_array * 2)

        failure_program = {
            "instructions": [
                {
                    "op": "upload_tensor",
                    "dst": 0,
                    "blob": "missing_blob",
                    "shape": [8],
                    "dtype": "float16",
                }
            ],
            "timeout_seconds": 30,
        }
        failure_response = requests.post(
            f"http://127.0.0.1:{port}/execute",
            files={
                "program": (
                    "program.json",
                    json.dumps(failure_program).encode("utf-8"),
                    "application/json",
                )
            },
            timeout=30.0,
        )
        assert failure_response.status_code == 400
        failure_payload = failure_response.json()
        assert failure_payload["error"] == "invalid_program"
        assert "Missing blob" in failure_payload["message"]
    finally:
        server_process.terminate()
        server_process.wait(timeout=30.0)


def test_low_level_gpu_server_module_handle_e2e():
    port = _find_free_port()
    command = [
        sys.executable,
        "-m",
        "flashinfer_bench.serve.low_level.app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--devices",
        "cuda:0",
        "--timeout",
        "30",
        "--log-level",
        "INFO",
    ]
    server_process = subprocess.Popen(command)
    try:
        _wait_for_server(port, timeout_seconds=60.0)

        with tempfile.TemporaryDirectory(prefix="fib_low_level_cpp_") as temp_dir:
            source_path = Path(temp_dir) / "module.cc"
            source_path.write_text(
                """
#include <tvm/ffi/function.h>

int64_t ReturnSeven() {
  return 7;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(return_seven, ReturnSeven);
""",
                encoding="utf-8",
            )
            so_path = tvm_ffi.cpp.build(
                name="low_level_module_handle_e2e",
                cpp_files=[str(source_path)],
                build_directory=temp_dir,
            )
            so_bytes = Path(so_path).read_bytes()

        module_hash = hashlib.sha256(so_bytes).hexdigest()
        program = {
            "instructions": [
                {"op": "upload_ffi_module", "dst": 0, "blob": module_hash},
                {"op": "call", "dst": 1, "func": "return_seven", "module": {"r": 0}, "args": []},
                {"op": "return", "reg": 1, "key": "value"},
            ],
            "timeout_seconds": 30,
        }

        response = requests.post(
            f"http://127.0.0.1:{port}/execute",
            files={
                "program": (
                    "program.json",
                    json.dumps(program).encode("utf-8"),
                    "application/json",
                ),
                f"blob:{module_hash}": ("module.so", so_bytes, "application/octet-stream"),
            },
            timeout=120.0,
        )
        assert response.status_code == 200, response.text
        result, binary_parts = _parse_multipart_response(response)
        assert binary_parts == {}
        assert result["status"] == "ok"
        assert result["returns"]["value"]["type"] == "json"
        assert result["returns"]["value"]["value"] == 7
    finally:
        server_process.terminate()
        server_process.wait(timeout=30.0)


if __name__ == "__main__":
    pytest.main(sys.argv)
