import json
from pathlib import Path
from typing import Optional


class KernelCallLogger:
    def __init__(
        self,
        name: str,
        type: str,
        output_dir: str = "YOUR_PATH_TO/workloads",
        environment: Optional[dict] = None,
    ):
        self.name = name
        self.type = type
        self.env = environment or {}
        self.output_dir = Path(output_dir) / type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.call_index = 0
        self.log_file = self.output_dir / f"{self.name}.wrokload.jsonl"

        # # Optional: Write metadata header as first line
        # header = {
        #     "name": self.name,
        #     "type": self.type,
        #     "environment": self.env,
        #     "format": "jsonl",
        #     "version": 1
        # }
        # with open(self.log_file, "w") as f:
        #     f.write(json.dumps(header) + "\n")

    def log_call(self, inputs: dict, axes: dict):
        input_shapes = {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype)
            }
            for name, tensor in inputs.items()
        }

        entry = {
            "axes": axes,
            "input_shapes": input_shapes
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self.call_index += 1
