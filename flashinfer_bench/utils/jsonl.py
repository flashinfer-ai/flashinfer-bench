import json
from typing import List, Dict

def write_jsonl(path: str, entries: List[Dict]):
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

def append_jsonl(path: str, entries: List[Dict]):
    with open(path, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
