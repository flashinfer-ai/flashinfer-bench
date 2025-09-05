# Installation

## Prerequisites

- OS: Linux
- Python: 3.8, 3.9, 3.10, 3.11, 3.12

## Python Package (WIP)

FlashInfer-Bench is available via pip:

```bash
python3 -m pip install flashinfer-bench
```

## Install from Source

You may want to install FlashInfer-Bench from source code for development purposes.

```bash
# Clone the FlashInfer-Bench repository
git clone https://github.com/flashinfer-ai/flashinfer-bench.git

# Install the Python package
pip install -v -e .
```

## Verify installation

```python
>>> import flashinfer_bench as flb
>>> flb.__version__
'0.0.1'
```