#!/bin/bash
set -eo pipefail
set -x
echo "Linting..."

# Check if ruff is available, if not install it
if ! command -v ruff &> /dev/null; then
    echo "ruff not found, installing ruff..." && pip install ruff
fi

ruff check . --fix
