#!/bin/bash
set -eo pipefail
set -x
echo "Linting..."

ruff check . --fix
