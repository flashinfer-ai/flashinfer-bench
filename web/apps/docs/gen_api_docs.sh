#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root from this script's location: web/apps/doc/gen_api_docs.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$ROOT_DIR"

if ! command -v pydoc-markdown >/dev/null 2>&1; then
  echo "pydoc-markdown is not installed. Install with:"
  echo "  pip install pydoc-markdown>=4"
  exit 1
fi

echo "Generating API docs → docs/api/reference.md (from flashinfer_bench) ..."

# Build an explicit module list from the filesystem to include submodules
mapfile -t MODULES < <( \
  find flashinfer_bench -type f -name "*.py" \
  | sed -e 's#^flashinfer_bench/##' -e 's#/__init__\.py$##' -e 's#\.py$##' \
  | sed -e 's#/#.#g' \
  | grep -vE '^__init__$' \
  | awk 'NF{print "flashinfer_bench" (length($0)? "." $0:"")}' \
  | sort -u \
)

# Generate a temporary YAML config that enumerates all modules
TMP_YML=$(mktemp -t pydocmd.XXXXXX.yml)
OUT_MD="$ROOT_DIR/docs/api/reference.md"
echo "Found ${#MODULES[@]} modules. Writing to $OUT_MD"
{
  echo "loaders:"
  echo "  - type: python"
  echo "    search_path: [\"$ROOT_DIR\"]"
  echo "    modules:"
  for m in "${MODULES[@]}"; do
    echo "      - \"$m\""
  done
  echo "processors:"
  echo "  - type: smart"
  echo "renderer:"
  echo "  type: markdown"
  echo "  filename: $OUT_MD"
  echo "  render_toc: false"
} > "$TMP_YML"

set +e
pydoc-markdown "$TMP_YML"
STATUS=$?
set -e
rm -f "$TMP_YML"

if [ $STATUS -ne 0 ]; then
  echo "Failed to generate API docs. Ensure flashinfer_bench is importable in this Python env." >&2
  echo "Try: pip install -e .  and re-run." >&2
  exit $STATUS
fi

echo "Done."
