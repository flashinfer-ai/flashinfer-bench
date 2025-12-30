#!/bin/bash
# Build Sphinx documentation locally
#
# Usage:
#   ./build_docs.sh          # Build HTML docs
#   ./build_docs.sh clean    # Clean build directory
#   ./build_docs.sh deps     # Install dependencies
#   ./build_docs.sh serve    # Build and serve locally

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/_build"

cd "$SCRIPT_DIR"

# Install dependencies if needed
install_deps() {
    echo "Installing documentation dependencies..."
    pip install -r requirements.txt
    pip install -e "$REPO_ROOT"
}

# Clean build directory
clean() {
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
}

# Build HTML documentation
build() {
    echo "Building HTML documentation..."
    sphinx-build -b html . "$BUILD_DIR/html" -W --keep-going
    echo ""
    echo "Documentation built successfully!"
    echo "Open: file://$BUILD_DIR/html/index.html"
}

# Serve documentation locally
serve() {
    build
    echo ""
    echo "Serving documentation at http://localhost:8000"
    python -m http.server 8000 --directory "$BUILD_DIR/html"
}

case "${1:-build}" in
    clean)
        clean
        ;;
    deps)
        install_deps
        ;;
    serve)
        serve
        ;;
    build|"")
        build
        ;;
    *)
        echo "Usage: $0 {build|clean|deps|serve}"
        exit 1
        ;;
esac
