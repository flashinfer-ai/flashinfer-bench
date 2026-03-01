#!/bin/bash
# =============================================================================
# run_all_traces.sh
# 
# Trace and verify all supported models for flashinfer-bench transformers
# integration.
#
# Usage:
#   ./run_all_traces.sh                    # Run all models
#   ./run_all_traces.sh --models llama-8b  # Run specific model(s)
#   ./run_all_traces.sh --verify-only      # Only run verification
#   ./run_all_traces.sh --clean            # Clean workloads before tracing
#   ./run_all_traces.sh --help             # Show help
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Original dataset with definitions (read-only, not modified by tests)
SOURCE_DATASET_PATH="${SCRIPT_DIR}/../../flashinfer_trace"
# Test output directory (where traces are written during testing)
TEST_OUTPUT_DIR="${SCRIPT_DIR}/../../flashinfer_trace_test"
# By default, use the test output directory to avoid polluting git status
DATASET_PATH="${TEST_OUTPUT_DIR}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Model configurations: key -> "model_id|verify_key"
declare -A MODELS=(
    ["llama-8b"]="meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b"
    ["llama-70b"]="meta-llama/Llama-3.1-70B-Instruct|llama-3.1-70b"
    ["llama-70b-fp8"]="RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8|llama-3.1-70b-fp8"
    ["qwen3-30b"]="Qwen/Qwen3-30B-A3B-Instruct-2507|qwen3-30b-moe"
    ["qwen3-30b-fp8"]="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8|qwen3-30b-moe-fp8"
    ["gpt-120b"]="openai/gpt-oss-120b|gpt-oss-120b"
)

# Default order for running models (smallest to largest)
MODEL_ORDER=("llama-8b" "qwen3-30b" "qwen3-30b-fp8" "llama-70b" "llama-70b-fp8" "gpt-120b")

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --models MODEL1,MODEL2,...   Comma-separated list of models to trace"
    echo "  --verify-only                Only run verification (skip tracing)"
    echo "  --trace-only                 Only run tracing (skip verification)"
    echo "  --clean                      Clean workloads directory before tracing"
    echo "  --use-source                 Write traces to source flashinfer_trace/ instead of test dir"
    echo "  --list                       List available models and exit"
    echo "  --max-tokens N               Max new tokens to generate (default: 16)"
    echo "  --help                       Show this help message"
    echo ""
    echo "Available models:"
    for key in "${MODEL_ORDER[@]}"; do
        IFS='|' read -r model_id verify_key <<< "${MODELS[$key]}"
        echo "  $key -> $model_id"
    done
    echo ""
    echo "Examples:"
    echo "  $0                                    # Trace and verify all models"
    echo "  $0 --models llama-8b                  # Trace only LLaMA-8B"
    echo "  $0 --models llama-8b,qwen3-30b        # Trace LLaMA-8B and Qwen3-30B"
    echo "  $0 --verify-only                      # Only verify existing traces"
    echo "  $0 --clean --models llama-8b          # Clean and trace LLaMA-8B"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo ""
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
}

# Parse arguments
DO_TRACE=true
DO_VERIFY=true
DO_CLEAN=false
USE_SOURCE=false
SELECTED_MODELS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            IFS=',' read -ra SELECTED_MODELS <<< "$2"
            shift 2
            ;;
        --verify-only)
            DO_TRACE=false
            shift
            ;;
        --trace-only)
            DO_VERIFY=false
            shift
            ;;
        --clean)
            DO_CLEAN=true
            shift
            ;;
        --use-source)
            USE_SOURCE=true
            DATASET_PATH="${SOURCE_DATASET_PATH}"
            shift
            ;;
        --list)
            echo "Available models:"
            for key in "${MODEL_ORDER[@]}"; do
                IFS='|' read -r model_id verify_key <<< "${MODELS[$key]}"
                echo "  $key -> $model_id (verify: $verify_key)"
            done
            exit 0
            ;;
        --max-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Use all models if none specified
if [ ${#SELECTED_MODELS[@]} -eq 0 ]; then
    SELECTED_MODELS=("${MODEL_ORDER[@]}")
fi

# Validate selected models
for model_key in "${SELECTED_MODELS[@]}"; do
    if [ -z "${MODELS[$model_key]}" ]; then
        log_error "Unknown model: $model_key"
        echo "Available models: ${!MODELS[*]}"
        exit 1
    fi
done

# Helper function to set up the test output directory
# Copies definitions from source dataset to test directory
setup_test_directory() {
    log_info "Setting up test output directory: ${DATASET_PATH}"
    
    # Create the test directory if it doesn't exist
    mkdir -p "${DATASET_PATH}"
    
    # Copy definitions from source (these are the ground truth definitions)
    if [ -d "${SOURCE_DATASET_PATH}/definitions" ]; then
        rm -rf "${DATASET_PATH}/definitions"
        cp -r "${SOURCE_DATASET_PATH}/definitions" "${DATASET_PATH}/definitions"
        log_info "Copied definitions from ${SOURCE_DATASET_PATH}/definitions"
    else
        log_error "Source definitions not found at ${SOURCE_DATASET_PATH}/definitions"
        exit 1
    fi
    
    # Create empty workloads and blob directories
    mkdir -p "${DATASET_PATH}/workloads"
    mkdir -p "${DATASET_PATH}/blob"
}

# Helper function to clean workloads
clean_workloads() {
    if [ -d "${DATASET_PATH}/workloads" ]; then
        rm -rf "${DATASET_PATH}/workloads"/*
    fi
    if [ -d "${DATASET_PATH}/blob" ]; then
        rm -rf "${DATASET_PATH}/blob"/*
    fi
}

# Clean workloads if requested (initial clean before any tracing)
if [ "$DO_CLEAN" = true ]; then
    log_header "Cleaning Workloads Directory"
    log_info "Removing ${DATASET_PATH}/workloads/*"
    log_info "Removing ${DATASET_PATH}/blob/*"
    clean_workloads
    log_success "Workloads cleaned"
fi

# Track results
declare -A TRACE_RESULTS
declare -A VERIFY_RESULTS

# Run tracing and verification for each model
# Each model gets a clean workload directory to ensure isolated verification
if [ "$DO_TRACE" = true ]; then
    log_header "Tracing and Verifying Models"
    
    # Set up the test output directory with definitions (unless using source)
    if [ "$USE_SOURCE" = false ]; then
        setup_test_directory
        log_info "NOTE: Traces are written to a test directory (${DATASET_PATH})"
        log_info "      to avoid modifying the source flashinfer_trace/ directory."
    else
        log_warning "Using source directory: ${DATASET_PATH}"
        log_warning "This will modify files tracked by git!"
    fi
    
    log_info "Dataset path: ${DATASET_PATH}"
    log_info "Max new tokens: ${MAX_NEW_TOKENS}"
    log_info "Models to trace: ${SELECTED_MODELS[*]}"
    log_info ""
    log_info "Workloads are cleaned between each model to ensure"
    log_info "verification only checks traces from that specific model."
    echo ""

    for model_key in "${SELECTED_MODELS[@]}"; do
        IFS='|' read -r model_id verify_key <<< "${MODELS[$model_key]}"
        
        echo ""
        log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_info "Model: ${model_key} (${model_id})"
        log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Step 1: Clean workloads before tracing this model
        log_info "Step 1: Cleaning workloads directory..."
        clean_workloads
        
        # Step 2: Trace the model
        log_info "Step 2: Tracing model..."
        if python "${SCRIPT_DIR}/trace_models.py" \
            --model "${model_id}" \
            --dataset "${DATASET_PATH}" \
            --max-new-tokens "${MAX_NEW_TOKENS}"; then
            TRACE_RESULTS[$model_key]="success"
            log_success "Tracing completed for ${model_key}"
            
            # Step 3: Verify immediately after tracing (while workloads only contain this model's traces)
            log_info "Step 3: Verifying traces for ${model_key}..."
            if python "${SCRIPT_DIR}/verify_traces.py" \
                --traces "${DATASET_PATH}" \
                --model "${verify_key}" \
                --verbose; then
                VERIFY_RESULTS[$model_key]="success"
            else
                VERIFY_RESULTS[$model_key]="failed"
            fi
        else
            TRACE_RESULTS[$model_key]="failed"
            VERIFY_RESULTS[$model_key]="skipped"
            log_error "Tracing failed for ${model_key}"
        fi
    done
elif [ "$DO_VERIFY" = true ]; then
    # Verify-only mode: verify existing traces (no per-model isolation)
    log_header "Verifying Existing Traces"
    
    # Check if test directory exists
    if [ ! -d "${DATASET_PATH}/workloads" ]; then
        log_warning "Test output directory not found. Using source dataset."
        log_warning "Run tracing first to populate the test directory."
        DATASET_PATH="${SOURCE_DATASET_PATH}"
    fi
    
    log_info "Dataset path: ${DATASET_PATH}"
    log_warning "Note: In verify-only mode, traces from all previously traced models are checked."
    log_warning "      For per-model verification, use tracing mode (which cleans between models)."
    
    # List all traces
    log_info "Listing all traced operators:"
    echo ""
    python "${SCRIPT_DIR}/verify_traces.py" --traces "${DATASET_PATH}" --list
    
    # Verify each model
    for model_key in "${SELECTED_MODELS[@]}"; do
        IFS='|' read -r model_id verify_key <<< "${MODELS[$model_key]}"
        
        echo ""
        log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_info "Verifying: ${model_key} (${verify_key})"
        log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        if python "${SCRIPT_DIR}/verify_traces.py" \
            --traces "${DATASET_PATH}" \
            --model "${verify_key}" \
            --verbose; then
            VERIFY_RESULTS[$model_key]="success"
        else
            VERIFY_RESULTS[$model_key]="failed"
        fi
    done
fi

# Print summary
log_header "Summary"

echo ""
echo "Models processed: ${SELECTED_MODELS[*]}"
echo ""

if [ "$DO_TRACE" = true ]; then
    echo "Tracing Results:"
    for model_key in "${SELECTED_MODELS[@]}"; do
        result="${TRACE_RESULTS[$model_key]:-skipped}"
        if [ "$result" = "success" ]; then
            echo -e "  ${GREEN}✓${NC} ${model_key}"
        elif [ "$result" = "failed" ]; then
            echo -e "  ${RED}✗${NC} ${model_key}"
        else
            echo -e "  ${YELLOW}-${NC} ${model_key} (skipped)"
        fi
    done
    echo ""
fi

# Show verification results (from tracing mode or verify-only mode)
if [ "$DO_TRACE" = true ] || [ "$DO_VERIFY" = true ]; then
    echo "Verification Results:"
    for model_key in "${SELECTED_MODELS[@]}"; do
        result="${VERIFY_RESULTS[$model_key]:-skipped}"
        if [ "$result" = "success" ]; then
            echo -e "  ${GREEN}✓${NC} ${model_key}: All required operators covered"
        elif [ "$result" = "failed" ]; then
            echo -e "  ${RED}✗${NC} ${model_key}: Missing required operators"
        else
            echo -e "  ${YELLOW}-${NC} ${model_key} (skipped)"
        fi
    done
    echo ""
fi

# Check for any failures
has_failures=false
for model_key in "${SELECTED_MODELS[@]}"; do
    if [ "${TRACE_RESULTS[$model_key]}" = "failed" ] || [ "${VERIFY_RESULTS[$model_key]}" = "failed" ]; then
        has_failures=true
        break
    fi
done

if [ "$has_failures" = true ]; then
    log_error "Some operations failed. Check the output above for details."
    exit 1
else
    log_success "All operations completed successfully!"
    exit 0
fi
