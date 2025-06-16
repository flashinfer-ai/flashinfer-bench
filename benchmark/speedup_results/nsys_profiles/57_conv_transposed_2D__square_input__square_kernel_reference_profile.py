
import torch
import os
import sys
import tempfile
import importlib.util

# Set device
device = torch.device("cuda:0")
torch.cuda.set_device(device)

def load_model_from_file(file_path):
    """Load model from Python file"""
    spec = importlib.util.spec_from_file_location("model_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    Model = getattr(module, "Model", None)
    get_init_inputs = getattr(module, "get_init_inputs", None)
    get_inputs = getattr(module, "get_inputs", None)
    
    return Model, get_init_inputs, get_inputs

def load_custom_model_from_file(file_path, build_dir):
    """Load custom model from Python file"""
    # Set build directory environment
    os.environ["TORCH_EXTENSIONS_DIR"] = build_dir
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    spec = importlib.util.spec_from_file_location("custom_model_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    ModelNew = getattr(module, "ModelNew", None)
    return ModelNew

# Load model
print("Loading model...")
ref_code_file = "speedup_results/nsys_profiles/57_conv_transposed_2D__square_input__square_kernel_reference_profile_ref_code.py"
Model, get_init_inputs, get_inputs = load_model_from_file(ref_code_file)

if Model is None or get_init_inputs is None or get_inputs is None:
    print("Error: Could not load required functions from reference code")
    sys.exit(1)

# Initialize model
print("Initializing model...")
torch.manual_seed(42)
init_inputs = get_init_inputs()
init_inputs = [
    x.cuda(device) if isinstance(x, torch.Tensor) else x 
    for x in init_inputs
]

with torch.no_grad():
    torch.manual_seed(42)
    if False:
        # Load custom model
        build_dir = tempfile.mkdtemp()
        custom_code_file = ""
        ModelNew = load_custom_model_from_file(custom_code_file, build_dir)
        if ModelNew is None:
            print("Error: Could not load custom model")
            sys.exit(1)
        model = ModelNew(*init_inputs)
    else:
        # Load reference model
        model = Model(*init_inputs)
    
    model = model.cuda(device)

# Get inputs
print("Preparing inputs...")
torch.manual_seed(42)
inputs = get_inputs()
inputs = [
    x.cuda(device) if isinstance(x, torch.Tensor) else x 
    for x in inputs
]

print("Starting warmup runs...")
# Warmup
for i in range(3):
    with torch.no_grad():
        result = model(*inputs)
        torch.cuda.synchronize(device)
    print(f"Warmup {i+1}/3 completed")

print("Starting profiled runs...")
# Profiled runs
for i in range(10):
    with torch.no_grad():
        result = model(*inputs)
        torch.cuda.synchronize(device)
    print(f"Profile run {i+1}/10 completed")

print("Profiling completed successfully")
