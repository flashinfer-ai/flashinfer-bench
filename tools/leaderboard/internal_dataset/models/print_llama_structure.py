from transformers import AutoModelForCausalLM
import sys
import os
import json

def build_structure(module, name='', max_depth=10, current_depth=0):
    if current_depth > max_depth:
        return None

    children = []
    for child_name, child in module.named_children():
        full_name = f'{name}.{child_name}' if name else child_name
        child_structure = build_structure(child, full_name, max_depth, current_depth + 1)
        children.append({
            "type": f"{child_name}: {child.__class__.__name__}",
            "children": child_structure["children"] if child_structure else []
        })

    return {
        "type": f"{name}: {module.__class__.__name__}",
        "children": children
    }

def main(model_path, output_path):
    if not os.path.isdir(model_path):
        print(f"Error: {model_path} is not a valid directory.")
        sys.exit(1)

    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    structure = build_structure(model, name='model')
    
    model_name = os.path.basename(os.path.normpath(model_path))
    full = {
        "model_name": model_name,
        "structure": structure
    }

    with open(output_path, 'w') as f:
        json.dump(full, f, indent=2)

    print(f"Model structure saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the local LLaMA model directory")
    parser.add_argument("--output", default="test.json", help="Output JSON file path")
    args = parser.parse_args()

    main(args.model_path, args.output)
