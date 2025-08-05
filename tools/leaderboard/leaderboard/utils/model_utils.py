import os
import json
from pathlib import Path

current_dir = Path(__file__).parent
leaderboard_root = current_dir.parent.parent  # Go up to tools/leaderboard
MODELS_DIR = os.path.join(leaderboard_root, "internal_dataset", "models")

def get_model_structure(model_name: str):
    """Get the structure of a specific model."""
    
    model_path = os.path.join(MODELS_DIR, f"{model_name}.json")
    
    if not os.path.exists(model_path):
        return None

    with open(model_path, "r") as f:
        data = json.load(f)
        return data

def list_available_models():
    """List all available models in the models directory."""
    
    models = []
    for filename in sorted(os.listdir(MODELS_DIR)):
        if filename.endswith(".json"):
            model_name = filename[:-5]  # remove ".json"
            try:
                with open(os.path.join(MODELS_DIR, filename), "r") as f:
                    metadata = json.load(f)
                    models.append({
                        "file_name": model_name,
                        "model_name": metadata.get("model_name", model_name)
                    })
            except Exception:
                continue
    return models