from flask import Blueprint, render_template, abort
import json
import os

blueprint = Blueprint('model', __name__, url_prefix='/model')

MODEL_KERNELS_FILE = 'leaderboard/static/framework_kernels.jsonl'

def load_model_kernels():
    models = {}
    if not os.path.exists(MODEL_KERNELS_FILE):
        return models
    with open(MODEL_KERNELS_FILE) as f:
        for line in f:
            entry = json.loads(line.strip())
            model = entry.get("model")
            kernel_name = entry.get("kernel_name")
            leaderboard_id = entry.get("leaderboard_id")

            if model and kernel_name and leaderboard_id is not None:
                if model not in models:
                    models[model] = []

                models[model].append({
                    "kernel_name": kernel_name,
                    "leaderboard_id": leaderboard_id
                })
    return models


@blueprint.route('')
def model_index():
    models = load_model_kernels()
    return render_template('model_index.html', models=models)

@blueprint.route('/<model_name>/')
def model_detail(model_name):
    models = load_model_kernels()
    if model_name not in models:
        abort(404)
    kernels = models[model_name]
    return render_template('model_detail.html', model_name=model_name, kernels=kernels)
