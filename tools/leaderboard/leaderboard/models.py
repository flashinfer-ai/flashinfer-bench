from flask import Blueprint, render_template
from leaderboard.utils.model_utils import list_available_models

blueprint = Blueprint("models", __name__)

@blueprint.route("/models")
def list_models():
    models = list_available_models()
    return render_template("models.html", models=models)