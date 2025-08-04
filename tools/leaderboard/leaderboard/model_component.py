from flask import Blueprint, render_template
from leaderboard.utils.model_utils import get_model_structure

blueprint = Blueprint("model_component", __name__, url_prefix='/models')

@blueprint.route("/<model_name>")
def show_model(model_name: str):
    model_data = get_model_structure(model_name)
    if not model_data:
        abort(404)

    return render_template(
        "model_component.html",
        model_name=model_data["model_name"],
        structure=model_data["structure"]
    )