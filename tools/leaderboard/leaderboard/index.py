from flask import Blueprint, render_template
from leaderboard.utils.export_utils import get_definitions

blueprint = Blueprint('index', __name__, url_prefix='/')

@blueprint.route('')
def index():
    definitions = get_definitions()
    print(f"[Info] Available definitions: {definitions}")
    return render_template("index.html", definitions=definitions)