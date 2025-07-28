from flask import Blueprint, render_template
from leaderboard.utils.export_utils import get_definitions, grouped_definitions

blueprint = Blueprint('index', __name__, url_prefix='/')

@blueprint.route('')
def index():
    definitions = get_definitions()
    grouped = grouped_definitions()
    return render_template("index.html", definitions=definitions, grouped=grouped)