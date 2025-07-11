import json
from flask import Blueprint, render_template
from datetime import datetime, timezone

blueprint = Blueprint('index', __name__, url_prefix='/')

@blueprint.route('')
def index():
    # Load pre-converted leaderboard data
    with open("leaderboard/static/leaderboard.json") as f:
        leaderboards = json.load(f)

    return render_template(
        'index.html',
        leaderboards=leaderboards,
        now=datetime.now(timezone.utc)
    )
