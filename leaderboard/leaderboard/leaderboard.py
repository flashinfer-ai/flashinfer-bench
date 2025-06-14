from flask import Blueprint, render_template, abort
from datetime import datetime, timezone
import json
import os
from leaderboard.time import to_time_left

blueprint = Blueprint('leaderboard', __name__, url_prefix='/leaderboard')

@blueprint.route('/<int:leaderboard_id>')
def leaderboard(leaderboard_id: int):
    # Path to your precomputed JSON files
    path = f"leaderboard/static/leaderboard_{leaderboard_id}.json"
    
    if not os.path.exists(path):
        abort(404)

    with open(path) as f:
        data = json.load(f)

    leaderboard_data = data["leaderboard"]
    rankings_data = data["rankings"]

    name = leaderboard_data["name"]
    deadline = leaderboard_data.get("deadline", "")
    time_left = to_time_left(deadline) if deadline else None

    lang = leaderboard_data.get("lang", "Unknown")
    if lang == "py":
        lang = "Python"

    description = leaderboard_data.get("description", "")
    reference = leaderboard_data.get("reference", "")
    gpu_types = leaderboard_data.get("gpu_types", [])
    gpu_types.sort()

    # Compute ranks and score deltas
    rankings = {}
    for gpu_type, ranking_ in rankings_data.items():
        ranking = []
        prev_score = None

        for i, entry in enumerate(ranking_):
            entry["rank"] = i + 1
            entry["prev_score"] = entry["score"] - prev_score if prev_score is not None else None
            prev_score = entry["score"]
            ranking.append(entry)

        if ranking:
            rankings[gpu_type] = ranking

    return render_template(
        "leaderboard.html",
        name=name,
        deadline=deadline,
        time_left=time_left,
        lang=lang,
        gpu_types=gpu_types,
        description=description,
        reference=reference,
        rankings=rankings
    )
