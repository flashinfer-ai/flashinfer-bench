from flask import Blueprint, render_template, abort
from collections import defaultdict
from leaderboard.utils.export_utils import get_leaderboard, get_a_definition

blueprint = Blueprint("leaderboard", __name__)

@blueprint.route("/leaderboard/<definition_name>")
def show_leaderboard(definition_name: str):
    leaderboard_data = get_leaderboard()

    if definition_name not in leaderboard_data:
        abort(404)

    definition = get_a_definition(definition_name)
    entries = leaderboard_data[definition_name]

    entries_by_device_and_workload = defaultdict(lambda: defaultdict(list))
    for device, entries_for_device in entries.items():
        for entry in entries_for_device:
            workload = entry["workload"]
            entries_by_device_and_workload[device][workload].append(entry)

    return render_template(
        "leaderboard.html",
        definition=definition,
        entries_by_device_and_workload=dict(entries_by_device_and_workload)
    )