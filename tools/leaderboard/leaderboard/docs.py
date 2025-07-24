from flask import Blueprint, render_template
import markdown
from markupsafe import Markup

blueprint = Blueprint('docs', __name__, url_prefix='/')

@blueprint.route("/docs")
def docs():
    with open("docs/index.md", "r") as f:
        content = markdown.markdown(f.read())
    return render_template("docs.html", content=Markup(content))