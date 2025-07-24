import os
from dotenv import load_dotenv
from flask import Flask
from flask_talisman import Talisman

from . import error, index, leaderboard, docs

def create_app(test_config=None):
    # Check if we're in development mode:
    is_dev = os.getenv('FLASK_DEBUG') == '1'
    if is_dev:
        load_dotenv()

    app = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(
        TALISMAN_FORCE_HTTPS=True,
    )

    if test_config is not None:
        app.config.from_mapping(test_config)

    # HTTPS and CSP headers
    Talisman(
        app,
        content_security_policy={
            'default-src': "'self'",
            'script-src': "'self' https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
        },
        force_https=app.config.get('TALISMAN_FORCE_HTTPS', True),
    )

    app.register_blueprint(index.blueprint)
    app.add_url_rule('/', endpoint='index')

    app.register_blueprint(leaderboard.blueprint)
    app.add_url_rule('/leaderboard/<definition>', endpoint='leaderboard')
    
    app.register_blueprint(docs.blueprint)
    app.add_url_rule('/docs', endpoint='docs')

    app.errorhandler(404)(error.page_not_found)
    app.errorhandler(500)(error.server_error)

    return app
