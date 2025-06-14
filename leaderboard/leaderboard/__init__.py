import os
from dotenv import load_dotenv
from flask import Flask
from flask_login import LoginManager
from flask_session import Session
from flask_talisman import Talisman
from . import color, error, index, leaderboard, score, time
from .redis_connection import create_redis_connection

def create_app(test_config=None):
    # Check if we're in development mode:
    is_dev = os.getenv('FLASK_DEBUG') == '1'
    if is_dev:
        load_dotenv()

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        TALISMAN_FORCE_HTTPS=True,
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',
        SESSION_PERMANENT=True,
        PERMANENT_SESSION_LIFETIME=1209600, # 14 days
        SESSION_TYPE='redis',

        # Heroku Redis uses self-signed certificates:
        # https://devcenter.heroku.com/articles/heroku-redis#security-and-compliance
        # In Heroku we use the config key REDIS_SSL_CERT_REQS to have redis-py
        # accept self-signed certificates.
        SESSION_REDIS=create_redis_connection(
            cert_reqs=os.getenv('REDIS_SSL_CERT_REQS')),

    )

    if test_config is not None:
        app.config.from_mapping(test_config)

    Session(app)

    login_manager = LoginManager()

    @login_manager.user_loader
    def load_user(user_id):
        return auth.User(user_id) if user_id else None

    login_manager.init_app(app)

    Talisman(
        app,
        content_security_policy={
            'default-src': "'self'",
            'script-src': "'self' https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
        },
        force_https=app.config.get('TALISMAN_FORCE_HTTPS', True))

    try:
        os.makedirs(app.instance_path)
    except OSError:
        if not os.path.exists(app.instance_path):
            raise

    app.add_template_filter(color.to_color, 'to_color')
    app.add_template_filter(score.format_score, 'format_score')
    app.add_template_filter(time.to_time_left, 'to_time_left')
    app.add_template_filter(time.format_datetime, 'format_datetime')

    app.register_blueprint(index.blueprint)
    app.add_url_rule('/', endpoint='index')

    app.register_blueprint(leaderboard.blueprint)
    app.add_url_rule('/leaderboard/<int:id>', endpoint='leaderboard')

    app.errorhandler(401)(error.unauthorized)
    app.errorhandler(404)(error.page_not_found)
    app.errorhandler(500)(error.server_error)

    return app
