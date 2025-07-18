import logging.config
import os
import sys
import warnings

from flask import Flask

from app.extensions import cors, ma, mail
from app.log import LOG_CONFIG
from config import config

logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)

log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

logger.level = log_levels.get(os.environ.get("LOG_LEVEL", "INFO"))


def silence_warnings(config_name):
    if config_name == "PROD" and not sys.warnoptions:
        warnings.simplefilter("ignore")


def create_app(config_name = "DEV"):
    from app.routes import routes

    app = Flask(__name__)

    # Initialize configuration
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    silence_warnings(config_name)

    # Initialize routes
    routes.init_routes(app)

    # Initialize extensions
    cors.init_app(app)
    ma.init_app(app)
    mail.init_app(app)

    return app