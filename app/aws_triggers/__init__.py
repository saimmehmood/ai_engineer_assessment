from app import create_app, logger

import json
import os


def sample_handler(event, context):
    app = create_app(os.getenv("FLASK_CONFIG") or "DEV")

    try:
        event_body = json.loads(event["Records"][0]["body"])
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f'Error parsing event: {e}')
        return False

    with app.app_context():
        logger.debug("Sample handler executed successfully.")

    return True
