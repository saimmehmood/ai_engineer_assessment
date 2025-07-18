from flask import current_app, request


def is_internal_request():
    lowercase_headers = {k.lower(): v for k, v in request.headers.items()}
    api_key = lowercase_headers.get("x-api-key") or request.args.get("api_key")

    if not api_key:
        return False

    # Validate against the ENV variable
    if api_key == current_app.config["API_KEY"]:
        return True

    return False


def get_current_user():
    # Make all header keys lowercase
    lowercase_headers = {k.lower(): v for k, v in request.headers.items()}

    api_key = lowercase_headers.get("x-api-key") or request.args.get("api_key")

    bearer_token = lowercase_headers.get("authorization")
    if bearer_token:
        parts = bearer_token.split()
        if len(parts) == 2 and parts[0].lower() == 'bearer':
            token = parts[1]

    # Use either the API key or the bearer token
