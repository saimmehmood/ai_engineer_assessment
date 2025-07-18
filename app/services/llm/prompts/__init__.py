from flask import current_app
from functools import wraps
from langfuse import Langfuse

from app.utils import logger


def prompt(name=None):
    """
    Decorator that attempts to fetch and compile a prompt from Langfuse using either the provided name
    or the function name. If Langfuse is not configured or an error occurs, it falls back to the 
    function's default prompt.
    
    Args:
        name (str, optional): The name of the prompt to fetch from Langfuse. If not provided,
                            defaults to the function name.
    
    Usage:
        @prompt()  # Uses function name
        def system_message(**kwargs):
            return "default prompt text ..."

        @prompt(name="custom_prompt")  # Uses custom name
        def system_message(**kwargs):
            return "default prompt text ..."
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            prompt_name = name or func.__name__
            # Check if Langfuse is configured
            if all([
                current_app.config['LANGFUSE_PUBLIC_KEY'],
                current_app.config['LANGFUSE_SECRET_KEY'],
                current_app.config['LANGFUSE_HOST']
            ]):
                try:
                    langfuse_prompt = Langfuse().get_prompt(prompt_name, type="chat")
                    return langfuse_prompt.compile(**kwargs, fallback=func(*args, **kwargs))
                except Exception as e:
                    return func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator

