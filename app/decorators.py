import traceback
from functools import wraps

from app import logger
from app.errors import ProcessingException, ValidationException
from app.utils.messages import Error
from app.utils.response import Response

from marshmallow import ValidationError


def handle_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ProcessingException as pe:
            logger.info(pe)
            return Response.make(pe.messages, Response.HTTP_BAD_REQUEST)
        except ValidationException as ve:
            logger.info(str(ve))
            return Response.make(ve.messages, Response.HTTP_BAD_REQUEST)
        except ValidationError as err:
            logger.info(err)
            return Response.make(err.messages, Response.HTTP_BAD_REQUEST)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"general exception {e}")
            return Response.make(Error.REQUEST_FAILED, Response.HTTP_ERROR)

    return wrapper