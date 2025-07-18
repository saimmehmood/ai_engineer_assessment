from app.errors.base import BaseAPIException


class ValidationException(BaseAPIException):
    def __init__(self, message, *args: object) -> None:
        super().__init__(message, *args)