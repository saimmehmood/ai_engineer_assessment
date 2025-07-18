
class BaseAPIException(Exception):
    message = None

    def __init__(self, message, *args: object) -> None:
        super().__init__(*args)
        self.messages = message

    def get_message(self):
        return self.messages
