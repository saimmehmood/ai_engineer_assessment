from app.commands.threads.process_chat_message import ProcessChatMessageCommand

from app.controllers.controller import Controller


class ThreadController(Controller):
    """
    A controller for threads.
    """
    def process_chat_message(self, chat_messages: list) -> list:
        return self.executor.execute_write(ProcessChatMessageCommand(chat_messages))
