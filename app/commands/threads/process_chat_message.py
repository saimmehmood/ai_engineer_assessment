from typing import Dict, List
from flask import current_app, g

from app import logger
from app.core.commands import ReadCommand
from app.errors import ValidationException
from app.services.llm.prompts.chat_prompt import chat_prompt
from app.services.llm.session import LLMSession
from app.services.llm.structured_outputs import text_to_sql
from app.services.llm.tools.text_to_sql import text_to_sql as text_to_sql_tool
from app.services.llm.response_quality import ResponseQualityService
from app.utils.formatters import get_timestamp

from langfuse.decorators import observe
from openai import BadRequestError
from vaul import Toolkit
from uuid import uuid4

import json


class ProcessChatMessageCommand(ReadCommand):
    """
    Process a chat message.
    """
    def __init__(self, chat_messages: List[Dict[str, str]]) -> None:
        self.chat_messages = chat_messages
        self.llm_session = LLMSession(
            chat_model=current_app.config.get("CHAT_MODEL"),
            embedding_model=current_app.config.get("EMBEDDING_MODEL"),
        )
        self.toolkit = Toolkit()
        self.toolkit.add_tools(*[text_to_sql_tool])
        self.response_quality_service = ResponseQualityService()

    def validate(self) -> None:
        """
        Validate the command.
        """
        if not self.chat_messages:
            raise ValidationException("Chat messages are required.")
        
        return True
    
    def execute(self) -> None:
        """
        Execute the command.
        """
        logger.debug(
            f'Command {self.__class__.__name__} started with {len(self.chat_messages)} messages.'
        )

        self.validate()

        chat_kwargs = {
            "messages": self.prepare_chat_messages(),
            "tools": self.toolkit.tool_schemas(),
        }

        try:
            response = self.llm_session.chat(**chat_kwargs)
        except BadRequestError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to fetch chat response: {e}")
            raise ValidationException("Error in fetching chat response.")

        tool_messages = []
        sql_result = None

        response_message_config = {
            "role": "assistant",
            "content": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
        }

        if response.choices[0].finish_reason == "tool_calls":
            tool_calls = response.choices[0].message.tool_calls

            response_message_config["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in tool_calls
            ]

            response_message = self.format_message(**response_message_config)

            for tool_call in tool_calls:
                tool_run = self.execute_tool_call(tool_call)
                if tool_call.function.name == "text_to_sql":
                    sql_result = tool_run
                tool_messages.append(
                    self.format_message(
                        role="tool",
                        tool_call_id=tool_call.id,
                        content=tool_run if isinstance(tool_run, str) else json.dumps(tool_run),
                    )
                )
        else:
            response_message = self.format_message(**response_message_config)

        # Apply response quality improvement
        user_question = self.get_user_question()
        improved_content = self.response_quality_service.process_response_with_quality_improvement(
            user_question=user_question,
            response=response_message["content"],
            sql_result=sql_result,
            enable_improvement=current_app.config.get("ENABLE_RESPONSE_QUALITY_IMPROVEMENT", True)
        )
        
        if improved_content != response_message["content"]:
            response_message["content"] = improved_content
            logger.info("Response quality improvement applied")

        # Add the messages as the last elements of the list
        self.chat_messages.append(response_message)
        self.chat_messages.extend(tool_messages)

        return self.chat_messages
    
    def execute_stream(self):
        """
        Execute the command with streaming support.
        """
        logger.debug(
            f'Command {self.__class__.__name__} started with streaming for {len(self.chat_messages)} messages.'
        )

        self.validate()

        chat_kwargs = {
            "messages": self.prepare_chat_messages(),
            "tools": self.toolkit.tool_schemas(),
            "stream": True,
        }

        try:
            response_stream = self.llm_session.chat_stream(**chat_kwargs)
            
            for chunk in response_stream:
                print("Streaming chunk from LLM:", chunk)
                if chunk.choices[0].delta.content:
                    print("Yielding LLM content chunk:", chunk.choices[0].delta.content)
                    yield {
                        "type": "content",
                        "content": chunk.choices[0].delta.content,
                        "finish_reason": None
                    }
                elif chunk.choices[0].finish_reason:
                    if chunk.choices[0].finish_reason == "tool_calls":
                        tool_calls = chunk.choices[0].delta.tool_calls
                        if tool_calls:
                            for tool_call in tool_calls:
                                if tool_call.function:
                                    print("Executing tool call:", tool_call.function.name)
                                    try:
                                        tool_run = self.execute_tool_call(tool_call)
                                        print("Tool run result:", tool_run)
                                        if isinstance(tool_run, str):
                                            print("Yielding tool result as content:", tool_run)
                                            yield {
                                                "type": "content",
                                                "content": tool_run,
                                                "finish_reason": None
                                            }
                                        else:
                                            print("Yielding tool result as tool_result:", tool_run)
                                            yield {
                                                "type": "tool_result",
                                                "content": json.dumps(tool_run),
                                                "finish_reason": None
                                            }
                                    except Exception as e:
                                        logger.error(f"Error during tool call execution: {e}")
                                        yield {
                                            "type": "error",
                                            "content": f"Tool call error: {str(e)}",
                                            "finish_reason": "error"
                                        }
                    else:
                        print("Yielding finish chunk.")
                        yield {
                            "type": "finish",
                            "content": "",
                            "finish_reason": chunk.choices[0].finish_reason
                        }
                        
        except BadRequestError as e:
            logger.error(f"BadRequestError in streaming: {e}")
            yield {
                "type": "error",
                "content": str(e),
                "finish_reason": "error"
            }
        except Exception as e:
            logger.error(f"Failed to fetch streaming chat response: {e}")
            yield {
                "type": "error",
                "content": "Error in fetching chat response.",
                "finish_reason": "error"
            }

    @observe()
    def prepare_chat_messages(self) -> list:
        trimmed_messages = self.llm_session.trim_message_history(
            messages=self.chat_messages,
        )

        system_prompt = chat_prompt()

        trimmed_messages = system_prompt + trimmed_messages

        return trimmed_messages

    @observe()
    def format_message(self, role: str, content: str, **kwargs) -> dict:
        return {
            "id": str(uuid4()),
            "role": role,
            "content": content,
            "timestamp": (get_timestamp(with_nanoseconds=True),),
            **kwargs,
        }

    @observe()
    def execute_tool_call(self, tool_call: dict) -> dict:
        return self.toolkit.run_tool(
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
    
    def get_user_question(self) -> str:
        """
        Extract the user's question from the chat messages.
        """
        for message in reversed(self.chat_messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""
