from typing import List, Dict, Any, Optional
from flask import current_app

from app import logger

from langfuse.decorators import langfuse_context
from litellm import completion, embedding
from vaul import StructuredOutput

import tiktoken


class LLMSession:
    """
    A session class for interacting with Litellm and any underlying models.
    """

    AVAILABLE_CHAT_MODELS = [
        {
            "name": "gpt-4o-mini",
            "description": "The GPT-4o Mini model.",
            "token_limit": 128_000,
        },
        {
            "name": "gpt-4o",
            "description": "The GPT-4o model.",
            "token_limit": 128_000,
        },
        {
            "name": "o3-mini",
            "description": "The O3 Mini model.",
            "token_limit": 200_000,
        },
        {
            "name": "o1",
            "description": "The O1 model",
            "token_limit": 200_000,
        },
        {
            "name": "o1-mini",
            "description": "The O1 Mini model.",
            "token_limit": 200_000,
        },
        {
            "name": "gpt-4.5-preview",
            "description": "The GPT-4.5 Preview model.",
            "token_limit": 128_000,
        },
        {
            "name": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "description": "The Claude 3.7 Sonnet model.",
            "token_limit": 200_000,
        },
        {
            "name": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "description": "The Claude 3.5v2 Sonnet model.",
            "token_limit": 200_000,
        },
        {
            "name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "description": "The Claude 3 Sonnet model.",
            "token_limit": 28_000,
        },
        {
            "name": "anthropic.claude-3-haiku-20240307-v1:0",
            "description": "The Claude 3 Haiku model.",
            "token_limit": 48_000,
        },
    ]

    AVAILABLE_EMBEDDING_MODELS = [
        {
            "name": "text-embedding-3-small",
            "description": "The OpenAI Embedding 3 Small model.",
            "dimensions": 1536,
        },
        {
            "name": "text-embedding-3-large",
            "description": "The OpenAI Embedding 3 Large model.",
            "dimensions": 3072,
        },
        {
            "name": "cohere.embed-english-v3",
            "description": "The Embed English v3 model from Cohere.",
            "dimensions": 1024,
        },
        {
            "name": "amazon.titan-embed-text-v2:0",
            "description": "The Titan Embed Text v2 model from Amazon.",
            "dimensions": 1024,
        },
    ]

    DEFAULT_CHAT_MODEL = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"

    def __init__(
        self,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        """
        Initialize a new LLMSession instance.

        :param chat_model: The chat model name.
        :param embedding_model: The embedding model name.
        """
        self.chat_model = self.validate_chat_model(chat_model)
        self.embedding_model = self.validate_embedding_model(embedding_model)
        self.knn_embedding_dimensions = self._get_embedding_model_dimensions(
            self.embedding_model
        )

        # Try to get expected dimension from app config, fallback to embedding model dimensions
        try:
            expected_dim = current_app.config.get("KNN_EMBEDDING_DIMENSION")
            if expected_dim and self.knn_embedding_dimensions != expected_dim:
                logger.warning(
                    f"Class-level knn_embedding_dimensions ({self.knn_embedding_dimensions}) does not match "
                    f"config's KNN_EMBEDDING_DIMENSION ({expected_dim}). This mismatch may lead to errors during KNN searches."
                )
        except RuntimeError:
            # Outside app context, use embedding model dimensions
            logger.debug("Outside Flask app context, using embedding model dimensions")
            pass

    @classmethod
    def _find_model(
        cls, models: List[Dict[str, Any]], model_name: str, model_type: str
    ) -> Dict[str, Any]:
        """
        Helper method to find a model in the given list by name.

        :param models: List of model dictionaries.
        :param model_name: The model name to find.
        :param model_type: The type of model (used in error messages).
        :return: The model dictionary.
        :raises ValueError: If model is not found.
        """
        for model in models:
            if model["name"] == model_name:
                return model
        raise ValueError(
            f"Invalid {model_type} model: {model_name}. Must be one of {[m['name'] for m in models]}"
        )

    def validate_chat_model(self, chat_model: str) -> str:
        """
        Validate and return the chat model name.

        :param chat_model: The chat model to validate.
        :return: Validated chat model name.
        """
        return self._find_model(self.AVAILABLE_CHAT_MODELS, chat_model, "chat")["name"]

    def validate_embedding_model(self, embedding_model: str) -> str:
        """
        Validate and return the embedding model name.

        :param embedding_model: The embedding model to validate.
        :return: Validated embedding model name.
        """
        return self._find_model(
            self.AVAILABLE_EMBEDDING_MODELS, embedding_model, "embedding"
        )["name"]

    def _get_chat_model_token_limit(self, model_name: str) -> int:
        """
        Get token limit for the specified chat model.

        :param model_name: Chat model name.
        :return: Token limit.
        """
        return self._find_model(self.AVAILABLE_CHAT_MODELS, model_name, "chat")[
            "token_limit"
        ]

    def _get_embedding_model_dimensions(self, model_name: str) -> int:
        """
        Get dimensions for the specified embedding model.

        :param model_name: Embedding model name.
        :return: Dimensions.
        """
        return self._find_model(
            self.AVAILABLE_EMBEDDING_MODELS, model_name, "embedding"
        )["dimensions"]

    def _get_metadata(self) -> Dict[str, str]:
        """
        Helper method to obtain metadata from langfuse context.

        :return: Metadata dictionary.
        """
        return {
            "existing_trace_id": langfuse_context.get_current_trace_id(),
            "parent_observation_id": langfuse_context.get_current_observation_id(),
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Any]] = None,
        **kwargs,
    ) -> Any:
        """
        Send messages to the chat model and return the response.

        :param messages: List of message dictionaries.
        :param tools: Optional list of tool dictionaries.
        :param kwargs: Additional parameters for the chat call.
        :return: Chat model response.
        """
        chat_config: Dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            **kwargs,
        }
        if tools:
            chat_config["tools"] = tools

        guardrail_id = current_app.config.get("BEDROCK_GUARDRAILS_ID")
        if guardrail_id:
            chat_config["guardrailConfig"] = {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": "DRAFT",
                "trace": "enabled",
            }

        chat_config.setdefault("metadata", {}).update(self._get_metadata())

        try:
            response = completion(**chat_config)
            logger.debug(f"Chat response: {response.to_dict()}")
            return response
        except Exception as e:
            logger.error(f"Error sending messages to chat model: {e}")
            raise

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Any]] = None,
        **kwargs,
    ):
        """
        Send messages to the chat model and return a streaming response.

        :param messages: List of message dictionaries.
        :param tools: Optional list of tool dictionaries.
        :param kwargs: Additional parameters for the chat call.
        :yield: Streaming chat model response chunks.
        """
        chat_config: Dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        if tools:
            chat_config["tools"] = tools

        guardrail_id = current_app.config.get("BEDROCK_GUARDRAILS_ID")
        if guardrail_id:
            chat_config["guardrailConfig"] = {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": "DRAFT",
                "trace": "enabled",
            }

        chat_config.setdefault("metadata", {}).update(self._get_metadata())

        try:
            response_stream = completion(**chat_config)
            for chunk in response_stream:
                yield chunk
        except Exception as e:
            logger.error(f"Error sending messages to chat model (streaming): {e}")
            raise

    def get_structured_output(
        self,
        messages: List[Dict[str, str]],
        structured_output: StructuredOutput,
    ) -> StructuredOutput:
        """
        Retrieve structured output from the chat model.

        :param messages: List of message dictionaries.
        :param structured_output: StructuredOutput instance to parse the output.
        :return: Parsed StructuredOutput.
        :raises ValueError: If messages are empty or an error occurs.
        """
        if not messages:
            logger.error("No messages provided to send to the API.")
            raise ValueError("Messages list is empty.")

        try:
            response = completion(
                model=self.chat_model,
                messages=messages,
                tools=[
                    {"type": "function", "function": structured_output.tool_call_schema}
                ],
                tool_choice={
                    "type": "function",
                    "function": {"name": structured_output.tool_call_schema["name"]},
                },
                metadata=self._get_metadata(),
            )
            logger.debug("API response received successfully.")
        except Exception as e:
            logger.exception("Error during API call.")
            raise ValueError("Error in fetching API response.") from e

        try:
            result = structured_output.from_response(response)
            logger.debug("Structured output parsed successfully.")
            return result
        except Exception as e:
            logger.exception("Error parsing structured output.")
            raise ValueError("Error parsing structured output.") from e

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Count tokens in the provided text.

        :param text: Input text.
        :return: Token count.
        """
        tokenizer = tiktoken.get_encoding("p50k_base")
        return len(tokenizer.encode(text))

    def validate_token_length(self, text: str, token_limit: int) -> None:
        """
        Ensure text token count does not exceed token_limit.

        :param text: Input text.
        :param token_limit: Maximum allowed tokens.
        :raises ValueError: If text is empty or too long.
        """
        if not isinstance(text, str) or not text:
            raise ValueError("Text must be a non-empty string.")
        if self.count_tokens(text) > token_limit:
            raise ValueError(f"Text exceeds max token length of {token_limit}.")

    def trim_message_history(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Trim message history to fit within the chat model's token limit.

        :param messages: List of message dictionaries in ascending order.
        :return: Trimmed list of messages.
        """
        tokenizer = tiktoken.get_encoding("p50k_base")
        token_limit = self._get_chat_model_token_limit(self.chat_model)

        # Tokenize all messages
        tokenized_messages = []
        for msg in messages:
            content = msg.get("content", "")
            tokens = tokenizer.encode(content, disallowed_special=()) if content else []
            tokenized_messages.append((msg, tokens))

        # Calculate total token length
        total_tokens = sum(len(tokens) for _, tokens in tokenized_messages)

        # Trim messages from the beginning until we fit within the token limit
        while total_tokens > token_limit and tokenized_messages:
            _, removed_tokens = tokenized_messages.pop(0)
            total_tokens -= len(removed_tokens)

        # Reconstruct the trimmed message history
        trimmed_message_history = []
        for message, tokens in tokenized_messages:
            trimmed_message = {
                "role": message["role"],
                "content": tokenizer.decode(tokens),
            }
            # Only add tool_calls if non-empty
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                trimmed_message["tool_calls"] = tool_calls
            # Only add tool_call_id if non-empty
            tool_call_id = message.get("tool_call_id")
            if tool_call_id:
                trimmed_message["tool_call_id"] = tool_call_id

            trimmed_message_history.append(trimmed_message)

        return trimmed_message_history

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.

        :param text: Input text.
        :return: List of floats representing the embedding.
        :raises ValueError: If embedding generation fails.
        """
        try:
            response = embedding(
                model=self.embedding_model, input=text, metadata=self._get_metadata()
            ).to_dict()
            embeddings = response.get("data", [])
            if embeddings:
                embedding_vector = embeddings[0].get(
                    "embedding", [0.0] * self.knn_embedding_dimensions
                )
            else:
                embedding_vector = [0.0] * self.knn_embedding_dimensions

            logger.debug(f"Generated embedding for text: {text}")
            return embedding_vector

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise ValueError("Error generating embeddings.") from e
