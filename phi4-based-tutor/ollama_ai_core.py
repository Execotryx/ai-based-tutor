from ollama import chat, ChatResponse, Message
from ollama_ai_config import OllamaAIConfig
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

TAiResponse = TypeVar("TAiResponse", default=Any)

class HistoryManager:
    """
    Manage chat history and system behavior.
    """
    @property
    def system_behavior(self) -> str:
        """System instruction for the conversation."""
        return self.__system_behavior

    @property
    def chat_history(self) -> list[Message]:
        """Chat history for the conversation."""
        if not self.__chat_history[0].role == "system":
            self.__chat_history.insert(0, self.__create_message_with_role("system", self.system_behavior))
        return self.__chat_history

    @property
    def config(self) -> OllamaAIConfig:
        return self.__config

    def __init__(self, system_behavior: str, config: OllamaAIConfig) -> None:
        """Create a HistoryManager with the given system behavior.

        Parameters:
            system_behavior: System instruction string.
        """
        self.__system_behavior: str = system_behavior
        self.__chat_history: list[Message] = []
        self.__config: OllamaAIConfig = config

    def add_user_message(self, message: str) -> None:
        """Add a user message to the chat history."""
        self.__chat_history.append(self.__create_message_with_role("user", message))

    def add_assistant_message(self, message: str) -> None:
        """Add an assistant message to the chat history."""
        self.__chat_history.append(self.__create_message_with_role("assistant", message))

    def __create_message_with_role(self, role: str, content: str) -> Message:
        """Create a message with the given role and content."""
        return Message(role=role, content=content)


class AICore(ABC, Generic[TAiResponse]):

    @property
    def _config(self) -> OllamaAIConfig:
        return self.__config

    @property
    def _history_manager(self) -> HistoryManager:
        return self.__history_manager

    def __init__(self, system_behavior: str, config: OllamaAIConfig) -> None:
        self.__config: OllamaAIConfig = config
        self.__history_manager: HistoryManager = HistoryManager(system_behavior, self._config)

    def ask(self, request: str) -> TAiResponse:
        self._history_manager.add_user_message(request)
        response: ChatResponse = chat(model=self._config.model_id, messages=self._history_manager.chat_history)
        self._history_manager.add_assistant_message(
            response.message.content if response.message.content else "No response was received."
        )
        return self._process_response(response)

    @abstractmethod
    def _process_response(self, response: ChatResponse) -> TAiResponse:
        pass