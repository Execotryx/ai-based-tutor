from ollama import chat, ChatResponse
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
    def chat_history(self) -> list[dict[str, str]]:
        """Chat history for the conversation."""
        if not self.__chat_history[0].get("role") == "system":
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
        self.__chat_history: list[dict[str, str]] = []
        self.__config: OllamaAIConfig = config

    def add_user_message(self, message: str) -> None:
        """Add a user message to the chat history."""
        self.__chat_history.append(self.__create_message_with_role("user", message))

    def add_assistant_message(self, message: str) -> None:
        """Add an assistant message to the chat history."""
        self.__chat_history.append(self.__create_message_with_role("assistant", message))

    def __create_message_with_role(self, role: str, content: str) -> dict[str, str]:
        """Create a message with the given role and content."""
        return {"role": role, "content": content}

