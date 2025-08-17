import openai
from abc import ABC, abstractmethod
from ai_brochure_config import AIBrochureConfig
from typing import Any, Generic, TypeVar
from openai.types.responses import Response

TAiResponse = TypeVar('TAiResponse', default=Any)

class HistoryManager:
	"""
	Manage chat history and system behavior.
	"""
	@property
	def system_behavior(self) -> str:
		"""System instruction for the conversation."""
		return self.__system_behavior

	@property
	def last_assistant_response_id(self) -> str | None:
		"""ID of the last assistant response (or None)."""
		return self.__last_assistant_response_id

	@last_assistant_response_id.setter
	def last_assistant_response_id(self, response_id: str | None) -> None:
		"""Set the last assistant response ID.

		Parameters:
			response_id: The assistant response ID to store, or None.
		"""
		self.__last_assistant_response_id = response_id

	def __init__(self, system_behavior: str) -> None:
		"""Create a HistoryManager with the given system behavior.

		Parameters:
			system_behavior: System instruction string.
		"""
		self.__system_behavior: str = system_behavior
		self.__last_assistant_response_id: str | None = None


class AICore(ABC, Generic[TAiResponse]):
	"""Base class for AI calls and history handling."""
	@property
	def config(self) -> AIBrochureConfig:
		"""Get the current AIBrochureConfig."""
		return self.__config

	@config.setter
	def config(self, config: AIBrochureConfig | None) -> None:
		"""Set configuration, or reset to default if None.

		Parameters:
			config: New AIBrochureConfig, or None to reset to default.
		"""
		if config is None:
			self.__config = AIBrochureConfig()
		else:
			self.__config = config

	@property
	def _ai_api(self) -> openai.OpenAI:
		"""Lazily initialize and return the OpenAI client.

		Raises:
			ValueError: If configuration is not set.
		"""
		if self.__ai_api is None:
			if self.config is None:
				raise ValueError("Configuration must be set before accessing AI API")
			self.__ai_api = openai.OpenAI(api_key=self.config.openai_api_key)
		return self.__ai_api

	@property
	def history_manager(self) -> HistoryManager:
		"""Return the HistoryManager instance."""
		return self.__history_manager

	def __init__(self, config: AIBrochureConfig, system_behavior: str) -> None:
		"""Initialize with config and system behavior.

		Parameters:
			config: AIBrochureConfig for this instance.
			system_behavior: System behavior string (instructions).
		"""
		# Initialize all instance-level attributes here
		self.__config: AIBrochureConfig = config
		self.__history_manager: HistoryManager = HistoryManager(system_behavior)
		self.__ai_api: openai.OpenAI | None = None

		if __debug__:
			# Sanity check: confirm attributes are initialized
			assert hasattr(self, "_AICore__config")
			assert hasattr(self, "_AICore__history_manager")
			assert hasattr(self, "_AICore__ai_api")

	def ask(self, request: str) -> TAiResponse:
		"""Send a request to the model and return the processed response.

		Parameters:
			request: Input text to send to the model.

		Returns:
			Processed AI response.
		"""
		call_configuration: dict = self.form_call_configuration(request)
		response: Response = self._ai_api.responses.create(
			**call_configuration
		)
		self.history_manager.last_assistant_response_id = response.id
		return self.process_response(response)

	@abstractmethod
	def form_call_configuration(self, request: str) -> dict:
		"""Build the call configuration dict for the API call.

		Parameters:
			request: The input text to include in the call configuration.

		Returns:
			dict: Configuration for the API call.
		"""
		call_configuration: dict = {
			"model": self.config.model_name,
			"instructions": self.history_manager.system_behavior,
			"input": request
		}
		if self.history_manager.last_assistant_response_id:
			call_configuration["previous_response_id"] = self.history_manager.last_assistant_response_id
		return call_configuration


	@abstractmethod
	def process_response(self, response: Response) -> TAiResponse:
		"""Convert the model Response into TAiResponse.

		Parameters:
			response: Raw Response object from the API.

		Returns:
			TAiResponse: Processed response.
		"""
		pass