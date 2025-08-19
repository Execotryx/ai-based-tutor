"""AITutor: wrapper that uses AICore to provide tutoring behavior and dynamic role clarification."""

from openai.types.responses import Response
from ai_core import AICore
from ai_config import AIConfig
from ai_self_reference import AISelfReference
from typing import Any

class AITutor(AICore[str]):
	"""Tutor that adapts system behavior based on the user's question.

	Parameters:
		config: AIBrochureConfig used to configure the underlying AICore.
	"""

	@property
	def _clarified_system_behavior(self) -> str | None:
		"""Optional adjusted system instructions for a specific knowledge area."""
		return self.__clarified_system_behavior

	@property
	def _self_reference(self) -> AISelfReference:
		"""AISelfReference instance used to infer and refine tutor role."""
		return self.__self_reference

	def __init__(self, config: AIConfig) -> None:
		"""Create an AITutor with a base tutor system behavior.

		Parameters:
			config: Configuration for API keys and model selection.
		"""
		system_behavior: str = ("You are a helpful tutor, who excels at explaining complex concepts in simple terms."
								" You will provide detailed explanations and examples to help the user understand."
								" If it will be needed - use analogies.")
		super().__init__(config, system_behavior)
		self.__self_reference: AISelfReference = AISelfReference(config)
		self.__clarified_system_behavior: str | None = None

	def _form_call_configuration(self, request: str) -> dict[str, Any]:
		"""Build call configuration, applying clarified behavior if present.

		Parameters:
			request: Input text to send to the model.

		Returns:
			dict: Call configuration for the API.
		"""
		basic_call_configuration: dict[str, Any] = super()._form_call_configuration(request)
		if self._clarified_system_behavior:
			basic_call_configuration["instructions"] = self._clarified_system_behavior

		return basic_call_configuration

	def _process_response(self, response: Response) -> str:
		"""Extract text from the API Response.

		Parameters:
			response: Raw Response from the model.

		Returns:
			str: Output text.
		"""
		return response.output_text

	def explain_this(self, user_question: str) -> str:
		"""Explain a user question, adjusting the tutor role when helpful.

		Parameters:
			user_question: The user's question to explain.

		Returns:
			str: Tutor's explanation for the question.
		"""
		inferred_area_of_knowledge: str = self._self_reference.infer_area_of_knowledge(user_question)
		if inferred_area_of_knowledge:
			self.__clarified_system_behavior = self._self_reference.clarify_tutor_role(inferred_area_of_knowledge, self.history_manager.system_behavior)
		response: str = self.ask(user_question)
		return response

if __name__ == "__main__":
    # Example usage
    config = AIConfig()
    tutor = AITutor(config)
    
    user_question = "Can you explain the concept of recursion in programming?"
    response = tutor.explain_this(user_question)
    print(response)  # Outputs the explanation provided by the AI tutor