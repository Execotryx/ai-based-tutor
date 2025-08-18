"""AISelfReference: helper AICore subclass that infers knowledge areas and refines tutor role."""
from ai_core import AICore
from ai_brochure_config import AIBrochureConfig
from openai.types.responses import Response
from typing import Any

class AISelfReference(AICore[str]):
	"""Component that produces concise self-referential outputs for AITutor.

	Parameters:
		config: AIBrochureConfig used to configure the underlying AICore.
	"""

	def __init__(self, config: AIBrochureConfig) -> None:
		"""Initialize with a compact system behavior for self-reference.

		Parameters:
			config: Configuration for API keys and model selection.
		"""
		system_behavior: str = ("You are a companion for AI Tutor."
								"You will provide self-referential information that will help the AI tutor to assume a proper role in proper area of knowledge.\n"
								"Be as concrete as possible in your answers."
								"Always answer concisely.")
		super().__init__(config, system_behavior)

	def infer_area_of_knowledge(self, user_question: str) -> str:
		"""Infer the area of knowledge from a user question.

		Parameters:
			user_question: The user's question.

		Returns:
			str: Inferred area of knowledge (as plain text).
		"""
		prompt: str = (f"User question: {user_question}\n"
						"Based on the question, infer the area of knowledge.\n"
						"Respond with the inferred area of knowledge and nothing else.")
		area_of_knowledge: str = self.ask(prompt)
		return area_of_knowledge

	def clarify_tutor_role(self, area_of_knowledge: str, previous_system_behavior: str) -> str:
		"""Return a corrected tutor role tailored to the given area.

		Parameters:
			area_of_knowledge: Inferred area to tailor the role for.
			previous_system_behavior: The prior tutor system instructions.

		Returns:
			str: Corrected tutor role text.
		"""
		prompt: str = (f"Area of knowledge: {area_of_knowledge}\n"
						f"Previous tutor's role: {previous_system_behavior}\n"
						"Based on the area of knowledge and previous behavior, correct the tutor's role to reflect that tutor is an expert in the provided area of knowledge.\n"
						"Respond ONLY with the corrected tutor's role, nothing else.")
		clarified_tutor_role: str = self.ask(prompt)
		return clarified_tutor_role

	def _form_call_configuration(self, request: str) -> dict[str, Any]:
		"""Extend call configuration with a reasoning hint.

		Parameters:
			request: Input text to send to the model.

		Returns:
			dict: Call configuration for the API.
		"""
		basic_call_configuration: dict[str, Any] = super()._form_call_configuration(request)
		basic_call_configuration["reasoning"] = {"effort": "medium"}
		return basic_call_configuration

	def _process_response(self, response: Response) -> str:
		"""Extract output text from the API Response.

		Parameters:
			response: Raw Response from the model.

		Returns:
			str: Output text.
		"""
		return response.output_text
