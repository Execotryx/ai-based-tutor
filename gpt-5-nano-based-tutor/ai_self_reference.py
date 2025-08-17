from ai_core import AICore
from ai_brochure_config import AIBrochureConfig
from openai.types.responses import Response
from typing import Any

class AISelfReference(AICore[str]):

    def __init__(self, config: AIBrochureConfig) -> None:
        system_behavior: str = ("You are a companion for AI Tutor."
                                "You will provide self-referential information that will help the AI tutor to assume a proper role in proper area of knowledge.\n"
                                "Be as concrete as possible in your answers."
                                "Always answer concisely.")
        super().__init__(config, system_behavior)

    def infer_area_of_knowledge(self, user_question: str) -> str:
        prompt: str = (f"User question: {user_question}\n"
                       f"Based on the question, the area of knowledge is:")
        area_of_knowledge: str = self.ask(prompt)
        return area_of_knowledge

    def clarify_tutor_role(self, area_of_knowledge: str, previous_system_behavior: str) -> str:
        prompt: str = (f"Area of knowledge: {area_of_knowledge}\n"
                       f"Previous tutor's role: {previous_system_behavior}\n"
                       f"Based on the area of knowledge and previous behavior, correct the tutor's role to reflect that tutor is an expert in the provided area of knowledge.")
        clarified_tutor_role: str = self.ask(prompt)
        return clarified_tutor_role

    def _form_call_configuration(self, request: str) -> dict[str, Any]:
        basic_call_configuration: dict[str, Any] = super()._form_call_configuration(request)
        basic_call_configuration["temperature"] = 0.2
        basic_call_configuration["reasoning"] = {"effort": "medium"}
        return basic_call_configuration

    def _process_response(self, response: Response) -> str:
        return response.output_text
