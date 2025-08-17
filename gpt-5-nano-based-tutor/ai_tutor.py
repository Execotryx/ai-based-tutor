from openai.types.responses import Response
from ai_core import AICore
from ai_brochure_config import AIBrochureConfig
from ai_self_reference import AISelfReference
from typing import Any

class AITutor(AICore[str]):

    @property
    def _clarified_system_behavior(self) -> str | None:
        """
        On a certain step the system behavior may be clarified or adjusted to assume a role of a tutor 
        who specializes exactly in that area, to which the user's question relates.
        With Responses API it is now possible to switch system behavior on the go while keeping history.
        """
        return self.__clarified_system_behavior

    @property
    def _self_reference(self) -> AISelfReference:
        return self.__self_reference

    def __init__(self, config: AIBrochureConfig) -> None:
        system_behavior: str = ("You are a helpful tutor, who excels at explaining complex concepts in simple terms."
                                "You will provide detailed explanations and examples to help the user understand."
                                "If it will be needed - use analogies.")
        super().__init__(config, system_behavior)
        self.__self_reference: AISelfReference = AISelfReference(config)
        self.__clarified_system_behavior: str | None = None

    def _form_call_configuration(self, request: str) -> dict[str, Any]:
        basic_call_configuration: dict[str, Any] = super()._form_call_configuration(request)
        if self._clarified_system_behavior:
            basic_call_configuration["instructions"] = self._clarified_system_behavior

        return basic_call_configuration

    def _process_response(self, response: Response) -> str:
        return response.output_text

    def explain_this(self, user_question: str) -> str:
        inferred_area_of_knowledge: str = self._self_reference.infer_area_of_knowledge(user_question)
        if inferred_area_of_knowledge:
            self.__clarified_system_behavior = self._self_reference.clarify_tutor_role(inferred_area_of_knowledge, self.history_manager.system_behavior)
        response: str = self.ask(user_question)
        return response

if __name__ == "__main__":
    # Example usage
    config = AIBrochureConfig()
    tutor = AITutor(config)
    
    user_question = "Can you explain the concept of recursion in programming?"
    response = tutor.explain_this(user_question)
    print(response)  # Outputs the explanation provided by the AI tutor