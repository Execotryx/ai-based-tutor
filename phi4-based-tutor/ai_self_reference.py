from ollama import ChatResponse
from ollama_ai_config import OllamaAIConfig
from ollama_ai_core import AICore

class SelfReferencingAI(AICore[str]):

    def __init__(self, config: OllamaAIConfig) -> None:
        system_behavior: str = ("You are a companion for AI Tutor."
								"You will provide self-referential information that will help the AI tutor to assume a proper role in proper area of knowledge.\n"
								"Be as concrete as possible in your answers."
								"Always answer concisely.")
        super().__init__(system_behavior, config)
    
    def _process_response(self, response: ChatResponse) -> str:
        return response.message.content.strip(" .,") if response.message.content else "No response was received."

    def infer_area_of_knowledge(self, user_input: str) -> str:
        prompt: str = (f"User question: {user_input}\n"
                       f"Infer the area of knowledge required for AI Tutor to provide the best possible answer.\n"
                       "Respond ONLY with area of knowledge.\n"
                       "Area of knowledge:")
        area: str = self.ask(prompt)
        return area

    def clarify_tutor_role(self, area_of_knowledge: str, user_question: str) -> str:
        prompt: str = (f"User question: {user_question}\n"
                       f"Area of knowledge: {area_of_knowledge}\n"
                       "Basing on area and question, provided above - infer the role,"
                       "directly connected to the area of knowledge, that will allow the AI Tutor "
                       "to provide the best possible answer on the question.\n"
                       "Respond only with inferred role.\n"
                       "Inferred role:")
        role: str = self.ask(prompt)
        return role