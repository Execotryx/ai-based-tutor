from ollama import ChatResponse
from ollama_ai_config import OllamaAIConfig
from ollama_ai_core import AICore
from ai_self_reference import SelfReferencingAI

class KnowledgeGuideAI(AICore[str]):

    @property
    def _self_reference(self) -> SelfReferencingAI:
        return self.__self_reference
    
    def _process_response(self, response: ChatResponse) -> str:
        return response.message.content if response.message.content else "No response was received."

    def explain_this(self, user_question: str) -> str:
        area: str = self._self_reference.infer_area_of_knowledge(user_question)
        clarified_role: str = self._self_reference.clarify_tutor_role(area, user_question)
        prompt: str = (f"As a {clarified_role} and an expert in the {area}, explain this question: {user_question}\nExplanation:")
        explanation: str = self.ask(prompt)
        return explanation

    def __init__(self, config: OllamaAIConfig) -> None:
        system_behavior = ("You are a helpful tutor, who excels at explaining complex concepts in simple terms."
								" You will provide detailed explanations and examples to help the user understand."
								" If it will be needed - use analogies.")
        super().__init__(system_behavior, config)
        self.__self_reference: SelfReferencingAI = SelfReferencingAI(config)

if __name__ == "__main__":
    config: OllamaAIConfig = OllamaAIConfig()
    ai: KnowledgeGuideAI = KnowledgeGuideAI(config)
    user_question: str = "Can you explain the concept of recursion in programming?"
    explanation: str = ai.explain_this(user_question)
    print(explanation)
