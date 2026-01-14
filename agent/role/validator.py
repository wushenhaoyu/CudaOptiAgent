from agent.llm import LLM


class Validator(LLM):
    def __init__(self):
        super().__init__()

    def validate(self, code: str) -> str:
        return self.chat(code)