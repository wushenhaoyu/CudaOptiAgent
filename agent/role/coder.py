from agent.llm import LLM


class Coder(LLM):
    def __init__(self):
        super().__init__()

    def code(self, prompt: str) -> str:
        return self.chat(prompt)