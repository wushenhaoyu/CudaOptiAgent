from agent.llm import LLM


class Analyzer(LLM):
    def __init__(self):
        super().__init__()

    def analyze(self, code: str) -> str:
        return self.chat(code)