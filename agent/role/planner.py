from agent.llm import LLM


class Planner(LLM):
    def __init__(self):
        super().__init__()

    def plan(self, objective: str) -> str:
        return self.chat(objective)