from agent.llm import LLM


class Planner(LLM):
    def __init__(self, server_name: str = "deepseek",model: str = "deepseek-chat", max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 1.0):
        super().__init__(server_name=server_name, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    def plan(self, objective: str) -> str:
        return self.chat(objective)