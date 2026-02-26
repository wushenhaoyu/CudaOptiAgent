from typing import Dict
from agent.llm import LLM
from agent.settings import Planner_settings



class Planner(LLM):
    def __init__(self, args: Dict):

        super().__init__(server_name=args.server_name, model=args.model, max_tokens=16384, temperature=1.0, top_p=1.0)
    def plan(self, objective: str) -> str:
        return self.chat(objective)