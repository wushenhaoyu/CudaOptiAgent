from typing import Dict
from agent.llm import LLM
from agent.settings import Planner_settings



class Planner(LLM):
    def __init__(self, args: Dict):
        setting_id = args.model_choice
        setting = Planner_settings[setting_id]

        super().__init__(server_name=setting["server_name"], model=setting["model"], max_tokens=setting["max_tokens"], temperature=setting["temperature"], top_p=setting["top_p"])

    def plan(self, objective: str) -> str:
        return self.chat(objective)