from pathlib import Path
from typing import Dict
from agent.llm import LLM
from utils.utils import write_file, read_file
from agent.template.analyzer import INIT_ANALYZER_TEMPLATE
from agent.template.gpu_info import GPU_SPEC_INFO
class Analyzer(LLM):
    def __init__(self, server_name: str = "deepseek",model: str = "deepseek-chat", max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 1.0):
        super().__init__(server_name=server_name, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def init_analyzer(self, root_dir: Path, args: Dict):
        gpu_info = GPU_SPEC_INFO.get(args["gpu_name"])
        prompt = INIT_ANALYZER_TEMPLATE.substitute(
            gpu_info = gpu_info,
            source_code = read_file(root_dir / "spec" / "ref.py")
        )
        return prompt, self.chat(prompt)
        