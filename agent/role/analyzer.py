from pathlib import Path
from typing import Dict
from agent.llm import LLM
from agent.settings import Analyzer_settings
from utils.utils import write_file, read_file
from agent.template.analyzer import INIT_ANALYZER_TEMPLATE, INIT_REPAIR_ANALYZER_TEMPLATE
from agent.template.gpu_info import GPU_SPEC_INFO



class Analyzer(LLM):
    def __init__(self, args: Dict):
        setting_id = args.model_choice
        setting = Analyzer_settings[setting_id]

        super().__init__(server_name=setting["server_name"], model=setting["model"], max_tokens=setting["max_tokens"], temperature=setting["temperature"], top_p=setting["top_p"])

    def init_repair_analyzer(self, root_dir: Path, current_dir: Path, error_report: str, args: Dict):
        #gpu_info = GPU_SPEC_INFO.get(args.gpu_name)
        prompt = INIT_REPAIR_ANALYZER_TEMPLATE.substitute(
            source_code=read_file(root_dir / "spec" / "ref.py"),
            kernel_code=read_file(root_dir / "spec" / "kernel.cu"),
            error_report=error_report,
        )

        out = self.chat(prompt)
        write_file(current_dir / "analyzer_io.txt", f"Input Prompt:\n{prompt}\n\nOutput Response:\n{out}")
        return out

    
        