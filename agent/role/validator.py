from pathlib import Path
from typing import Dict
from agent.llm import LLM
from agent.settings import Validator_settings
from agent.template.validator import INIT_CPU_ERROR_VALIDATOR_TEMPLATE, INIT_CUDA_ERROR_VALIDATOR_TEMPLATE
from utils.utils import extract_error_report, write_file



class Validator(LLM):
    def __init__(self, args: Dict):
        setting_id = args.model_choice
        setting = Validator_settings[setting_id]

        super().__init__(server_name=setting["server_name"], model=setting["model"], max_tokens=setting["max_tokens"], temperature=setting["temperature"], top_p=setting["top_p"])

    def init_cuda_validator(self, current_dir: Path, source_code: str, entry_code: str, kernel_code: str, error_log: str):
        prompt = INIT_CUDA_ERROR_VALIDATOR_TEMPLATE.substitute(
            source_code=source_code,
            entry_code=entry_code,
            kernel_code=kernel_code,
            error_log=error_log
        )
        out = self.chat(prompt)
        error_report = extract_error_report(out)
        write_file(current_dir / "validator_io.txt", out)
        write_file(current_dir / "error_report.txt", str(error_report))  
        return error_report
    
    def init_cpu_validator(self, current_dir: Path, source_code: str, entry_code: str, cpu_code: str, error_log: str):
        prompt = INIT_CPU_ERROR_VALIDATOR_TEMPLATE.substitute(
            source_code=source_code,
            entry_code=entry_code,
            cpu_code=cpu_code,
            error_log=error_log
        )
        out = self.chat(prompt)
        error_report = extract_error_report(out)
        return error_report