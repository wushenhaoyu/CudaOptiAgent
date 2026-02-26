from tqdm import tqdm

from pathlib import Path
from typing import Dict
from agent.llm import LLM
from agent.settings import Validator_settings
from agent.template.validator import INIT_CUDA_ERROR_VALIDATOR_TEMPLATE
from utils.utils import extract_error_report, write_file



class Validator(LLM):
    def __init__(self, args: Dict):

        super().__init__(server_name=args.server_name, model=args.model, max_tokens=16384, temperature=1.0, top_p=1.0)

    def generate_init_error_report(self, current_dir: Path, source_code: str, entry_code: str, kernel_code: str, error_log: str):
        tqdm.write("generate_init_error_report")
        prompt = INIT_CUDA_ERROR_VALIDATOR_TEMPLATE.substitute(
            source_code=source_code,
            entry_code=entry_code,
            kernel_code=kernel_code,
            error_log=error_log
        )
        while True:
            out = self.chat(prompt)
            error_report = extract_error_report(out)
            if error_report is not None:
                break
        write_file(current_dir / "validator_io.txt", out)
        write_file(current_dir / "error_report.txt", str(error_report))  
        return error_report
    
    

