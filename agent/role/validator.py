import json

from tqdm import tqdm
from pathlib import Path
from typing import Dict
from agent.llm import LLM
from agent.settings import Validator_settings
from scripts.run_ncu import profile_with_ncu
from agent.template.validator import ANALYZE_CUDA_ERROR_TEMPLATE, GENERATE_ERROR_REPORT_TEMPLATE, INIT_CUDA_ERROR_VALIDATOR_TEMPLATE, INIT_CUDA_IMPLEMNT_REPORT_VALIDATOR_TEMPLATE
from utils.utils import extract_error_report, extract_json, write_file



class Validator(LLM):
    def __init__(self, args: Dict):

        #super().__init__(server_name=args.server_name, model=args.model, max_tokens=16384, temperature=1.0, top_p=1.0)
        super().__init__(server_name="openai", model="gpt-5-mini", max_tokens=16384, temperature=0.8, top_p=1.0)

    def analyze_init_error(self, root_dir: Path, current_dir: Path, error_message: str, file_list: str, task_description: str):
        prompt = ANALYZE_CUDA_ERROR_TEMPLATE.substitute(
            error_message=error_message,
            file_list=file_list,
            task_description=task_description
        )
        out = self.chat(prompt)
        out = extract_json(out)
        write_file(current_dir / "error_analysis.json",  json.dumps(out, indent=2))
        return out
    
    def generate_error_report(self,  root_dir: Path, current_dir: Path, error_message:str, task_description: str, file_list: str, selected_files_content: str):
        
        prompt = GENERATE_ERROR_REPORT_TEMPLATE.substitute(
            error_message=error_message,
            task_description=task_description,
            file_list=file_list,
            selected_files_content=selected_files_content
        )
        out = self.chat(prompt)
        out = extract_json(out)
        write_file(current_dir / "error_report.json",  json.dumps(out, indent=2))
        return out
    

    def generate_init_error_report(self, current_dir: Path, source_code: str, entry_code: str, kernel_code: str, error_log: str):
        tqdm.write("generate_init_error_report")
        prompt = INIT_CUDA_ERROR_VALIDATOR_TEMPLATE.substitute(
            source_code=source_code,
            #entry_code=entry_code,
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
    
    def generate_init_cuda_impl_report(self, root_dir: Path, source_code: str, kernel_code: str):
        tqdm.write("generate_init_cuda_impl_report")
        prompt = INIT_CUDA_IMPLEMNT_REPORT_VALIDATOR_TEMPLATE.substitute(
            source_code=source_code,
            kernel_code=kernel_code,
        )
        while True:
            out = self.chat(prompt)
            impl_reprot = extract_json(out)
            if impl_reprot is not None:
                break
        write_file(root_dir /  "bootstrap" / "impl_report.txt", str(impl_reprot))  
        return impl_reprot
    
    def generate_init_ncu_report(self, root_dir: Path):
        tqdm.write("generate_init_ncu_report...")
        ncu_report = profile_with_ncu(str(root_dir / "spec" / "entry.py"))
        write_file(root_dir / "bootstrap" / "ncu_report.csv", ncu_report)
        return ncu_report
    

