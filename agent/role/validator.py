import json

from tqdm import tqdm
from pathlib import Path
from typing import Dict
from agent.llm import LLM
from agent.settings import Validator_settings
from scripts.run_ncu import profile_with_ncu
from agent.template.validator import ANALYZE_CUDA_ERROR_TEMPLATE, ANALYZE_CUDA_ERROR_TEMPLATE_SECOND, DEBUG_SCRIPT_TEMPLATE, GENERATE_ERROR_REPORT_TEMPLATE, GENERATE_ERROR_REPORT_TEMPLATE_NO_CONTENT_WITH_LAST
from utils.utils import extract_error_report, extract_json, strip_fence, write_file



class Validator(LLM):
    def __init__(self, args: Dict):

        #super().__init__(server_name=args.server_name, model=args.model, max_tokens=16384, temperature=1.0, top_p=1.0)
        super().__init__(server_name="openai", model="gemini-3-flash-preview", max_tokens=16384, temperature=0.8, top_p=1.0)

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
    
    def analyze_init_error_second(self, root_dir: Path, current_dir: Path, error_type: str, error_location: str, error_message: str, file_list: str, task_description: str):
        prompt = ANALYZE_CUDA_ERROR_TEMPLATE_SECOND.substitute(
            error_type=error_type,
            error_location=error_location,
            error_message=error_message,
            file_list=file_list,
            task_description=task_description
        )
        out = self.chat(prompt)
        out = extract_json(out)
        write_file(current_dir / "error_analysis_second.json",  json.dumps(out, indent=2))
        return out
    
    def generate_error_report(self,  root_dir: Path, current_dir: Path, error_message:str, task_description: str, file_list: str, selected_files_content: str, problem_kernel_name: str, last_error_report: dict):
        tqdm.write("generate_init_error_report")
        prompt = None
        #if last_error_report != None:
        #    for file in last_error_report.get("last_error_report",[]):
        #        if file["file_name"] == problem_kernel_name:
        #            prompt = GENERATE_ERROR_REPORT_TEMPLATE_NO_CONTENT_WITH_LAST.substitute(
        #                error_message=error_message,
        #                task_description=task_description,
        #                file_list=file_list,
        #                selected_files_content=selected_files_content,
        #                last_error_report=str(last_error_report)
        #            )
        if prompt == None:
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
    

    def generate_debug_script(self, root_dir: Path, current_dir: Path, debug_example: str, entry_code: str, ref_code: str):
        tqdm.write("generate value debug script...")
        prompt = DEBUG_SCRIPT_TEMPLATE.substitute(
            debug_example = debug_example,
            entry_code = entry_code,
            ref_code =  ref_code
        )
        out = self.chat(prompt)
        out = strip_fence(out)
        write_file(root_dir / "spec" / "value_debug.py", out)

    
    
    
    def generate_init_ncu_report(self, root_dir: Path):
        tqdm.write("generate_init_ncu_report...")
        ncu_report = profile_with_ncu(str(root_dir / "spec" / "entry.py"))
        write_file(root_dir / "bootstrap" / "ncu_report.csv", ncu_report)
        return ncu_report
    

