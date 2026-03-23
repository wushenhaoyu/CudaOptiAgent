import json
from tqdm import tqdm
from pathlib import Path
from typing import Dict
from agent.llm import LLM
from agent.settings import Analyzer_settings
from utils.utils import extract_json, extract_recommendation, strip_fence, write_file, read_file
from agent.template.analyzer import FUSE_ANALYZER_BASE_TEMPLATE, GENERATE_ERROR_REPORT_TEMPLATE
from agent.template.gpu_info import GPU_SPEC_INFO



class Analyzer(LLM):
    def __init__(self, args: Dict):
        #super().__init__(server_name=args.server_name, model=args.model, max_tokens=16384, temperature=1.0, top_p=1.0)
        super().__init__(server_name="openai", model="gemini-3-flash-preview", max_tokens=16384, temperature=0.8, top_p=1.0)

    def gernerate_fuse_operator_plan(self, root_dir: Path, source_code: str):
        tqdm.write("Generating fusion plan...")
        prompt = FUSE_ANALYZER_BASE_TEMPLATE.substitute(
            #example_source_code=read_file("./agent/template/example/fusion_example.py"),
            #example_fusion_plan=read_file("./agent/template/example/fusion_example.json"),
            source_code=source_code
        )
        while True:
            out = self.chat(prompt)
            plan = extract_json(out)
            if plan is not None:
                break
        write_file(root_dir / "bootstrap" / "fusion_plan.json", json.dumps(plan, indent=2))
        return plan
    
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
    
        