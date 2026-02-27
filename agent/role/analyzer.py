import json
from tqdm import tqdm
from pathlib import Path
from typing import Dict
from agent.llm import LLM
from agent.settings import Analyzer_settings
from utils.utils import extract_json, extract_recommendation, strip_fence, write_file, read_file
from agent.template.analyzer import FUSE_OPERATOR_TEMPLATE
from agent.template.gpu_info import GPU_SPEC_INFO



class Analyzer(LLM):
    def __init__(self, args: Dict):
        super().__init__(server_name=args.server_name, model=args.model, max_tokens=16384, temperature=1.0, top_p=1.0)

    def gernerate_fuse_operator_plan(self, root_dir: Path, source_code: str):
        tqdm.write("Generating fusion plan...")
        prompt = FUSE_OPERATOR_TEMPLATE.substitute(
            #example_source_code=read_file("./agent/template/example/fusion_example.py"),
            #example_fusion_plan=read_file("./agent/template/example/fusion_example.json"),
            source_code=source_code
        )
        while True:
            out = self.chat(prompt)
            plan = extract_json(out)
            if plan is not None:
                break
        write_file(root_dir / "spec" / "fusion_plan.json", json.dumps(plan, indent=2))
        return plan
    
        