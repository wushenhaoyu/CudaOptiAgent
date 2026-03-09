from tqdm import tqdm

from pathlib import Path
from typing import Dict
from agent.llm import LLM
from agent.settings import Coder_settings
from agent.template.coder import INIT_ENTRY_CODER_TEMPLATE, INIT_CUDA_CODER_TEMPLATE, REPAIR_CUDA_CODER_TEMPLATE, REPAIR_ENTRY_CODER_TEMPLATE
from utils.utils import load_related_files, save_cuda_files_clean, strip_fence, write_file, read_file



class Coder(LLM):
    def __init__(self, args: Dict):


        #super().__init__(server_name=args.server_name, model=args.model, max_tokens=16384, temperature=0.0, top_p=1.0)
        super().__init__(server_name="openai", model="gpt-5.3-codex", max_tokens=16384, temperature=0.1, top_p=1.0)

    def generate_entry_code(self, root_dir: Path , fusion_plan: str, example_source_code: str, example_entry_code: str, source_code: str, cuda_module_name: str, cuda_function_name: str, kernel_dir: str):
        tqdm.write("generate_entry_code")
        prompt = INIT_ENTRY_CODER_TEMPLATE.substitute(
            fusion_plan=fusion_plan,
            example_source_code=example_source_code,
            example_entry_code=example_entry_code,
            source_code=source_code,
            #cuda_module_name=cuda_module_name,
            #cuda_function_name=cuda_function_name,
            kernel_dir=kernel_dir
        )

        entry_code = strip_fence(self.chat(prompt))

        write_file(root_dir / "spec" / "entry.py", entry_code)

    def repair_entry_code(self, root_dir: Path ,source_code: str, cuda_module_name: str, cuda_function_name: str, kernel_dir: str, entry_code: str, error_report: str):
        tqdm.write("repair_entry_code")
        prompt = REPAIR_ENTRY_CODER_TEMPLATE.substitute(
            source_code=source_code,
            cuda_module_name=cuda_module_name,
            cuda_function_name=cuda_function_name,
            kernel_dir=kernel_dir,
            entry_code=entry_code,
            error_report=error_report
        )

        entry_code = strip_fence(self.chat(prompt))

        write_file(root_dir / "spec" / "entry.py", entry_code)
        

    def gernerate_init_cuda_code(self, 
                            current_dir: Path, 
                            example_source_code: str, 
                            example_cuda_code: str, 
                            source_code: str, 
                            entry_code:str,
                            fusion_plan: str,
                            cuda_module_name: str, 
                            cuda_function_name: str):
        prompt = INIT_CUDA_CODER_TEMPLATE.substitute(
            #example_source_code = example_source_code,
            #example_cuda_code = example_cuda_code,
            source_code=source_code,
            entry_code=entry_code,
            fusion_plan=fusion_plan,
        )
        tqdm.write("generate_init_cuda")

        write_file(current_dir / "coder_in.txt", f"Input Prompt:\n{prompt}\n")

        out = self.chat(prompt)

        write_file(current_dir / "coder_out.txt", f"Output:\n{out}\n")

        save_cuda_files_clean(out, str(current_dir / "kernel"))

        

        #cuda_code = strip_fence(self.chat(prompt))
        
        #write_file(current_dir / "kernel.cu", cuda_code)

        
    def repair_init_cuda_code(self,
                              root_dir: Path, 
                              current_dir: Path, 
                              file_list: str, 
                              repair_file_list: list):
        tqdm.write("repair_init_cuda")
        for i , file  in enumerate(repair_file_list):
            target_file_name = file["file_name"]
            related_files = file["related_files"]
            related_files_content = load_related_files(related_files, str(root_dir / "spec"))
            target_file_content = read_file(root_dir / "spec" / target_file_name)
            error_items = file["issues"]
            prompt = REPAIR_CUDA_CODER_TEMPLATE.substitute(
                file_list=file_list,
                target_file_name=target_file_name,
                target_file_content=target_file_content,
                related_files_content=related_files_content,
                error_items=error_items
            )
            write_file(current_dir / f"coder_io_{i}.txt", f"Input Prompt:\n{prompt}\n")
            out = strip_fence(self.chat(prompt))
            write_file(current_dir / "spec" / target_file_name, out)




