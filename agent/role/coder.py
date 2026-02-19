from pathlib import Path
from typing import Dict
from agent.llm import LLM
from agent.settings import Coder_settings
from agent.template.coder import INIT_CPU_CODER_TEMPLATE, INIT_ENTRY_CODER_TEMPLATE, INIT_CUDA_CODER_TEMPLATE, REPAIR_CPU_CODER_TEMPLATE, REPAIR_CUDA_CODER_TEMPLATE, REPAIR_CUDA_CODER_TEMPLATE_, REPAIR_ENTRY_CODER_TEMPLATE, RESTORE_ENTRY_CODE_TEMPLATE
from utils.utils import strip_fence, write_file, read_file



class Coder(LLM):
    def __init__(self, args: Dict):
        setting_id = args.model_choice
        setting = Coder_settings[setting_id]

        super().__init__(server_name=setting["server_name"], model=setting["model"], max_tokens=setting["max_tokens"], temperature=setting["temperature"], top_p=setting["top_p"])

    def generate_init_cpu_code(self, current_dir: Path, source_code: str, entry_code: str):

        prompt = INIT_CPU_CODER_TEMPLATE.substitute(
            source_code=source_code,
            entry_code=entry_code
        )
        

        cpu_code = strip_fence(self.chat(prompt))

        write_file(current_dir/ "kernel.cu", cpu_code)

    def repair_init_cpu_code(self, current_dir: Path, last_cpu_code: str, source_code: str, entry_code: str, hints: str):

        prompt = REPAIR_CPU_CODER_TEMPLATE.substitute(
            last_cpu_code=last_cpu_code,
            source_code=source_code,
            entry_code=entry_code,
            hints=hints
        )

        cpu_code = strip_fence(self.chat(prompt))

        write_file(current_dir/ "kernel.cu", cpu_code)

    def generate_entry_code(self, root_dir: Path , exmaple_source_code: str, example_entry_code: str, source_code: str, cuda_module_name: str, cuda_function_name: str, kernel_dir: str):

        prompt = INIT_ENTRY_CODER_TEMPLATE.substitute(
            example_source_code=exmaple_source_code,
            example_entry_code=example_entry_code,
            source_code=source_code,
            cuda_module_name=cuda_module_name,
            cuda_function_name=cuda_function_name,
            kernel_dir=kernel_dir
        )

        entry_code = strip_fence(self.chat(prompt))

        write_file(root_dir / "spec" / "entry.py", entry_code)

    

    def repair_entry_code(self, root_dir: Path ,source_code: str, cuda_module_name: str, cuda_function_name: str, kernel_dir: str, entry_code: str, error_report: str):

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
                            cuda_module_name: str, 
                            cuda_function_name: str):
        prompt = INIT_CUDA_CODER_TEMPLATE.substitute(
            example_source_code = example_source_code,
            example_cuda_code = example_cuda_code,
            source_code=source_code,
            entry_code=entry_code,
            cuda_module_name=cuda_module_name,
            cuda_function_name=cuda_function_name
        )

        write_file(current_dir / "coder_io.txt", f"Input Prompt:\n{prompt}\n")

        cuda_code = strip_fence(self.chat(prompt))
        
        write_file(current_dir / "kernel.cu", cuda_code)

        
    def repair_init_cuda_code(self, 
                            current_dir: Path, 
                            last_kernel_code: str,
                            cpu_code: str, 
                            cuda_module_name: str, 
                            cuda_function_name: str,
                            hints: str):
        prompt = REPAIR_CUDA_CODER_TEMPLATE.substitute(
            last_kernel_code=last_kernel_code,
            cpu_code=cpu_code,
            cuda_module_name=cuda_module_name,
            cuda_function_name=cuda_function_name,
            hints=hints
        )

        write_file(current_dir / "coder_io.txt", f"Input Prompt:\n{prompt}\n")

        cuda_code = strip_fence(self.chat(prompt))

        write_file(current_dir / "kernel.cu", cuda_code)


    def restore_entry_code(self, root_dir: Path, source_code: str, entry_code: str):
        prompt = RESTORE_ENTRY_CODE_TEMPLATE.substitute(
            source_code=source_code,
            entry_code=entry_code
        )
        entry_code = strip_fence(self.chat(prompt))

        write_file(root_dir / "spec" / "entry.py", entry_code)


    def repair_init_cuda_code_(self, 
                            current_dir: Path,
                            entry_code: str, 
                            last_kernel_code: str,
                            cuda_module_name: str, 
                            cuda_function_name: str,
                            hints: str):
        prompt = REPAIR_CUDA_CODER_TEMPLATE_.substitute(
            entry_code=entry_code,
            last_kernel_code=last_kernel_code,
            cuda_module_name=cuda_module_name,
            cuda_function_name=cuda_function_name,
            hints=hints
        )

        write_file(current_dir / "coder_io.txt", f"Input Prompt:\n{prompt}\n")

        cuda_code = strip_fence(self.chat(prompt))

        write_file(current_dir / "kernel.cu", cuda_code)
        
    def generate_entry_code_(self, root_dir: Path , exmaple_source_code: str, example_entry_code: str, source_code: str, cuda_module_name: str, cuda_function_name: str, kernel_dir: str):

        prompt = INIT_ENTRY_CODER_TEMPLATE.substitute(
            example_source_code=exmaple_source_code,
            example_entry_code=example_entry_code,
            source_code=source_code,
            cuda_module_name=cuda_module_name,
            cuda_function_name=cuda_function_name,
            kernel_dir=kernel_dir
        )

        entry_code = strip_fence(self.chat(prompt))

        write_file(root_dir / "spec" / "entry.py", entry_code)