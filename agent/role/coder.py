from pathlib import Path
from typing import Dict
from agent.llm import LLM
from agent.settings import Coder_settings
from agent.template.coder import INIT_CUDA_CODER_TEMPLATE_, INIT_ENTRY_CODER_TEMPLATE, INIT_CUDA_CODER_TEMPLATE
from utils.utils import strip_fence, write_file, read_file



class Coder(LLM):
    def __init__(self, args: Dict):
        setting_id = args.model_choice
        setting = Coder_settings[setting_id]

        super().__init__(server_name=setting["server_name"], model=setting["model"], max_tokens=setting["max_tokens"], temperature=setting["temperature"], top_p=setting["top_p"])

    def generate_entry_code(self, root_dir: Path ,source_code: str, cuda_module_name: str, cuda_function_name: str, kernel_dir: str):

        prompt = INIT_ENTRY_CODER_TEMPLATE.substitute(
            source_code=source_code,
            cuda_module_name=cuda_module_name,
            cuda_function_name=cuda_function_name,
            kernel_dir=kernel_dir
        )

        entry_code = strip_fence(self.chat(prompt))

        write_file(root_dir / "spec" / "entry.py", entry_code)
        

    def gernerate_init_cuda_code(self, 
                            root_dir: Path, 
                            example_source_code: str, 
                            example_cuda_code: str, 
                            source_code: str, 
                            cuda_module_name: str, 
                            cuda_function_name: str):
        prompt = INIT_CUDA_CODER_TEMPLATE.substitute(
            example_source_code = example_source_code,
            example_cuda_code = example_cuda_code,
            source_code=source_code,
            cuda_module_name=cuda_module_name,
            cuda_function_name=cuda_function_name
        )

        cuda_code = strip_fence(self.chat(prompt))

        return prompt, cuda_code
    

    def gernerate_init_cuda_code_(self, 
                            root_dir: Path, 
                            last_kernel_code: str,
                            source_code: str, 
                            cuda_module_name: str, 
                            cuda_function_name: str,
                            hints: str):
        prompt = INIT_CUDA_CODER_TEMPLATE_.substitute(
            last_kernel_code=last_kernel_code,
            source_code=source_code,
            cuda_module_name=cuda_module_name,
            cuda_function_name=cuda_function_name,
            hints=hints
        )

        cuda_code = strip_fence(self.chat(prompt))

        return prompt, cuda_code
        