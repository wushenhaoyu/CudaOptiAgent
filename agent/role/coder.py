from pathlib import Path
from agent.llm import LLM
from agent.template.coder import INIT_ENTRY_CODER_TEMPLATE, INIT_CUDA_CODER_TEMPLATE
from utils.utils import strip_fence, write_file, read_file

class Coder(LLM):
    def __init__(self, server_name: str = "deepseek",model: str = "deepseek-chat", max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 1.0):
        super().__init__(server_name=server_name, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def generate_entry_code(self, root_dir: Path ,source_code: str, cuda_module_name: str, cuda_function_name: str):

        prompt = INIT_ENTRY_CODER_TEMPLATE.substitute(
            source_code=source_code,
            cuda_module_name=cuda_module_name,
            cuda_function_name=cuda_function_name
        )

        entry_code = strip_fence(self.chat(prompt))

        write_file(root_dir / "spec" / "entry.py", entry_code)
        

    def gernerate_cuda_code(self, root_dir: Path, source_code: str, cuda_module_name: str, cuda_function_name: str):
        prompt = INIT_CUDA_CODER_TEMPLATE.substitute(
            source_code=source_code,
            cuda_module_name=cuda_module_name,
            cuda_function_name=cuda_function_name
        )

        cuda_code = strip_fence(self.chat(prompt))

        write_file(root_dir / "spec" / "kernel.cu", cuda_code)

        return prompt, cuda_code
        