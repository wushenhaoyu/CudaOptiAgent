from pathlib import Path
from scripts.run_value_debug import _run_model_debug_script
from utils.utils import write_file

if __name__ == "__main__":
    try:
        print(_run_model_debug_script(Path('/home/haoyu/code/CudaOptiAgent/run/openai_gpt-5-mini_v1/level3/28_VisionTransformer/spec/value_debug.py'),4))
    except Exception as e:
        write_file("debug.log",str(e))
