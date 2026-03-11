from scripts.run_ncs_debug import run_ncs_debug,_ncs_process
from pathlib import Path
if __name__ == "__main__":
    #print(_ncs_process(Path("/home/haoyu/code/CudaOptiAgent/run/openai_gpt-5-mini/level3/44_MiniGPTBlock/spec/entry.py"),0,None, Path("ncs_output.log")))
    print(run_ncs_debug(Path("/home/haoyu/code/CudaOptiAgent/run/openai_gpt-5-mini/level3/44_MiniGPTBlock/spec/entry.py")))