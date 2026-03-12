import os
from pathlib import Path
from scripts.run_test_kernel import _test_kernel_process, test_kernel
from utils.utils import sanitize_torch_error, write_file


if __name__ == "__main__":
    try:
        print( _test_kernel_process(Path("/home/haoyu/code/CudaOptiAgent/run/openai_gpt-5-mini_v2/level2/2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide"),Path("/home/haoyu/code/CudaOptiAgent/run/openai_gpt-5-mini/level2/3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU"),3))
    except Exception as e:
        #write_file("error.log", sanitize_torch_error(str(e)))
        write_file("error.log", str(e))