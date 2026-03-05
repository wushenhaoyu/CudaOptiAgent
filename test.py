import os
from pathlib import Path
from scripts.test_kernel import _test_kernel_process, test_kernel


if __name__ == "__main__":
    print(_test_kernel_process(Path("/home/haoyu/code/CudaOptiAgent/run/openai_gpt-5-mini/level2/3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU"),Path("/home/haoyu/code/CudaOptiAgent/run/openai_gpt-5-mini/level2/3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU"),1))