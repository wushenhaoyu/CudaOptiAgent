import os
from pathlib import Path
from scripts.run_test_kernel import _test_kernel_process, test_kernel


if __name__ == "__main__":
    print(test_kernel(Path("/home/haoyu/code/CudaOptiAgent/run/openai_gpt-5-mini/level3/44_MiniGPTBlock"),Path("/home/haoyu/code/CudaOptiAgent/run/openai_gpt-5-mini/level2/3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU"),1))