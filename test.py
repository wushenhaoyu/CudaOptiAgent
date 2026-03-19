from utils.utils import extract_all_kernels_flat, extract_cuda_kernel_names
from pathlib import Path
from scripts.run_ncu import profile_with_ncu, read_ncu_csv_clean

if __name__ == "__main__":
    #profile_with_ncu("/home/haoyu/code/CudaOptiAgent/spec/entry.py",0,"output.csv",kernel_names=extract_cuda_kernel_names(Path('/home/haoyu/code/CudaOptiAgent/spec/kernel/SelfAttn_AttnV_MatMul.cu')))
    #print(extract_all_kernels_flat(Path("/home/haoyu/code/CudaOptiAgent/spec/kernel")))
    profile_with_ncu("/home/haoyu/code/CudaOptiAgent/spec/entry.py",0,"output.csv",kernel_names=extract_all_kernels_flat(Path('/home/haoyu/code/CudaOptiAgent/spec/kernel')))
    a = read_ncu_csv_clean(Path('/home/haoyu/code/fake/CudaOptiAgent/output.csv'))
    a.to_csv("data.csv", index=False, encoding='utf-8')