import re
import subprocess
import os
import argparse
import pathlib
import pandas as pd

from pathlib import Path
from typing import List, Optional
from io import StringIO

GPU_Speed_Of_Light_Throughput = ",".join([ 
    "sm__throughput.avg.pct_of_peak_sustained_elapsed", # SM Throughput 
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", # GPU Memory Throughput 
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed", # L1 Throughput 
    "lts__throughput.avg.pct_of_peak_sustained_elapsed", # L2 Throughput 
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", # DRAM Throughput 
    "gpu__time_duration.avg", # Kernel Duration 
]) 

Compute_Workload_Analysis = ",".join([ 
    "sm__inst_executed.avg.per_cycle_elapsed", # Executed Ipc Elapsed 
    "sm__inst_executed.avg.per_cycle_active", # Executed Ipc Active 
    "sm__inst_issued.avg.per_cycle_active", # Issued Ipc Active 
    "sm__instruction_throughput.avg.pct_of_peak_sustained_active", # SM Busy 
    "sm__inst_issued.avg.pct_of_peak_sustained_active", # Issue Slots Busy 
]) 

Memory_Workload_Analysis = ",".join([ 
    "dram__bytes.avg.per_second", # DRAM Throughput (bytes/s) 
    "l1tex__t_sector_hit_rate.pct", # L1 Hit Rate 
    "lts__t_sector_hit_rate.pct", # L2 Hit Rate 
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed", # Memory Busy % 
    "gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed", # Max Bandwidth % 
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", # Shared Load Bank Conflicts 
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", # Shared Store Bank Conflicts 
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", # Total Shared Bank Conflicts 
]) 

Scheduler_Statistics = ",".join([ 
    "smsp__warps_active.avg.peak_sustained", # GPU Maximum Warps per Schedule 
    "smsp__warps_active.avg.per_cycle_active", # Active Warps per Schedule 
    "smsp__warps_eligible.avg.per_cycle_active", # Eligible Warps per Schedule 
    "smsp__issue_active.avg.per_cycle_active", # Issue Warps Per Schedule 
    "smsp__issue_inst0.avg.pct_of_peak_sustained_active"# No Eligible Rate 
])

FLOPS_METRICS = ",".join([
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", # FP32 Add
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", # FP32 Mul
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", # FP32 FMA
    "sm__inst_executed_pipe_tensor.sum",                 # Tensor Core Instructions
])


def read_ncu_csv_clean(csv_path: str | Path) -> pd.DataFrame:
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    header_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('"') and ('Kernel Name' in line or 'sm__' in line or 'launch__' in line):
            header_idx = i
            break

    df = pd.read_csv(StringIO(''.join(lines[header_idx:])))
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[pd.to_numeric(df.iloc[:, 0], errors='coerce').notnull()].copy()

    metric_map = {
        "Kernel Name": "Kernel Name",
        "launch__grid_size": "Grid Size",
        "launch__block_size": "Block Size",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM Throughput (%)",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "GPU Memory Throughput (%)",
        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed": "L1 Throughput (%)",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed": "L2 Throughput (%)",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "DRAM Throughput (%)",
        "dram__bytes.avg.per_second" : "DRAM Throughput (bytes/s)",
        "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed": "Momery Busy (%)",
        "gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed": "Max Bandwidth (%)",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "Shared Load Bank Conflicts", 
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "Shared Store Bank Conflicts", 
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": "Total Shared Bank Conflicts", 
        "gpu__time_duration.avg": "Duration (ns)",
        "sm__inst_executed.avg.per_cycle_elapsed": "Executed IPC (Elapsed)",
        "sm__inst_executed.avg.per_cycle_active": "Executed IPC (Active)",
        "sm__inst_issued.avg.per_cycle_active": "Issued IPC (Active)",
        "sm__instruction_throughput.avg.pct_of_peak_sustained_active": "SM Busy (%)",
        "sm__inst_issued.avg.pct_of_peak_sustained_active": "Issue Slots Busy (%)",            
        "l1tex__t_sector_hit_rate.pct": "L1 Hit Rate (%)",
        "lts__t_sector_hit_rate.pct": "L2 Hit Rate (%)",
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": "_fadd",
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": "_fmul",
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": "_ffma",
        "smsp__warps_active.avg.peak_sustained"  : "GPU Maximum Warps per Schedule ",
        "smsp__warps_active.avg.per_cycle_active" : "Active Warps per Schedule ",
        "smsp__warps_eligible.avg.per_cycle_active" : "Eligible Warps per Schedule "
    }

    final_metric_map = {}
    for raw_col in df.columns:
        for official_key, friendly_name in metric_map.items():
            if official_key in raw_col:
                final_metric_map[raw_col] = friendly_name

    df_clean = df[list(final_metric_map.keys())].rename(columns=final_metric_map)

    def clean_val(x):
        if pd.isna(x): return 0.0
        s = str(x).replace(',', '').replace('%', '').strip()
        return float(s) if s else 0.0

    numeric_cols = [c for c in df_clean.columns if c not in ["Kernel Name", "Grid Size", "Block Size"]]
    for col in numeric_cols:
        df_clean[col] = df_clean[col].apply(clean_val)
    if "DRAM Throughput (bytes/s)" in df_clean.columns:
        df_clean["DRAM Throughput (GB/s)"] =  df_clean["DRAM Throughput (bytes/s)"] / 1000 / 1000

    if "_fadd" in df_clean.columns and "Duration (us)" in df_clean.columns:
        total_flops = df_clean["_fadd"] + df_clean["_fmul"] + 2 * df_clean["_ffma"]
        df_clean["GFLOPS (FP32)"] = total_flops / (df_clean["Duration (us)"] * 1000)

    if "Duration (ns)" in df_clean.columns:
        df_clean["Duration (us)"] = df_clean["Duration (ns)"] / 1000

    if "Kernel Name" in df_clean.columns:
        df_clean["Kernel Name"] = df_clean["Kernel Name"].str.extract(r'^([A-Za-z_]\w*)', expand=False)
        df_clean = df_clean.drop_duplicates(subset=["Kernel Name"], keep="first")

    temp_cols = ["_fadd", "_fmul", "_ffma", "Duration (ns)", "DRAM Throughput (bytes/s)"]
    df_clean = df_clean.drop(columns=[c for c in temp_cols if c in df_clean.columns])

    return df_clean.copy()

def profile_with_ncu(
    script_path: str,
    device_idx: int,
    output_csv: str,
    python_executable: str = "python",
    ncu_path: str = "ncu",
    kernel_names: Optional[List[str]] = None,
):

    script_path = pathlib.Path(script_path).resolve()
    all_metrics = ",".join([
        GPU_Speed_Of_Light_Throughput,
        Compute_Workload_Analysis,
        Memory_Workload_Analysis,
        Scheduler_Statistics,
        FLOPS_METRICS
    ])

    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)

    cmd = [
        ncu_path,
        "--metrics", all_metrics,
        "--target-processes=all",
        "--replay-mode=kernel",
        "--csv",
        "--page=raw",
        "--profile-from-start=on",
        f"--log-file={str(output_csv)}",
        "--launch-skip=0",
        "--launch-count=20",
        python_executable,
        str(pathlib.Path(__file__).parent / "sanitizer_runner.py"),
        str(script_path),
        str(device_idx)
    ]

    if kernel_names:
        names = sorted({k.strip() for k in kernel_names if k and k.strip()})
        if names:
            insert_pos = cmd.index(python_executable)
            if len(names) == 1:
                cmd.insert(insert_pos, f"--kernel-name={names[0]}")
            else:
                pattern = "|".join(re.escape(k) for k in names)
                cmd.insert(insert_pos, f"--kernel-name=::regex:^({pattern})(\\(|$)")

    print("Running command:")
    print(" ".join(cmd))

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("NCU profiling failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_path", type=str, required=True,
                        help="Path to the Python script to profile")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Optional path to save CSV")

    args = parser.parse_args()
    profile_with_ncu(args.script_path, 1, args.csv_path)