import subprocess
import os
import argparse
import pathlib
from typing import Optional

GPU_Speed_Of_Light_Throughput = ",".join([
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__time_duration.sum",
])

Compute_Workload_Analysis = ",".join([
    "sm__inst_executed.avg.per_cycle_elapsed",
    "sm__inst_executed.avg.per_cycle_active",
    "sm__inst_issued.avg.per_cycle_active",
    "sm__instruction_throughput.avg.pct_of_peak_sustained_active",
    "sm__inst_issued.avg.pct_of_peak_sustained_active",
])

Memory_Workload_Analysis = ",".join([
    "dram__bytes.sum.per_second",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
])

Scheduler_Statistics = ",".join([
    "smsp__warps_active.avg.peak_sustained",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__issue_inst0.avg.pct_of_peak_sustained_active"
])

def profile_with_ncu(
    script_path: str,
    python_executable: str = "python",
    ncu_path: str = "ncu",
    write_csv: Optional[str] = None
) -> str:
    """
    Run Nsight Compute on a Python script and return the CSV string.
    If write_csv is provided, also save the CSV to that file.
    """
    script_path = pathlib.Path(script_path).resolve()
    all_metrics = ",".join([
        GPU_Speed_Of_Light_Throughput,
        Compute_Workload_Analysis,
        Memory_Workload_Analysis,
        Scheduler_Statistics
    ])

    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)


    cmd = [
        ncu_path,
        "--metrics", all_metrics,
        "--target-processes", "all",
        "--replay-mode", "application",
        "--csv",
        python_executable,
        str(script_path)
    ]

    print("Running command:")
    print(" ".join(cmd))

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("NCU profiling failed.")
    return result.stdout  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_path", type=str, required=True,
                        help="Path to the Python script to profile")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Optional path to save CSV")

    args = parser.parse_args()
    csv_content = profile_with_ncu(args.script_path, write_csv=args.csv_path)
    print("\n--- CSV Output Preview ---\n")
    print(csv_content[:1000])  