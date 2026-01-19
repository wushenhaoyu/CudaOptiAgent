import argparse
import logging
from pathlib import Path
from datetime import datetime
from scripts.init_task import init_task


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="CUDA device index for benchmarking")
    parser.add_argument("--server_name", type=str, default="deepseek")
    parser.add_argument("--model", type=str, default="deepseek-reasoner")
    parser.add_argument("--model_choice", type=int, default=0)
    parser.add_argument("--task_level", type=int, default=2, choices=[0, 1, 2, 3, 4], help="task level")
    parser.add_argument("--task_id", type=int, default=3, help="task id")
    parser.add_argument("--task_dir", type=str, default="./benchmark/KernelBench")
    parser.add_argument("--gpu_name", type=str, default="RTX3070Ti_Laptop", help="GPU name for hwinfo task")
    parser.add_argument("--results_dir", type=str, default="./run")
    parser.add_argument("--bootstrap_iter", type=int, default=10)
    return parser

def collect_task(root: Path, task_level = 0, task_id = 0):
    tasks = []
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory")

    levels = range(1, 4) if task_level == 0 else [task_level]

    tasks = []
    for lvl in levels:
        lvl_dir = root / f"level{lvl}"
        if not lvl_dir.is_dir():
            continue
        py_files = sorted(lvl_dir.glob("*.py"), key=lambda p: int(p.stem.split('_')[0]))
        if task_id == 0:          
            tasks.extend(py_files)
        else:                     
            if 1 <= task_id <= len(py_files):
                tasks.append(py_files[task_id - 1])
    return tasks

def make_run_dir(base_dir: Path, server_name: str, model: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{server_name}_{model}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Run directory created at: {run_dir}")
    return run_dir

def main():
    parser = build_parser()

    args = parser.parse_args()

    tasks = collect_task(Path(args.task_dir), task_level=args.task_level, task_id=args.task_id)

    dir = make_run_dir(Path(args.results_dir), args.server_name, args.model)

    init_task(tasks, dir, args)

if __name__ == "__main__":
    main()