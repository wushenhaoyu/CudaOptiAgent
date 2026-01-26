import shutil
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple

from agent.role.analyzer import Analyzer
from agent.role.coder import Coder
from agent.role.validator import Validator
from agent.role.planner import Planner

from scripts.test_cpu import test_cpu
from utils.utils import dict_to_text, read_file, write_file, extract_recommendation
from scripts.test_kernel import test_kernel

def init_task(tasks: List[Path], run_dir: Path, args: Dict):

    analyzer = Analyzer(args=args)
    coder = Coder(args=args)
    validator = Validator(args=args)
    planner = Planner(args=args)

    for task in tasks:
        task_name = task.stem
        task_name_no_num = task_name.split('_', 1)[-1]
        task_root = (run_dir / task.parent.name / task_name).resolve()

        task_root.mkdir(parents=True, exist_ok=True)
        (task_root / "spec").mkdir(parents=True, exist_ok=True) 
        (task_root / "bootstrap").mkdir(parents=True, exist_ok=True) 
        (task_root / "optimize").mkdir(parents=True, exist_ok=True) 
        (task_root / "cpu").mkdir(parents=True, exist_ok=True) 
        shutil.copy2(task, task_root / "spec" / "ref.py")
        coder.generate_entry_code(task_root, read_file(task), task_name_no_num, task_name_no_num, str(task_root / "spec" / "kernel.cu"))
        bootstrap = Path(task_root / "bootstrap")
        cpu = Path(task_root / "cpu")
        hints = ""
        error_report = None
        for i in tqdm(range(args.bootstrap_iter), desc="cpu Iterations"):
            msg = {}
            (cpu / f"iter_{i}").mkdir(parents=True, exist_ok=True)
            current_dir = cpu / f"iter_{i}"
            if i == 0:
                coder.generate_init_cpu_code(current_dir, read_file(task), read_file(task_root / "spec" / "entry.py"))
            else:
                coder.repair_init_cpu_code(current_dir, read_file(str(task_root / "spec" / "kernel.cu")), read_file(task), read_file(task_root / "spec" / "entry.py"), str(error_report))
            shutil.copy2(current_dir / "kernel.cu", task_root / "spec" / "kernel.cu")
            while True:
                tqdm.write("test_cpu")
                msg = test_cpu(task_root, current_dir, args.device)
                write_file(current_dir / "result.log", dict_to_text(msg))
                if msg["runnable"] == True:
                    shutil.copy2(current_dir / "kernel.cu", task_root / "cpu" / "final.cu")
                    break
                error_report = validator.init_cpu_validator(
                    current_dir,
                    read_file(task_root / "spec" / "ref.py"),
                    read_file(task_root / "spec" / "entry.py"),
                    read_file(current_dir / "kernel.cu"),
                    read_file(current_dir / "result.log")
                )
                if error_report['ERROR_FILE'] == "entry.py":
                    coder.repair_entry_code(
                        task_root,
                        read_file(task),
                        task_name_no_num,
                        task_name_no_num,
                        str(task_root / "spec" / "kernel.cu"),
                        str(error_report)) 
                    continue
                else:
                    break
            if msg["runnable"] == True:
                break
               

        for i in tqdm(range(args.bootstrap_iter), desc="Bootstrap Iterations"):
            msg = {}
            (bootstrap / f"iter_{i}").mkdir(parents=True, exist_ok=True)
            current_dir = bootstrap / f"iter_{i}"
            if i == 0:
                tqdm.write("gernerate_init_cuda")
                coder.gernerate_init_cuda_code(current_dir, 
                                                read_file('./agent/template/example/example.py'), 
                                                read_file('./agent/template/example/example.cu'), 
                                                read_file(task), 
                                                read_file(task_root / "spec" / "entry.py"),
                                                task_name_no_num, 
                                                task_name_no_num, 
                                                )
            else:
                tqdm.write("gernerate_init_cuda")
                coder.repair_init_cuda_code(current_dir, 
                                            read_file(str(task_root / "spec" / "kernel.cu")), 
                                            read_file(str(task_root / "cpu" / "final.cu")), 
                                            task_name_no_num, 
                                            task_name_no_num, 
                                            str(error_report))
            shutil.copy2(current_dir / "kernel.cu", task_root / "spec" / "kernel.cu")
            msg = test_kernel(task_root, current_dir, args.device)
            write_file(current_dir / "result.log", dict_to_text(msg))
            if msg["runnable"] == True:
                    break
            error_report = validator.init_cuda_validator(
                    current_dir,
                    read_file(str(task_root / "cpu" / "final.cu")),
                    read_file(task_root / "spec" / "entry.py"),
                    read_file(task_root / "spec" / "kernel.cu"),
                    read_file(current_dir / "result.log")
                )









    