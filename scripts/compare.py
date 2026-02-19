import shutil
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple

from agent.role.analyzer import Analyzer
from agent.role.coder import Coder
from agent.role.validator import Validator
from agent.role.planner import Planner

from scripts.test_cpu import test_cpu
from utils.utils import dict_to_text, read_file, text_to_dict, write_file, extract_recommendation
from scripts.test_kernel import test_kernel


def init_task(tasks: List[Path], run_dir: Path, args: Dict):
    import ast
    
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
        
        final_bootstrap = task_root / "bootstrap" / "final.cu"
        if final_bootstrap.exists():
            print(f"[SKIP] {task_name}: already completed")
            continue
            
        if not (task_root / "spec" / "ref.py").exists():
            shutil.copy2(task, task_root / "spec" / "ref.py")
            
        if not (task_root / "spec" / "entry.py").exists():
            coder.generate_entry_code_(task_root, read_file('./agent/template/example/example.py'), read_file('./agent/template/example/example_entry_.py'), read_file(task), task_name_no_num, task_name_no_num, str(task_root / "spec" / "kernel.cu"))
            
        bootstrap = Path(task_root / "bootstrap")
        hints = ""
        error_report = None
        
        existing_iters = sorted([
            d for d in bootstrap.iterdir() 
            if d.is_dir() and d.name.startswith("iter_")
        ], key=lambda x: int(x.name.split("_")[1]))
        
        start_iter = 0
        if existing_iters:
            last_iter_dir = existing_iters[-1]
            result_log = last_iter_dir / "result.log"
            error_report_path = last_iter_dir / "error_report.txt"
            
            if result_log.exists():
                try:
                    last_result = text_to_dict(read_file(result_log))
                    if last_result.get("runnable") == True:
                        print(f"[RESUME] {task_name}: done at iter {len(existing_iters)-1}")
                        shutil.copy2(last_iter_dir / "kernel.cu", final_bootstrap)
                        continue
                except:
                    pass
            
            if error_report_path.exists():
                try:
                    error_report = ast.literal_eval(read_file(error_report_path))
                    print(f"[RESUME] {task_name}: loaded error report from iter {len(existing_iters)-1}")
                except:
                    error_report = None
            
            start_iter = len(existing_iters)
            print(f"[RESUME] {task_name}: resuming from iter {start_iter}")
        
        for i in tqdm(range(start_iter, args.bootstrap_iter), desc="Bootstrap Iterations", initial=start_iter, total=args.bootstrap_iter):
            msg = {}
            (bootstrap / f"iter_{i}").mkdir(parents=True, exist_ok=True)
            current_dir = bootstrap / f"iter_{i}"
            
            if (current_dir / "result.log").exists() and i < args.bootstrap_iter - 1:
                try:
                    cached_result = text_to_dict(read_file(current_dir / "result.log"))
                    if cached_result.get("runnable") == True:
                        shutil.copy2(current_dir / "kernel.cu", final_bootstrap)
                        print(f"[CACHE] iter_{i} already runnable")
                        break
                except:
                    pass
            
            if i == 0:
                tqdm.write("generate_init_cuda")
                coder.gernerate_init_cuda_code(current_dir, 
                                                read_file('./agent/template/example/example.py'), 
                                                read_file('./agent/template/example/example.cu'), 
                                                read_file(task), 
                                                read_file(task_root / "spec" / "entry.py"),
                                                task_name_no_num, 
                                                task_name_no_num)
            else:
                tqdm.write("repair_init_cuda")
                coder.repair_init_cuda_code_(current_dir,
                                            read_file(str(task_root / "spec" / "entry.py")),
                                            read_file(str(task_root / "spec" / "kernel.cu")), 
                                            task_name_no_num, 
                                            task_name_no_num, 
                                            str(error_report))
            shutil.copy2(current_dir / "kernel.cu", task_root / "spec" / "kernel.cu")
            msg = test_kernel(task_root, current_dir, args.device)
            write_file(current_dir / "result.log", dict_to_text(msg))
            
            if msg["runnable"] == True:
                shutil.copy2(current_dir / "kernel.cu", final_bootstrap)
                break
                
            error_report = validator.init_cuda_validator_(
                    current_dir,
                    read_file(str(task_root / "spec" / "ref.py")),
                    read_file(task_root / "spec" / "entry.py"),
                    read_file(task_root / "spec" / "kernel.cu"),
                    read_file(current_dir / "result.log")
                )
#=============================== Optimize ======================================================