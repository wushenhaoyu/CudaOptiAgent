import shutil

from pathlib import Path
from typing import Dict, List, Tuple

from agent.role.analyzer import Analyzer
from agent.role.coder import Coder
from agent.role.validator import Validator
from agent.role.planner import Planner

from utils.utils import read_file, write_file, extract_recommendation
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
        (task_root / "spec").mkdir(parents=True, exist_ok=True) #spec exist ref.py and run.py
        (task_root / "bootstrap").mkdir(parents=True, exist_ok=True) #bootstrap 放生成最初正确cuda代码的过程
        (task_root / "optimize").mkdir(parents=True, exist_ok=True) #optimize 放生成优化cuda代码的过程

        shutil.copy2(task, task_root / "spec" / "ref.py")
        coder.generate_entry_code(task_root, read_file(task), task_name_no_num, task_name_no_num)

        bootstrap = Path(task_root / "bootstrap")

        for i in range(args.bootstrap_iter):
            (bootstrap / f"iter_{i}").mkdir(parents=True, exist_ok=True)
            current_dir = bootstrap / f"iter_{i}"
            if i == 0:
                input , output = analyzer.init_analyzer(task_root, args)
                write_file(current_dir / "analyzer_io.txt", f"Input Prompt:\n{input}\n\nOutput Response:\n{output}")

                hints = extract_recommendation(output)
                input , output = coder.gernerate_cuda_code(current_dir, 
                                                           read_file('./agent/template/example/example.py'), 
                                                           read_file('./agent/template/example/example.cu'), 
                                                           read_file(task), 
                                                           task_name_no_num, 
                                                           task_name_no_num, 
                                                           hints)
                write_file(current_dir / "coder_io.txt", f"Input Prompt:\n{input}\n")
                write_file(current_dir / "kernel.cu", output)
                shutil.copy2(current_dir / "kernel.cu", task_root / "spec" / "kernel.cu")
                data = test_kernel(current_dir, args.device)
                print(data)








    