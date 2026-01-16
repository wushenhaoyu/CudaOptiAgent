import shutil

from pathlib import Path
from typing import Dict, List, Tuple

from agent.role.analyzer import Analyzer
from agent.role.coder import Coder
from agent.role.validator import Validator
from agent.role.planner import Planner

from utils.utils import read_file, write_file, extract_recommendation

def init_task(tasks: List[Path], run_dir: Path, args: Dict):

    analyzer = Analyzer(server_name = args.server_name, model = args.model)
    coder = Coder(server_name = args.server_name, model = args.model)
    validator = Validator(server_name = args.server_name, model = args.model)
    planner = Planner(server_name = args.server_name, model = args.model)

    for task in tasks:
        task_name = task.stem
        task_root = (run_dir / task.parent.name / task_name).resolve()

        task_root.mkdir(parents=True, exist_ok=True)
        (task_root / "spec").mkdir(parents=True, exist_ok=True) #spec exist ref.py and run.py
        (task_root / "bootstrap").mkdir(parents=True, exist_ok=True) #bootstrap 放生成最初正确cuda代码的过程
        (task_root / "optimize").mkdir(parents=True, exist_ok=True) #optimize 放生成优化cuda代码的过程

        shutil.copy2(task, task_root / "spec" / "ref.py")
        coder.generate_entry_code(task_root, read_file(task), task_name, task_name)

        bootstrap = Path(task_root / "bootstrap")

        for i in range(args.bootstrap_iter):
            (bootstrap / f"iter_{i}").mkdir(parents=True, exist_ok=True)
            current_dir = bootstrap / f"iter_{i}"
            if i == 0:
                input , output = analyzer.init_analyzer(current_dir, args)
                write_file(current_dir / "analyzer_io.txt", f"Input Prompt:\n{input}\n\nOutput Response:\n{output}")
                hints = extract_recommendation(output)
                
            else:
                input , output = analyzer.init_analyzer(current_dir, args)








    