import shutil
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple

from agent.role.analyzer import Analyzer
from agent.role.coder import Coder
from agent.role.validator import Validator
from agent.role.planner import Planner

from scripts.run_ncu import profile_with_ncu
from utils.utils import  delete_folder, copy_folder, dict_to_text, list_all_files, read_file, remove_justification, text_to_dict, write_file, extract_recommendation
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
        
            
        if not (task_root / "spec" / "ref.py").exists():
            shutil.copy2(task, task_root / "spec" / "ref.py")

        plan_file = task_root / "bootstrap" / "fusion_plan.json"
        plan = None

        if not plan_file.exists():
            plan = analyzer.gernerate_fuse_operator_plan(task_root, read_file(task))
        else:
            plan = text_to_dict(read_file(plan_file))
        plan = remove_justification(plan)
        task_description = plan["task_description"]
        if not (task_root / "spec" / "entry.py").exists():
            coder.generate_entry_code(task_root, 
                                      str(plan),
                                      read_file("./agent/template/example/example.py"), 
                                      read_file("./agent/template/example/example_entry.py"), 
                                      read_file(task), 
                                      task_name_no_num, 
                                      task_name_no_num, 
                                      str(task_root / "spec" / "kernel" / "kernel.cu"))
            

        bootstrap = Path(task_root / "bootstrap")
        error_report = None
        impl_report  = None
        bootstrap_final = task_root / "bootstrap" / "kernel.cu"
        for i in tqdm(range(args.bootstrap_iter), desc="Bootstrap Iterations"):
            msg = {}
            (bootstrap / f"iter_{i}").mkdir(parents=True, exist_ok=True)
            current_dir = bootstrap / f"iter_{i}"
            if i == 0:
                coder.gernerate_init_cuda_code(current_dir, 
                                                read_file('./agent/template/example/example.py'), 
                                                read_file('./agent/template/example/example.cu'), 
                                                read_file(task), 
                                                read_file(task_root / "spec" / "entry.py"),
                                                str(plan),
                                                task_name_no_num, 
                                                task_name_no_num)
            #else:
            #    coder.repair_init_cuda_code(current_dir,
            #                                read_file(str(task_root / "spec" / "ref.py")),
            #                                read_file(str(task_root / "spec" / "entry.py")),
            #                                read_file(str(task_root / "spec" / "kernel.cu")),
            #                                task_name_no_num, 
            #                                task_name_no_num, 
            #                                str(error_report))
            ##shutil.copy2(current_dir / "kernel.cu", task_root / "spec" / "kernel.cu")
            copy_folder(current_dir / "kernel", task_root / "spec" / "kernel")
            msg = test_kernel(task_root, current_dir, args.device)
            write_file(current_dir / "result.log", dict_to_text(msg))
            if msg['runnable'] == True:
                break
            error_analysis = validator.analyze_init_error(task_root,
                                                          current_dir,
                                                          read_file(current_dir / "result.log"),
                                                          str(list_all_files(task_root / "spec")),
                                                          task_description)
            most_likely_error_file = error_analysis["most_likely_error_file"]
            error_type = error_analysis["error_type"]

            show_files = error_analysis["show_files"] if "show_files" in error_analysis else []

            msg["most_likely_error_file"] = most_likely_error_file
            msg["error_type"] = error_type 
            if error_type == "parameter_alignment_error":
                pass
            elif error_type in ["cuda_illegal_memory", "cuda_device_assert"]:
                pass
            elif error_type == "value_error":
                pass
            else:
                pass
                #error_report = validator.generate_error_report()
            ##while True:
            ##    if msg["runnable"] == True:
            ##        break
            ##    if msg["message"]['type'] != "parameter_alignment_error":
            ##        break
            ##    else:
            ##        msg["advice"] = "please align ref model and test model parameters, hold the deep learning model paramter variable name same as much as possible, and make sure the model forward can run without error. "
            ##        coder.repair_entry_code(task_root, 
            ##                                read_file(task), 
            ##                                task_name_no_num, 
            ##                                task_name_no_num, 
            ##                                str(task_root / "spec" / "kernel.cu"), 
            ##                                read_file(task_root / "spec" / "entry.py"), 
            ##                                str(msg)
            ##        )
            ##        msg = test_kernel(task_root, current_dir, args.device)
            #if msg["runnable"] == True:
            #    shutil.copy2(current_dir / "kernel.cu", bootstrap_final)
            #    impl_report = validator.generate_init_cuda_impl_report(current_dir, 
            #                                                           read_file(str(task_root / "spec" / "ref.py")), 
            #                                                           read_file(str(task_root / "spec" / "kernel.cu")))
            #    break
            #error_report = validator.generate_init_error_report(
            #            current_dir,
            #            read_file(str(task_root / "spec" / "ref.py")),
            #            read_file(task_root / "spec" / "entry.py"),
            #            read_file(task_root / "spec" / "kernel.cu"),
            #            read_file(current_dir / "result.log")
            #        )
            #if error_report['ERROR_FILE'] == "entry.py":
            #    coder.repair_entry_code(task_root, read_file(task), task_name_no_num, task_name_no_num, str(task_root / "spec" / "kernel.cu"), read_file(task_root / "spec" / "entry.py"), error_report)

#=========================================== Opti =====================================
        pass

        

            