import shutil
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple

from agent.role.analyzer import Analyzer
from agent.role.coder import Coder
from agent.role.validator import Validator
from agent.role.planner import Planner

from scripts.run_ncs_debug import run_ncs_debug
from scripts.run_value_debug import run_model_debug
from scripts.run_ncu import profile_with_ncu
from utils.utils import  delete_folder, copy_folder, dict_to_text, list_all_files, load_show_files, read_file, remove_justification, text_to_dict, write_file, extract_recommendation
from scripts.run_test_kernel import test_kernel


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
        msg = None
        bootstrap_final = task_root / "bootstrap" / "kernel.cu"
        kernel_iter = 0

        while kernel_iter < args.bootstrap_iter:

            print(f"\n=== Kernel Iteration {kernel_iter} ===")

            current_dir = bootstrap / f"iter_{kernel_iter}"
            current_dir.mkdir(parents=True, exist_ok=True)


            if kernel_iter == 0:
                if not (current_dir / "kernel").exists():
                    coder.gernerate_init_cuda_code(
                        current_dir,
                        read_file('./agent/template/example/example.py'),
                        read_file('./agent/template/example/example.cu'),
                        read_file(task),
                        read_file(task_root / "spec" / "entry.py"),
                        str(plan),
                        task_name_no_num,
                        task_name_no_num
                    )
                    copy_folder(current_dir / "kernel", task_root / "spec" / "kernel")
            else:
                repair_file_list = error_report["files"]
                coder.repair_init_cuda_code(
                    root_dir=task_root,
                    current_dir=current_dir,
                    file_list=str(list_all_files(task_root / "spec")),
                    repair_file_list=repair_file_list
                )
                copy_folder(task_root / "spec" / "kernel" , current_dir / "kernel")
            while True:

                msg = test_kernel(task_root, current_dir, args.device)
                write_file(current_dir / "result.log", dict_to_text(msg))

                if msg['runnable']:
                    print("✅ Success")
                    break

                error_analysis = validator.analyze_init_error(
                    task_root,
                    current_dir,
                    read_file(current_dir / "result.log"),
                    str(list_all_files(task_root / "spec")),
                    task_description
                )

                error_type = error_analysis["error_type"]
                most_likely_error_file = error_analysis["most_likely_error_file"]
                show_files = error_analysis["show_files"] if "show_files" in error_analysis else []

                msg["most_likely_error_file"] = most_likely_error_file 
                msg["error_type"] = error_type
                # -------------------------------------------------
                # cuda illegal memory
                # -------------------------------------------------
                if error_type == "cuda_illegal_memory":
                    print("⚠ CUDA illegal memory access detected.")
                    ncs_msg = run_ncs_debug(
                        task_root / "spec" / "entry.py",
                        args.device,
                        current_dir / "ncu_log.log"
                    )
                    msg['ncs_msg'] = ncs_msg
                    if ncs_msg.get("errors"):
                        first_error = ncs_msg["errors"][0]
                        problem_kernel_name = first_error.get("kernel", "")
                        problem_kernel_file = first_error.get("source", {}).get("file", "")

                        error_report = validator.generate_error_report_(
                            task_root,
                            current_dir,
                            str(msg),
                            read_file(task_root / "spec" / "entry.py"),
                            problem_kernel_name, 
                            read_file(task_root / "spec" / "kernel" / problem_kernel_name),
                        )


                # -------------------------------------------------
                # value error 
                # -------------------------------------------------
                elif error_type == "value_error":
                    print("⚠ Output mismatch detected.")
                    debug_script = task_root / "spec" / "value_debug.py"
                    if not debug_script.exists():
                        validator.generate_debug_script(
                            task_root,
                            current_dir,
                            read_file("./agent/template/example/value_debug.py"),
                            read_file(task_root / "spec" / "entry.py"),
                            read_file(task_root / "spec" / "ref.py")
                        )
                    debug_msg = run_model_debug(task_root / "spec" / "value_debug.py",args.device)
                    if debug_msg["runnable"]:
                        break
                    msg["debug_msg"] = debug_msg              
                    kernel_report = debug_msg.get("kernel_report", [])
                    problem_kernel_name = debug_msg.get("first_failing_kernel") + ".cu"
                    error_report = validator.generate_error_report_(
                        task_root,
                        current_dir,
                        str(msg),
                        read_file(task_root / "spec" / "entry.py"),
                        problem_kernel_name, 
                        read_file(task_root / "spec" / "kernel" / problem_kernel_name),
                    )
                    
                else:
                    print("⚠ error detected.")
                    error_report = validator.generate_error_report(task_root, 
                                                    current_dir, 
                                                    str(msg), 
                                                    task_description, 
                                                    str(list_all_files(task_root / "spec")),
                                                    load_show_files(show_files))
                    break

            if msg['runnable']:
                break

            kernel_iter += 1
#=========================================== Opti =====================================
        pass

  

            