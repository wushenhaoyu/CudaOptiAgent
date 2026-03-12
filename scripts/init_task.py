import json
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
from utils.utils import  delete_folder, copy_folder, dict_to_text, find_best_match, list_all_files, load_show_files, read_file, remove_justification, text_to_dict, write_file, extract_recommendation
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
        
        
        max_iter = -1
        for item in bootstrap.iterdir():
            if item.is_dir() and item.name.startswith("iter_"):
                try:
                    iter_num = int(item.name.split("_")[1])
                    max_iter = max(max_iter, iter_num)
                except (ValueError, IndexError):
                    continue
        
        if max_iter >= 0:
            
            last_iter_dir = bootstrap / f"iter_{max_iter}"
            error_report_file = last_iter_dir / "error_report.json"
            
            if error_report_file.exists():
               
                kernel_iter = max_iter + 1
                print(f"Resuming: found error_report in iter_{max_iter}, starting from iter_{kernel_iter}")
            else:
                result_file = last_iter_dir / "result.json"
                if result_file.exists():
                    result = json.loads(read_file(result_file))
                    if result.get("runnable", False):
                        print(f"Task already completed successfully at iter_{max_iter}")
                        continue
                    else:
                        kernel_iter = max_iter
                        print(f"Resuming: iter_{max_iter} has result but not passed, retrying")
                else:
                    kernel_iter = max_iter
                    print(f"Resuming: no result in iter_{max_iter}, starting from there")
        
        while kernel_iter < args.bootstrap_iter:

            print(f"\n=== Kernel Iteration {kernel_iter} ===")

            current_dir = bootstrap / f"iter_{kernel_iter}"
            current_dir.mkdir(parents=True, exist_ok=True)

            # Load error_report from previous iteration if kernel_iter > 0
            error_report = None
            if kernel_iter > 0:
                prev_iter_dir = bootstrap / f"iter_{kernel_iter - 1}"
                prev_error_report_file = prev_iter_dir / "error_report.json"
                if prev_error_report_file.exists():
                    error_report = json.loads(read_file(prev_error_report_file))
                    print(f"Loaded error_report from iter_{kernel_iter - 1}")

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
                # error_report is now loaded from previous iter
                if error_report:
                    repair_file_list = error_report["files"]
                    if not (current_dir / "kernel").exists():
                        coder.repair_init_cuda_code(
                            root_dir=task_root,
                            current_dir=current_dir,
                            file_list=str(list_all_files(task_root / "spec")),
                            repair_file_list=repair_file_list
                        )
                        copy_folder(task_root / "spec" / "kernel" , current_dir / "kernel")
                else:
                    print(f"Warning: No error_report found for iter_{kernel_iter - 1}, cannot repair")
                    # Optionally break or handle this case
                    break
            
            # Inner loop for testing and error fixing
            test_passed = False
            while True:
                # Check if we already have a result for this iteration
                result_file = current_dir / "result.json"
                if result_file.exists():
                    msg = json.loads(read_file(result_file))
                    print(f"Loaded existing result: runnable={msg.get('runnable', False)}")
                else:
                    msg = test_kernel(task_root, current_dir, args.device)
                    write_file(result_file, json.dumps(msg, indent=2))

                if msg['runnable']:
                    print("Success")
                    test_passed = True
                    error_report = None
                    break

                # Check for existing error analysis
                error_analysis_file = current_dir / "error_analysis.json"
                if error_analysis_file.exists():
                    error_analysis = json.loads(read_file(error_analysis_file))
                else:
                    error_analysis = validator.analyze_init_error(
                        task_root,
                        current_dir,
                        read_file(current_dir / "result.json"),
                        str(list_all_files(task_root / "spec")),
                        task_description
                    )
                    write_file(error_analysis_file, json.dumps(error_analysis, indent=2))

                error_type = error_analysis["error_type"]
                most_likely_error_file = error_analysis["most_likely_error_file"]
                show_files = error_analysis.get("show_files", [])

                msg["most_likely_error_file"] = most_likely_error_file 
                msg["error_type"] = error_type
                
                # -------------------------------------------------
                # cuda illegal memory
                # -------------------------------------------------
                if error_type == "cuda_illegal_memory":
                    print("CUDA illegal memory access detected.")
                    
                    # Check for existing NCS results
                    ncs_parsed_file = current_dir / "ncs_parsed.json"
                    if ncs_parsed_file.exists():
                        with open(ncs_parsed_file, 'r') as f:
                            ncs_parsed = json.load(f)
                        ncs_msg = {"parsed": ncs_parsed}
                    else:
                        ncs_msg = run_ncs_debug(
                            task_root / "spec" / "entry.py",
                            args.device,
                            current_dir / "ncu_log.json"
                        )
                    
                    msg['ncs_msg'] = ncs_msg.get('parsed', {})
                    if not msg['ncs_msg'].get("success", False):
                        first_error = msg['ncs_msg']["errors"][0]
                        problem_kernel = first_error["kernel"]
                        problem_kernel_name = find_best_match(problem_kernel, list_all_files(task_root / "spec"))

                        # Check for existing error report
                        error_report_file = current_dir / "error_report.json"
                        if error_report_file.exists():
                            error_report = json.loads(read_file(error_report_file))
                        else:
                            error_report = validator.generate_error_report_(
                                task_root,
                                current_dir,
                                str(msg),
                                task_description,
                                list_all_files(task_root / "spec"),
                                read_file(task_root / "spec" / "entry.py"),
                                problem_kernel_name, 
                                read_file(task_root / "spec" / problem_kernel_name),
                            )
                            write_file(error_report_file, json.dumps(error_report, indent=2))
                    break


                # -------------------------------------------------
                # value error 
                # -------------------------------------------------
                elif error_type == "result_error":
                    print("Output mismatch detected.")
                    debug_script = task_root / "spec" / "value_debug.py"
                    
                    # Check for existing debug results
                    debug_result_file = current_dir / "result_debug.json"
                    if debug_script.exists() and debug_result_file.exists():
                        debug_msg = json.loads(read_file(debug_result_file))
                    else:
                        if not debug_script.exists():
                            validator.generate_debug_script(
                                task_root,
                                current_dir,
                                read_file("./agent/template/example/value_debug.py"),
                                read_file(task_root / "spec" / "entry.py"),
                                read_file(task_root / "spec" / "ref.py")
                            )
                        debug_msg = run_model_debug(task_root / "spec" / "value_debug.py", args.device)
                        write_file(debug_result_file, json.dumps(debug_msg, indent=2))
                    
                    if debug_msg.get("runnable", False):
                        test_passed = True
                        error_report = None
                        break
                    
                    msg["debug_msg"] = debug_msg              
                    kernel_reports = debug_msg.get("kernel_report", [])
                    for kernel_report in kernel_reports:
                        if kernel_report.get('status') != "ok":
                            problem_kernel_name = kernel_report["kernel"] + ".cu"
                            
                            # Check for existing error report
                            error_report_file = current_dir / "error_report.json"
                            if error_report_file.exists():
                                error_report = json.loads(read_file(error_report_file))
                            else:
                                error_report = validator.generate_error_report_(
                                    task_root,
                                    current_dir,
                                    str(msg),
                                    task_description,
                                    str(list_all_files(task_root / "spec")),
                                    read_file(task_root / "spec" / "entry.py"),
                                    problem_kernel_name, 
                                    read_file(task_root / "spec" / "kernel" / problem_kernel_name),
                                )
                                write_file(error_report_file, json.dumps(error_report, indent=2))
                            break
                    break
                    
                else:
                    print("Error detected.")
                    
                    # Check for existing error report
                    error_report_file = current_dir / "error_report.json"
                    if error_report_file.exists():
                        error_report = json.loads(read_file(error_report_file))
                    else:
                        error_report = validator.generate_error_report(
                            task_root, 
                            current_dir, 
                            str(msg), 
                            task_description, 
                            str(list_all_files(task_root / "spec")),
                            load_show_files(show_files)
                        )
                        write_file(error_report_file, json.dumps(error_report, indent=2))
                    break

            if test_passed:
                break

            kernel_iter += 1
            
#=========================================== Opti =====================================
        pass
            
#=========================================== Opti =====================================
        pass
#def init_task(tasks: List[Path], run_dir: Path, args: Dict):
#    import ast
#    
#    analyzer = Analyzer(args=args)
#    coder = Coder(args=args)
#    validator = Validator(args=args)
#    planner = Planner(args=args)
#
#    for task in tasks:
#        task_name = task.stem
#        task_name_no_num = task_name.split('_', 1)[-1]
#        task_root = (run_dir / task.parent.name / task_name).resolve()
#
#        task_root.mkdir(parents=True, exist_ok=True)
#        (task_root / "spec").mkdir(parents=True, exist_ok=True) 
#        (task_root / "bootstrap").mkdir(parents=True, exist_ok=True) 
#        (task_root / "optimize").mkdir(parents=True, exist_ok=True) 
#        
#            
#        if not (task_root / "spec" / "ref.py").exists():
#            shutil.copy2(task, task_root / "spec" / "ref.py")
#
#        plan_file = task_root / "bootstrap" / "fusion_plan.json"
#        plan = None
#
#        if not plan_file.exists():
#            plan = analyzer.gernerate_fuse_operator_plan(task_root, read_file(task))
#        else:
#            plan = text_to_dict(read_file(plan_file))
#        plan = remove_justification(plan)
#        task_description = plan["task_description"]
#        if not (task_root / "spec" / "entry.py").exists():
#            coder.generate_entry_code(task_root, 
#                                      str(plan),
#                                      read_file("./agent/template/example/example.py"), 
#                                      read_file("./agent/template/example/example_entry.py"), 
#                                      read_file(task), 
#                                      task_name_no_num, 
#                                      task_name_no_num, 
#                                      str(task_root / "spec" / "kernel" / "kernel.cu"))
#            
#
#        bootstrap = Path(task_root / "bootstrap")
#        error_report = None
#        impl_report  = None
#        msg = None
#        bootstrap_final = task_root / "bootstrap" / "kernel.cu"
#        kernel_iter = 0
#
#        while kernel_iter < args.bootstrap_iter:
#
#            print(f"\n=== Kernel Iteration {kernel_iter} ===")
#
#            current_dir = bootstrap / f"iter_{kernel_iter}"
#            current_dir.mkdir(parents=True, exist_ok=True)
#
#
#            if kernel_iter == 0:
#                if not (current_dir / "kernel").exists():
#                    coder.gernerate_init_cuda_code(
#                        current_dir,
#                        read_file('./agent/template/example/example.py'),
#                        read_file('./agent/template/example/example.cu'),
#                        read_file(task),
#                        read_file(task_root / "spec" / "entry.py"),
#                        str(plan),
#                        task_name_no_num,
#                        task_name_no_num
#                    )
#                    copy_folder(current_dir / "kernel", task_root / "spec" / "kernel")
#            else:
#                repair_file_list = error_report["files"]
#                coder.repair_init_cuda_code(
#                    root_dir=task_root,
#                    current_dir=current_dir,
#                    file_list=str(list_all_files(task_root / "spec")),
#                    repair_file_list=repair_file_list
#                )
#                copy_folder(task_root / "spec" / "kernel" , current_dir / "kernel")
#            while True:
#
#                msg = test_kernel(task_root, current_dir, args.device)
#                write_file(current_dir / "result.json", json.dumps(msg, indent=2))
#
#                if msg['runnable']:
#                    print("✅ Success")
#                    break
#
#                error_analysis = validator.analyze_init_error(
#                    task_root,
#                    current_dir,
#                    read_file(current_dir / "result.log"),
#                    str(list_all_files(task_root / "spec")),
#                    task_description
#                )
#
#                error_type = error_analysis["error_type"]
#                most_likely_error_file = error_analysis["most_likely_error_file"]
#                show_files = error_analysis["show_files"] if "show_files" in error_analysis else []
#
#                msg["most_likely_error_file"] = most_likely_error_file 
#                msg["error_type"] = error_type
#                # -------------------------------------------------
#                # cuda illegal memory
#                # -------------------------------------------------
#                if error_type == "cuda_illegal_memory":
#                    print("⚠ CUDA illegal memory access detected.")
#                    ncs_msg = run_ncs_debug(
#                        task_root / "spec" / "entry.py",
#                        args.device,
#                        current_dir / "ncu_log.log"
#                    )
#                    msg['ncs_msg'] = ncs_msg['parsed']
#                    if not msg['ncs_msg']["success"]:
#                        first_error = msg['ncs_msg']["errors"][0]
#                        problem_kernel = first_error["kernel"]
#                        #problem_kernel_file = first_error.get("source", {}).get("file", "")
#                        problem_kernel_name = find_best_match(problem_kernel,list_all_files(task_root / "spec"))
#
#                        error_report = validator.generate_error_report_(
#                            task_root,
#                            current_dir,
#                            str(msg),
#                            task_description,
#                            list_all_files(task_root / "spec"),
#                            read_file(task_root / "spec" / "entry.py"),
#                            problem_kernel_name, 
#                            read_file(task_root / "spec" / problem_kernel_name),
#                        )
#                    break
#
#
#                # -------------------------------------------------
#                # value error 
#                # -------------------------------------------------
#                elif error_type == "result_error":
#                    print("⚠ Output mismatch detected.")
#                    debug_script = task_root / "spec" / "value_debug.py"
#                    if not debug_script.exists():
#                        validator.generate_debug_script(
#                            task_root,
#                            current_dir,
#                            read_file("./agent/template/example/value_debug.py"),
#                            read_file(task_root / "spec" / "entry.py"),
#                            read_file(task_root / "spec" / "ref.py")
#                        )
#                    debug_msg = run_model_debug(task_root / "spec" / "value_debug.py",args.device)
#                    write_file(current_dir / "result_debug.json", json.dumps(debug_msg, indent=2))
#                    if debug_msg["runnable"]:
#                        break
#                    msg["debug_msg"] = debug_msg              
#                    kernel_reports = debug_msg["kernel_report"]
#                    for kernel_report in kernel_reports:
#                        if kernel_report['status'] != "ok":
#                            problem_kernel_name = kernel_report["kernel"] + ".cu"
#                            error_report = validator.generate_error_report_(
#                                task_root,
#                                current_dir,
#                                str(msg),
#                                task_description,
#                                str(list_all_files(task_root / "spec")),
#                                read_file(task_root / "spec" / "entry.py"),
#                                problem_kernel_name, 
#                                read_file(task_root / "spec" / "kernel" / problem_kernel_name),
#                            )
#                            break
#                    break
#                    
#                else:
#                    print("⚠ error detected.")
#                    error_report = validator.generate_error_report(task_root, 
#                                                    current_dir, 
#                                                    str(msg), 
#                                                    task_description, 
#                                                    str(list_all_files(task_root / "spec")),
#                                                    load_show_files(show_files))
#                    break
#
#            if msg['runnable']:
#                break
#
#            kernel_iter += 1
##=========================================== Opti =====================================
#        pass

  

            