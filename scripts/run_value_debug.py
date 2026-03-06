from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
from typing import List, Dict, Tuple
from multiprocessing import get_context

def run_model_debug(script_path: Path, device_idx: int = 0):

    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    p = ctx.Process(
        target=run_model_debug_process,
        args=(script_path, device_idx, child_conn),
    )

    p.start()

    try:
        child_conn.close()
    except Exception:
        pass

    p.join()

    if p.exitcode != 0 and not parent_conn.poll():
        return {
            "runnable": False,
            "error_type": "process_crash",
            "message": f"Subprocess exited with code {p.exitcode}",
        }

    payload = parent_conn.recv() if parent_conn.poll() else None

    try:
        parent_conn.close()
    except Exception:
        pass

    if (
        isinstance(payload, tuple)
        and len(payload) == 2
        and payload[0] in ("ok", "err")
    ):
        tag, data = payload

        if tag == "ok":
            result = {"runnable": True}
            result.update(data)
        else:
            result = {"runnable": False}
            result.update(data)

    else:
        result = {"runnable": False}

    return result

def run_model_debug_process(script_path: Path, device_idx: int, conn):
    import torch
    import traceback

    try:
        torch.cuda.set_device(device_idx)

        first_kernel, report = _run_model_debug_script(
            script_path,
            device_idx,
        )

        # ---------------- 所有 kernel OK ----------------
        if first_kernel is None:

            conn.send((
                "ok",
                {}   # 不返回 report
            ))

        # ---------------- 存在 kernel mismatch ----------------
        else:

            conn.send((
                "err",
                {
                    "error_type": "kernel_output_mismatch",
                    "kernel_report": report
                }
            ))

    except RuntimeError as e:

        conn.send((
            "err",
            {
                "error_type": "runtime_error",
                "message": str(e)
            }
        ))

    except Exception:

        conn.send((
            "err",
            {
                "error_type": "unknown_error",
                "message": traceback.format_exc()
            }
        ))

def _run_model_debug_script(script_path: Path, device_idx: int = 0) -> Tuple[str, List[Dict]]:

    import torch

    spec = spec_from_file_location(script_path.stem, str(script_path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    ModelDebug = getattr(mod, "ModelDebug", None)
    get_inputs = getattr(mod, "get_inputs", None)
    get_init_inputs = getattr(mod, "get_init_inputs", None) 

    if ModelDebug is None or get_inputs is None:
        raise RuntimeError(f"{script_path} must define ModelDebug class and get_inputs() function")

    init_args: List = []
    init_kwargs: Dict = {}
    if callable(get_init_inputs):
        init_obj = get_init_inputs()
        if isinstance(init_obj, dict):
            init_kwargs = init_obj
        elif isinstance(init_obj, (list, tuple)):
            init_args = list(init_obj)
        elif init_obj is not None:
            raise TypeError("get_init_inputs() must return list/tuple (*args) or dict (**kwargs)")

    device = torch.device(f"cuda:{device_idx}")

    model = ModelDebug(*init_args, **init_kwargs).to(device).eval()

    inputs = get_inputs()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    inputs = [x.to(device) for x in inputs]

    first_failing_kernel = None

    try:
        _ = model(*inputs)

    except RuntimeError as e:
        msg = str(e)

        if msg.startswith("FIRST_KERNEL_MISMATCH"):
            first_failing_kernel = msg.split(":")[1].strip()
        else:
            raise

    return first_failing_kernel, model.report