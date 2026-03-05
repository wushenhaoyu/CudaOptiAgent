from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
from typing import List, Dict, Tuple

def run_model_debug_script(script_path: Path, device_idx: int = 0) -> Tuple[str, List[Dict]]:

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

    model = ModelDebug(*init_args, **init_kwargs).to(device)

    inputs = get_inputs()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    inputs = [x.to(device) for x in inputs]

    first_failing_kernel = None
    try:
        _ = model(inputs[0])
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("FIRST_KERNEL_MISMATCH"):
            first_failing_kernel = msg.split(":")[1].strip()
        else:
            raise e

    return first_failing_kernel, model.report