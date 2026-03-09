import sys
import importlib.util
import torch


def run_script(script_path, device_idx=0):

    torch.cuda.set_device(device_idx)

    spec = importlib.util.spec_from_file_location("target_module", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    Model = getattr(mod, "Model", None)
    get_inputs = getattr(mod, "get_inputs", None)
    get_init_inputs = getattr(mod, "get_init_inputs", None)

    if Model is None or get_inputs is None:
        raise RuntimeError(
            "Script must define Model and get_inputs"
        )

    init_args = []
    init_kwargs = {}

    if callable(get_init_inputs):
        obj = get_init_inputs()

        if isinstance(obj, dict):
            init_kwargs = obj
        elif isinstance(obj, (list, tuple)):
            init_args = list(obj)

    model = Model(*init_args, **init_kwargs).cuda()

    inputs = get_inputs()

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    inputs = [x.cuda() for x in inputs]

    with torch.no_grad():
        model(*inputs)

    torch.cuda.synchronize()


if __name__ == "__main__":

    script = sys.argv[1]
    device = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    run_script(script, device)