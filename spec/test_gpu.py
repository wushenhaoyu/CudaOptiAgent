import os
import re
import time
import torch
from ref import Model, get_init_inputs, get_inputs
from entry import ModelNew

def sanitize_torch_error(err_msg: str, max_lines: int = 6) -> str:
    """
    Sanitize PyTorch error messages for LLM input.

    1. Remove 'Invoked with:' large tensor dumps.
    2. Keep function signatures, traceback, and error type.
    3. Optionally truncate to a maximum number of lines.

    Args:
        err_msg (str): Original error message from PyTorch.
        max_lines (int): Maximum number of lines to keep (after cleanup).

    Returns:
        str: Sanitized, concise error message suitable for LLM.
    """

    lines = err_msg.splitlines()
    sanitized_lines = []
    skip_tensor = False

    for line in lines:
        # Detect start of tensor dump
        if re.match(r'^\s*Invoked with:', line):
            skip_tensor = True
            # Optional: replace with a short summary
            sanitized_lines.append("Invoked with: <tensor content omitted>")
            continue
        # Detect end of tensor dump: a line that does not start with whitespace
        if skip_tensor and not line.startswith(' '):
            skip_tensor = False  # stop skipping, process this line

        if not skip_tensor:
            sanitized_lines.append(line)

    # Optionally truncate if too many lines
    if len(sanitized_lines) > max_lines:
        head = sanitized_lines[:max_lines//2]
        tail = sanitized_lines[-max_lines//2:]
        sanitized_lines = head + ["... (truncated) ..."] + tail

    return "\n".join(sanitized_lines)
 
def write_file(file_path: str, content: str, encoding: str = "utf-8") -> bool:
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:  
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)
        return True

    except Exception as e:
        print(f"Error writing file {file_path}: {e}")
        return False
    
def print_truncated(tensor, name, head=3, tail=3):
    total = tensor.numel()
    if total <= head + tail:
        print(f"{name}:", tensor)
    else:
        head_vals = tensor[:head].tolist()
        tail_vals = tensor[-tail:].tolist()
        print(f"{name}: [{', '.join(f'{v:.4f}' for v in head_vals)} ... {', '.join(f'{v:.4f}' for v in tail_vals)}] (total: {total})")

@torch.no_grad()
def align_params(ref_model, test_model):
    ref_state = ref_model.state_dict()
    test_state = test_model.state_dict()
    
    for name in test_state:
        if name in ref_state and test_state[name].shape == ref_state[name].shape:
            test_state[name].copy_(ref_state[name])
            print(f"  Copied param: {name}, shape: {test_state[name].shape}")
        else:
            print(f"  Skipped param: {name} (not found or shape mismatch)")

if __name__ == '__main__':
    device = torch.device('cuda:4')
    
    init_inputs = get_init_inputs()
    inputs = get_inputs()
    inputs = [x.to(device) for x in inputs]
    
    model = Model(*init_inputs).to(device).eval()
    modelnew = ModelNew(*init_inputs).to(device).eval()
    

    print("Aligning parameters...")
    align_params(model, modelnew)
    

    torch.cuda.synchronize(device)
    

    with torch.inference_mode():
        out_ref = model(*inputs)
        torch.cuda.synchronize(device)
        print("out_ref\n")
        
        try:
            t1 = time.time()
            out_entry = modelnew(*inputs)
            torch.cuda.synchronize(device)  
            print("Time taken by entry model:", time.time() - t1)
            out_entry = out_entry.cpu()
            out_ref = out_ref.cpu()
            diff = (out_ref - out_entry).abs()
            max_err = diff.max().item()
            mean_err = diff.mean().item()
            print(f"max_err:{max_err},mean_err{mean_err}\n")
            
        except Exception as e:
            write_file("error.log", sanitize_torch_error(str(e)))
    