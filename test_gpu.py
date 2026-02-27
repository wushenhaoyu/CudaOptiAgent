import time
import torch
from ref import Model, get_init_inputs, get_inputs
from entry import ModelNew

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
    device = torch.device('cuda:1')
    
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
        
        t1 = time.time()
        out_entry = modelnew(*inputs)
        torch.cuda.synchronize(device)  
        print("Time taken by entry model:", time.time() - t1)
    

    out_ref = out_ref.cpu().flatten()
    out_entry = out_entry.cpu().flatten()
    
    print_truncated(out_ref, "ref output")
    print_truncated(out_entry, "entry output")
    
    diff = (out_ref - out_entry).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    
    print(f"max_abs_err={max_err:.3e}, mean_abs_err={mean_err:.3e}")
    print("allclose (1e-3):", torch.allclose(out_ref, out_entry, atol=1e-3, rtol=1e-3))
  

#if __name__ == "__main__":
#    device = torch.device('cuda:1')
#    init_inputs = get_init_inputs()
#    inputs = get_inputs()
#    inputs = [x.to(device) for x in inputs]
#    modelnew = ModelNew(*init_inputs).to(device).eval()
#    with torch.inference_mode():
#        out_entry = modelnew(*inputs)
#        print("out_entry\n")