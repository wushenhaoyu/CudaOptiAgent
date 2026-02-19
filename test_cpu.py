import time

import torch
from ref import Model, get_init_inputs, get_inputs
from entry import ModelNew

if __name__ == '__main__':
    init_inputs = get_init_inputs()
    inputs = get_inputs()
    model = Model(*init_inputs)
    modelnew = ModelNew(*init_inputs)
    inputs = [x.cuda() for x in inputs]
    model = model.cuda()
    out_ref = model(*inputs)
    inputs = [x.cpu() for x in inputs]
    out_entry = modelnew(*inputs)
    out_ref = out_ref.cpu()
    print(torch.allclose(out_ref, out_entry, atol=1e-4, rtol=1e-4))

