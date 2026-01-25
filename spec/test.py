import time
from ref import Model, get_init_inputs, get_inputs
from entry import ModelNew

if __name__ == '__main__':
    init_inputs = get_init_inputs()
    inputs = get_inputs()
    model = Model(*init_inputs)
    modelnew = ModelNew(*init_inputs)
    out_ref = model(*inputs)
    t1 = time.time()
    out_entry = modelnew(*inputs)
    print("Time taken by entry model:", time.time() - t1)
    print(out_entry == out_ref)

