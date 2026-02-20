import json
from pathlib import Path
from scripts.test_kernel import ParameterAlignmentError, _test_kernel_process, align_params_generic, try_align_params
from utils.utils import extract_error_report
from ref import Model, get_init_inputs, get_inputs
from entry import ModelNew



if __name__ == "__main__":
    init_inputs = get_init_inputs()
    model_ref = Model(*init_inputs)
    model_test = ModelNew(*init_inputs)
    #print(try_align_params(ref_model=model_ref, test_model=model_test, ref_mod=Model, test_mod=ModelNew))
    try:
        try_align_params(ref_model=model_ref, test_model=model_test, ref_mod=Model, test_mod=ModelNew)
    except ParameterAlignmentError as e:
        print(e)