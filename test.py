import json
from pathlib import Path
from scripts.test_kernel import _test_kernel_process
from utils.utils import extract_error_report






S = """[ERROR_REPORT]
{
  "ERROR_TYPE": "runtime_error",
  "ERROR_FILE": "entry.py",
  "KEY_ERROR_EXCERPT": "AttributeError: '_OpNamespace' 'aten' object has no attribute 'ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU'",
  "ROOT_CAUSE": "The Python code attempts to call the custom operator via torch.ops.aten, but custom operators are not registered in the 'aten' namespace. The operator should be accessed via the extension's own namespace (e.g., torch.ops.ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU)."
}
[/ERROR_REPORT]"""
if __name__ == "__main__":
    a = extract_error_report(S)
    print(a)