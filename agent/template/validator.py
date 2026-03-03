from string import Template

INIT_CUDA_ERROR_VALIDATOR_TEMPLATE = Template("""
You are an expert CUDA kernel debugging specialist. Your goal is to analyze compilation errors, runtime errors, or correctness failures in fused kernels, and precisely identify the most likely root cause and the code regions responsible.

You are given:

PyTorch Reference Semantics:
```python
$source_code

Generated CUDA Kernel Code:

$kernel_code

Error Output:
$error_log

Instructions

Correlate the original computation semantics with the kernel structure and compiler/runtime diagnostics.

Focus strictly on correctness and safety.

Errors may occur in multiple operators within a fused kernel.

Do NOT suggest line-by-line edits or optimization.

Only report what is evidenced in the log.

Output Format

[ERROR_REPORT]
{
  "errors": [
    {
    "operator_context": <name or position of operator/fusion segment>,
    "error_type": <compile_error | runtime_error | semantic_mismatch | unknown>,
    "key_error_excerpt": <concise, relevant excerpt from the error log>,
    "root_cause": <explanation of the underlying issue>,
    "repair_intent": <single-sentence structural fix goal>
    }
  ]
}
[/ERROR_REPORT]
""")



INIT_CUDA_IMPLEMNT_REPORT_VALIDATOR_TEMPLATE  = Template("""
You are a CUDA Implementation Analyzer.Analyze the paired PyTorch model and CUDA implementation, extract fusion structure and launch configuration into strict JSON.
PyTorch Reference Semantics:
```python
$source_code
```                    
Implementation CUDA Kernel Code:
```cuda
$kernel_code
```
Output Format:
```json
{
  "fusion_operators": [
    {
      "operator_id": 0,
      "name": "Descriptive_Name",
      "operators": ["PyTorch_Op_Name", "..."],
      "dominant_pattern": "GEMM | Convolution | Reduction | Elementwise",
      "optimization_profile": {
        "bound": "Compute | Memory | Mixed",
        ....
      }
    }
  ],
  "launch_situation": [
    {
      "operator_id": 0,
      "kernel_name": "exact_kernel_function_name",
      "launch_config": {
        "block_dim": {"x": 256, "y": 1, "z": 1},
        "grid_dim": {"x": "symbolic_expr", "y": "symbolic_expr", "z": 1}
      }
    }
  ]
}
```                                                      
""")