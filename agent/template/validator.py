from string import Template

INIT_CUDA_ERROR_VALIDATOR_TEMPLATE = Template("""
You are an expert CUDA kernel debugging specialist.Your goal is to analyze compilation errors, runtime errors, or correctness failures, and precisely identify the most likely root cause and the code region responsible.

You are given the following information:

PyTorch Reference Semantics:
```python
$source_code
```                    
Generated CUDA Kernel Code:
```cuda
$kernel_code
```
error Output:

$error_log

# Instructions

You must internally reason about the failure by correlating:
- The original computation semantics
- The kernel structure
- Compiler diagnostics or runtime error messages
Your task is to LOCALIZE the error and CLASSIFY it.
- Do NOT suggest specific line-by-line edits.
- Do NOT mention performance or optimization.
- Focus strictly on correctness and safety.
- Do NOT invent or assume errors; report only what is evidenced.

# Output Format

[ERROR_REPORT]
{
  "ERROR_TYPE": <compile_error | runtime_error | semantic_mismatch | unknown>,
  "KEY_ERROR_EXCERPT": <concise, relevant excerpt from the error log>,
  "ROOT_CAUSE": <clear explanation of the underlying issue>,
  "REPAIR_INTENT": <single-sentence structural fix goal>,   
  "MODIFICATION_GUIDANCE":<primary actionable change for the next kernel generation>
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