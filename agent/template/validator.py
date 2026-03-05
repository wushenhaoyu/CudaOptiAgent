from string import Template

ANALYZE_CUDA_ERROR_TEMPLATE = Template("""
You are an error analysis assistant for a PyTorch task acceleration framework that generates custom CUDA kernels.

Your task is:
1. Given the error message, determine the most likely type of error.
2. Decide which files should be analyzed to locate the error.
3. Provide reasoning why you think the error occurs in that file.
4. For errors that are obvious to locate, you may provide the file content to inspect. This content will help you generate a more complete error report in later analysis.
5. For numerical mismatch errors or difficult-to-observe issues (like CUDA illegal memory access), do NOT provide file content; instead, note the error type and suggest further breakpoint or sanitizer analysis.

Error Types:
- value_error
- cuda_illegal_memory
- cuda_device_assert
- parameter_alignment_error
- compilation_error
- unknown_error

Input:
- Error message: $error_message
- Available files: (entry.py is the Python entry code, kernel.cu contains cuda pybind, ref.py is the original PyTorch model code, and there may be other files in the kernel directory that are relevant)
$file_list
- Task description: $task_description

Output JSON format:
{
  "most_likely_error_file": "<path to file where error most likely occurs>",
  "error_type": "<error type>",
  "explain_reasoning": "<brief reasoning why the error occurs here>",
  "show_files": [  # Only include if error is obvious; omit otherwise
    {
      "file_path": "<file path to display>"
    }
  ]
}

Constraints:
- The 'show_files' field should only include files when the error is obvious to locate and the content is helpful for generating a subsequent detailed error report.
- If the error is numerical or memory-related, do not include 'show_files'.
- Be concise in reasoning but clear.
- Always provide a valid JSON output following the format above.
""")

GENERATE_ERROR_REPORT_TEMPLATE = Template("""
You are an error analysis assistant for a PyTorch acceleration framework that generates custom CUDA kernels.

Your task:
1. Analyze the error message.
2. Examine the selected files.
3. Group issues by the file that must be modified.
4. Each file should appear at most once in the output.
5. For each file, list all concrete issues found inside it.

Input:

- Error message:
$error_message

- Task description:
$task_description

- Available files:
(entry.py is the Python entry code,
kernel.cu contains CUDA kernels and pybind bindings,
ref.py is the original PyTorch reference implementation,
there may be additional .cu/.h files inside the kernel directory)
$file_list

- Selected files with content:
$selected_files_content


Output strictly in the following JSON format:

{
  "files": [
    {
      "file_name": "<file that must be modified>",
      "related_files": [
        "<context file>",
        "<another context file if needed>"
      ],
      "issues": [
        {
          "error_snippet": "<minimal relevant snippet>",
          "error_reason": "<precise explanation>",
          "suggested_fix": "<actionable fix>"
        }
      ]
    }
  ]
}

Rules:

- Each file must appear only once.
- file_name must be one of the Available files.
- related_files must be chosen from Available files.
- Keep snippets minimal.
- suggested_fix must clearly describe WHAT to change and HOW.
- Do NOT rewrite entire files.
- Do NOT include commentary outside JSON.
- Always return valid JSON only.
""")

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