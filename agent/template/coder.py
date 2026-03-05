from string import Template


INIT_ENTRY_CODER_TEMPLATE = Template("""
You are a Python interface engineer responsible for generating the Python entry code
that connects a PyTorch model to a custom CUDA extension. The CUDA kernel implementations themselves are assumed to exist.

You are given:

1. Original PyTorch model code:
```python
$source_code
```
Fusion plan (list of kernels / operators to implement):
```json
$fusion_plan
```
Example PyTorch model code:
```python
$example_source_code
```
Example entry code:
```python
$example_entry_code
```                                     
Constraints

You must:
- Preserve the original model's input/output semantics
- Class name must be "ModelNew"
- Sources directory is $kernel_dir; do NOT change it
- Variable names in __init__ must match $source_code exactly for param alignment
- Generate one CUDA function per fusion group/operator
- Forward must call the corresponding CUDA functions in order, reusing the same kernel for repeated operators if indicated
                                     
Output Requirements:
- Generate the complete Python code for the ModelNew class
- Forward should sequentially invoke the CUDA kernels from $fusion_plan
- No explanations or extra comments outside code
- CUDA extension names must be derived from the content hash of each kernel file
""")

REPAIR_ENTRY_CODER_TEMPLATE = Template("""
You are a Python interface engineer responsible for generating the Python entry code
that connects a PyTorch model to a custom CUDA extension.

You are given the original PyTorch model code:
```python
$source_code
```                                      
Here is your generated entry code, but it needs to be fixed:
```python
$entry_code
```                                       
Here is some useful information for you:
                                       
$error_report
                                       
You must:
- Preserve the original model's input/output semantics
- Class name must be "ModelNew"
- Sources directory is $kernel_dir; do NOT change it
- Variable names in __init__ must match $source_code exactly for param alignment
- Generate one CUDA function per fusion group/operator
- Forward must call the corresponding CUDA functions in order, reusing the same kernel for repeated operators if indicated
                                     
Output Requirements:
- Generate the complete Python code for the ModelNew class
- Forward should sequentially invoke the CUDA kernels from $fusion_plan
- No explanations or extra comments outside code
- CUDA extension names must be derived from the content hash of each kernel file"""
)

INIT_CUDA_CODER_TEMPLATE = Template("""
You are a CUDA kernel developer. Your goal is to implement **inference-only** kernels for a given model.

REQUIREMENTS:
1. Do NOT use PyTorch, ATen, Thrust, or any external tensor library.
2. All computations MUST be implemented using CUDA device code (__global__, __device__).
3. Input/output are plain device pointers (float*, __half*, etc.).
4. Respect the fusion plan strictly; do NOT fuse across forbidden boundaries.
5. Preserve execution order; do NOT redesign the computation graph.
6. Each fusion group in $fusion_plan must become a separate kernel with a pybind11 wrapper.
7. Generate one unified entry file `kernel.cu` that:
   - Includes all fusion group .cu files
   - Registers a Python module with pybind11
   - Exposes one function per fusion group as Python callable

INPUTS:
- Python entry code passes raw device pointers and sizes
- Fusion plan is provided as JSON in $fusion_plan

OUTPUT:
- Multiple `.cu` files, one per fusion group
- One `kernel.cu` file including all fusion groups and registering the Python module
- Each fusion group must have:
    - __global__ kernel
    - Python wrapper via pybind11

# Source code for reference:
```python
$source_code
Python entry interface:
$entry_code
Fusion Plan:
$fusion_plan
Output Format Instructions:

Do NOT include explanations or comments outside code

Generate complete CUDA source files ready for compilation

Each fusion group .cu file should be named {kernel_name}.cu

kernel.cu should include all fusion group files and register them with pybind11

Each wrapper function name matches kernel_name for Python calls
""")

REPAIR_CUDA_CODER_TEMPLATE = Template("""
You are a CUDA/PyTorch integration engineer responsible for fixing ONE specific file in a multi-file acceleration project.

Your goal is to fix the target file so that the previously detected errors are resolved.

You are ONLY allowed to modify the target file.
You MUST NOT modify or redesign other files.

----------------------------------------------------------------------
PROJECT CONTEXT
----------------------------------------------------------------------

Available files:
- entry.py: Python entry code
- kernel.cu: CUDA implementation + pybind binding
- ref.py: original PyTorch reference model
- There may be other auxiliary files.

All file names:
$file_list

----------------------------------------------------------------------
TARGET FILE TO MODIFY
----------------------------------------------------------------------

Target file:
$target_file_name

Current content of target file:
```python
$target_file_content
RELATED FILES (read-only)

These files are provided for interface and logic reference.
You must not modify them.

$related_files_content

ERRORS RELATED TO THIS FILE

$error_items

REQUIREMENTS

Fix ONLY issues that belong to the target file.

Do NOT redesign architecture.

Do NOT change function signatures unless required by error description.

If function signature is changed, it MUST stay compatible with related files.

Keep CUDA kernel launch, pybind definitions, and entry interface consistent.

Focus strictly on correctness.

Output the FULL corrected file content.

OUTPUT FORMAT

Return ONLY the full corrected content of:

$target_file_name

No explanations.
No JSON.
No markdown fences.
No extra commentary.
""")