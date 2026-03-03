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

Preserve the original model's input/output semantics

Class name must be "ModelNew"

Sources directory is $kernel_dir; do NOT change it

Variable names in __init__ must match $source_code exactly for param alignment

Generate one CUDA function per fusion group/operator

Forward must call the corresponding CUDA functions in order, reusing the same kernel for repeated operators if indicated

Output Requirements

Generate the complete Python code for the ModelNew class

Forward should sequentially invoke the CUDA kernels from $fusion_plan

No explanations or extra comments outside code

CUDA extension names must be derived from the content hash of each kernel file
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
1. Load a CUDA extension using torch.utils.cpp_extension.load
2. Call this CUDA function inside the model's forward method
3. Preserve the original model's input/output semantics
4. Class name must be "ModelNew"
5. The exposed CUDA function name is: $cuda_function_name
6. The sources name is included as $kernel_dir, please do NOT change it
7. The CUDA extension name MUST be derived from the CONTENT HASH of the file at $kernel_dir.You must read the file, compute a hash (e.g. md5 or sha1), and use this hash as part of the namepassed to torch.utils.cpp_extension.load, so that changing kernel.cu automatically triggers recompilation.
               
# Output                               
No any assumptions. Only generate the complete Python code now."""
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
You are a CUDA kernel developer tasked with fixing a previously generated CUDA kernel that failed to run. 

REQUIREMENTS:

1. Do NOT use PyTorch, ATen, Thrust, or any external tensor library.
2. All computations MUST be implemented explicitly using CUDA device code (__global__, __device__).
3. Input and output are plain device pointers (float*, __half*, etc.), not tensors.
4. Respect the fusion plan strictly. Do NOT fuse across forbidden boundaries.
5. Preserve execution order. Do NOT redesign the computation graph.
6. You may reuse the same kernel for multiple fusion groups if boundaries allow.
7. Focus strictly on correctness and interface compatibility. Do NOT optimize performance yet.

INPUTS:

- Original Python source code:
```python
$source_code

Python entry interface:

$entry_code

Last failed CUDA kernel:

$last_kernel_code

Hints for fixing:
$hints


OUTPUT:

A single .cu file implementing the corrected kernels.

Output ONLY the .cu file contents; no explanations or comments outside the code.

Must compile and run with the provided Python entry code.

No testing code included.

CUDA module name: $cuda_module_name

CUDA function name: $cuda_function_name

All computations must be implemented inside this .cu file using CUDA device code only.
""")