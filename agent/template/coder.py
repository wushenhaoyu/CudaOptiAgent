from string import Template


INIT_ENTRY_CODER_TEMPLATE = Template("""
You are a Python interface engineer responsible for generating the Python entry code
that connects a PyTorch model to a custom CUDA extension.The CUDA kernel implementation itself is NOT part of your task and is assumed to exist.
Here are example PyTorch model code:
                                     
$example_source_code
                                     
Here are exmaple entry code:
                                     
$example_entry_code
                                     
You are given the original PyTorch model code:

$source_code

# Constraints
                                     
You must:
1. Load a CUDA extension using torch.utils.cpp_extension.load
2. Call this CUDA function inside the model's forward method
3. Preserve the original model's input/output semantics
4. Class name must be "ModelNew"
5. The exposed CUDA function name is: $cuda_function_name
6. The sources name is included as $kernel_dir, please do NOT change it
7. The CUDA extension name MUST be derived from the CONTENT HASH of the file at $kernel_dir.You must read the file, compute a hash (e.g. md5 or sha1), and use this hash as part of the namepassed to torch.utils.cpp_extension.load, so that changing kernel.cu automatically triggers recompilation.
8. Variable names in __init__ must MATCH source_code EXACTLY for param alignment to work
                                     
No any assumptions. Only generate the complete Python code now.
""")

REPAIR_ENTRY_CODER_TEMPLATE = Template("""
You are a Python interface engineer responsible for generating the Python entry code
that connects a PyTorch model to a custom CUDA extension.

You are given the original PyTorch model code:

$source_code
                                       
Here is your generated entry code, but it needs to be fixed:

$entry_code
                                       
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
8. Variable names in __init__ must MATCH source_code EXACTLY for param alignment to work
                                       
# Output                               
No any assumptions. Only generate the complete Python code now."""
)

INIT_CUDA_CODER_TEMPLATE = Template("""
You write custom CUDA kernels to replace the PyTorch operators 
in the given architecture to get speedups.

Your responsibility is to WRITE A SINGLE CUDA SOURCE FILE (.cu) 
that correctly implements the specified computation and matches 
a predefined Python interface contract.

Your goal is:
1) Correctness
2) Interface compatibility
3) Respect the provided fusion plan

You must NOT redesign the computation graph.
You must NOT change the fusion grouping.
You must strictly follow the structural constraints.

------------------------------------------------------------
Example Task: PyTorch Reference

$example_source_code

Example Output: Corresponding CUDA Kernel

$example_cuda_code
------------------------------------------------------------

You are given the following task:

$source_code

The entry Python code has already been provided.
You MUST adapt to the entry interface exactly.

$entry_code

------------------------------------------------------------
FUSION PLAN:

$fusion_plan

IMPORTANT:

Fusion groups define logical segmentation only.
They DO NOT strictly define the number of CUDA kernels.

You MAY:
- Reuse the same CUDA kernel implementation for multiple fusion groups
- Generate fewer kernels than fusion groups
- Merge fusion groups into one kernel IF no boundary is violated
- Optimize execution structure

You MUST:
- Respect all fusion boundaries
- Preserve execution order
- Not fuse across forbidden boundaries
- Not introduce illegal intermediate global tensors

Kernel count does NOT need to equal fusion group count.
Correctness and boundary compliance are mandatory.

------------------------------------------------------------
# Output Requirements

- Output ONLY the contents of a single .cu file
- Do NOT include explanations or comments outside the code
- The code must be compilable and runnable when linked with the existing Python entry code
- No testing code
- The CUDA extension name is: $cuda_module_name
- The exposed CUDA function name is: $cuda_function_name
""")

REPAIR_CUDA_CODER_TEMPLATE = Template("""
You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
                               
Your responsibility is to WRITE A SINGLE CUDA SOURCE FILE (.cu) that correctly implements a specified computation and matches a predefined Python interface contract. You are only limited by your imagination. Your goal is correctness and interface compatibility ONLY.\n
                           
PyTorch Task:\n
                                                                 
$source_code \n
                                                                     
Here are given entry code: \n

$entry_code \n
                                     
This is the kernel you generated last time, but it failed to run. \n
                                     
$last_kernel_code \n

There are some hints for you to fix: \n
                                     
$hints \n
                                          
# Output Requirements 
- Output ONLY the contents of a single .cu file
- Do NOT include explanations or comments outside the code
- The code must be compilable and runnable when linked with the existing Python entry code
- No testing code
- The CUDA extension name is: $cuda_module_name
- The exposed CUDA function name is: $cuda_function_name          
""")
