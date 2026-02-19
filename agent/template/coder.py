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
8. Reduce computation for CPU testing by modifying ALL variables affecting tensor sizes (global vars like N/batch_size/channels/dims AND get_init_inputs() return values). Keep tensors small (e.g., <1000 elements).
                      
# Output                               
No any assumptions. Only generate the complete Python code now."""
)

INIT_CUDA_CODER_TEMPLATE = Template("""
You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
                               
Your responsibility is to WRITE A SINGLE CUDA SOURCE FILE (.cu) that correctly implements a specified computation and matches a predefined Python interface contract. You are only limited by your imagination. Your goal is correctness and interface compatibility ONLY.\n

Example Task: PyTorch Reference\n
                               
$example_source_code \n

Example Output: Corresponding CUDA Kernel \n
                               
$example_cuda_code \n
                               
You are given the following task: \n

$source_code \n
                                    
the entry python code has alreadt provided for you, you need to adapt to the entry python code.
                                    
$entry_code \n
                                          
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
                               
Here are correct cpu implements: \n

$cpu_code \n
                                     
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


REPAIR_CUDA_CODER_TEMPLATE_ = Template("""
You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
                               
Your responsibility is to WRITE A SINGLE CUDA SOURCE FILE (.cu) that correctly implements a specified computation and matches a predefined Python interface contract. You are only limited by your imagination. Your goal is correctness and interface compatibility ONLY.\n
                               
Here are given Pytorch Task: \n

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
- Parameters must linked with the given Pytorch Task, entry code parameters are wrong.      
""")

INIT_CPU_CODER_TEMPLATE = Template("""
You are a CUDA C code generator.Generate a COMPLETE and COMPILABLE .cu file that implements the same computation as the given PyTorch code, and matches the Python entry interface exactly.
The file will be compiled by NVCC but must run on CPU only.Correctness is the only goal.
## PyTorch Code
$source_code

## Python Entry Interface
$entry_code
# Constraints
1. Pure sequential CPU code.
2. Must be a complete .cu file.
3. Exposed functions must match entry.py exactly.
4. Only use standard C/C++ (e.g. math.h).
5. No parallel or external libraries.
6. Needs to accommodate both Python Entry Interface parameters and Pytorch Code Interface.
                                   
# Output
- Output only the .cu source file.
- No explanations or extra text.
- No main function unless required.
""")


REPAIR_CPU_CODER_TEMPLATE = Template("""
You are a CUDA C repair agent.

Generate a NEW COMPLETE .cu file
that fixes the previously generated CPU backend
while preserving its structure and interface.

The file will be compiled by NVCC but must run on CPU only.

Correctness and interface compatibility are the only goals.

---

# Inputs

## PyTorch Semantics
$source_code

## Broken CPU Backend
$last_cpu_code

## Repair Hints
$hints

---

# Task

Produce a corrected version based on the previous code.
Apply only the necessary fixes from the hints.
Do NOT redesign the algorithm or change the interface.

---

# Constraints

1. Must be a complete .cu file.
2. Pure sequential CPU code (no CUDA APIs, no parallelism).
3. Exposed functions must match entry.py exactly.
4. Only use standard C/C++ (e.g. math.h).
5. Preserve overall structure of the previous code.
6. Apply minimal and conservative changes only.
7. Needs to accommodate both Python Entry Interface parameters and Pytorch Code Interface.
# Output
- Output only the .cu source file.
- No explanations or extra text.
- No test code.
- No main function unless required.
""")

RESTORE_ENTRY_CODE_TEMPLATE = Template("""
Restore entry.py parameters to match ref.py values.
#ref.py:
$source_code
#entry.py (current):
$entry_code
# Output
- Only fix the parameter values (batch_size, channels, dimensions, etc.). Keep all other code unchanged. Return complete corrected entry.py.
- Output only the .py file
- No explanations or extra text.
- No test code.
- Ban to use pytorch kernels.
""")

