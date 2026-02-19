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
