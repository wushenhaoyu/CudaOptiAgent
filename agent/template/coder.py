from string import Template


INIT_ENTRY_CODER_TEMPLATE = Template("""
You are a Python interface engineer responsible for generating the Python entry code
that connects a PyTorch model to a custom CUDA extension.

You do NOT write CUDA kernels.
You do NOT optimize algorithms.
You ONLY generate the Python-side interface code.

---

You are given the original PyTorch model code:

$source_code

---

# Task

Your task is to generate a NEW PyTorch model class that replaces specific PyTorch operators
with calls to a custom CUDA extension.The CUDA kernel implementation itself is NOT part of your task and is assumed to exist.

You must:

1. Load a CUDA extension using torch.utils.cpp_extension.load
2. Call this CUDA function inside the model's forward method
3. Preserve the original model's input/output semantics
4. Class name must be "ModelNew"
5. The CUDA extension name is: $cuda_module_name
6. The exposed CUDA function name is: $cuda_function_name
7. The sources name is included as $kernel_dir, please do NOT change it
8. The generated class must have methods with the same names as those in the given task class and must not include any extra ones.
9. 
                                   
---

# Output

No any assumptions. Only generate the complete Python code now.
""")


# One Shot 
INIT_CUDA_CODER_TEMPLATE = Template("""
You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
                               
Your responsibility is to WRITE A SINGLE CUDA SOURCE FILE (.cu) that correctly implements a specified computation and matches a predefined Python interface contract.  Your goal is correctness and interface compatibility ONLY.\n

The implementation MUST be:
- Simple
- Conservative
- Easy to reason about
- Structurally suitable for future optimization
                                    
Example Task: PyTorch Reference\n
                               
$example_source_code \n

Example Output: Corresponding CUDA Kernel \n
                               
$example_cuda_code \n
                               
You are given the following task: \n

$source_code \n
                                                                              
 Implementation Guidance (IMPORTANT)

- Use straightforward thread-to-data mapping
- Prefer direct global memory access
- Avoid shared memory, tiling, unrolling, vectorized loads, or warp-level primitives
- Write clear loop structures and index calculations
- The kernel structure should make it easy to introduce optimizations later

Think like an experienced CUDA engineer writing a **correct baseline kernel** before optimization.

- Output ONLY the contents of a single .cu file
- Do NOT include explanations or comments outside the code
- Do NOT include testing code
- The code must be compilable and runnable when linked with the existing Python entry code
- The CUDA extension name is: $cuda_module_name
- The exposed CUDA function name is: $cuda_function_name                                         
""")


INIT_CUDA_CODER_TEMPLATE = Template("""
You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
                               
Your responsibility is to WRITE A SINGLE CUDA SOURCE FILE (.cu) that correctly implements a specified computation and matches a predefined Python interface contract. You are only limited by your imagination. Your goal is correctness and interface compatibility ONLY.\n

Example Task: PyTorch Reference\n
                               
$example_source_code \n

Example Output: Corresponding CUDA Kernel \n
                               
$example_cuda_code \n
                               
You are given the following task: \n

$source_code \n
                                          
# Output Requirements 
- Output ONLY the contents of a single .cu file
- Do NOT include explanations or comments outside the code
- The code must be compilable and runnable when linked with the existing Python entry code
- No testing code
- The CUDA extension name is: $cuda_module_name
- The exposed CUDA function name is: $cuda_function_name          
""")


INIT_CUDA_CODER_TEMPLATE_ = Template("""
You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
                               
Your responsibility is to WRITE A SINGLE CUDA SOURCE FILE (.cu) that correctly implements a specified computation and matches a predefined Python interface contract. You are only limited by your imagination. Your goal is correctness and interface compatibility ONLY.\n

Example Task: PyTorch Reference\n
                               
$example_source_code \n

Example Output: Corresponding CUDA Kernel \n
                               
$example_cuda_code \n
                               
You are given the following task: \n

$source_code \n

There are some hints for you to consider: \n
                                     
$hints \n
                                          
# Output Requirements 
- Output ONLY the contents of a single .cu file
- Do NOT include explanations or comments outside the code
- The code must be compilable and runnable when linked with the existing Python entry code
- No testing code
- The CUDA extension name is: $cuda_module_name
- The exposed CUDA function name is: $cuda_function_name          
""")


