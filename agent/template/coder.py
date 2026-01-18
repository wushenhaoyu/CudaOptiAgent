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
with calls to a custom CUDA extension.

You must:

1. Load a CUDA extension using torch.utils.cpp_extension.load_inline
2. Assume the CUDA extension exposes a function with a known name
3. Call this CUDA function inside the model's forward method
4. Preserve the original model's input/output semantics
5. Class name must be "NewModel"
The CUDA kernel implementation itself is NOT part of your task and is assumed to exist.

---

- The CUDA extension name is: $cuda_module_name
- The exposed CUDA function name is: $cuda_function_name
- The cuda_sources name is: kernel.cu
- The CUDA function signature matches the original operator semantics
- The CUDA extension compiles successfully when the correct CUDA code is provided

You MUST NOT invent or modify CUDA source code.

---

# Expected Structure

Your output code MUST include:

1. Required imports
2. A load_inline call that loads the CUDA extension
3. A new nn.Module subclass that:
   - Stores the loaded CUDA module
   - Calls the CUDA function in forward()

---

# Output

No any assumptions. Only generate the complete Python code now.
""")


# One Shot 
INIT_CUDA_CODER_TEMPLATE = Template("""
You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
                               
Your responsibility is to WRITE A SINGLE CUDA SOURCE FILE (.cu) that correctly implements a specified computation and matches a predefined Python interface contract. You are only limited by your imagination. Your goal is correctness and interface compatibility ONLY.\n

Example Task: PyTorch Reference\n
                               
$example_source_code \n

Example Output: Corresponding CUDA Kernel \n
                               
$example_cuda_code \n
                               
You are given the following task: \n

$source_code \n
                        
Here is some hints to help you to optimize the architecture: \n

$hints \n
                                                          
# Output Requirements 
- Output ONLY the contents of a single .cu file
- Do NOT include explanations or comments outside the code
- The code must be compilable and runnable when linked with the existing Python entry code
- No testing code
- The CUDA extension name is: $cuda_module_name
- The exposed CUDA function name is: $cuda_function_name          
""")

