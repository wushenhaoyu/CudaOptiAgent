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

The CUDA kernel implementation itself is NOT part of your task and is assumed to exist.

---

# Assumptions

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

Only generate the complete Python code now.
""")


# One Shot 
INIT_CUDA_CODER_TEMPLATE = Template("""
You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
                               
You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n

Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
                               
$example_source_code \n

The example new arch with custom CUDA kernels looks like this: \n
                               
$example_new_code \n
                               
You are given the following architecture: \n

$source_code \n
                        
Here is some hints to help you to optimize the architecture: \n

$hints \n
                                                          
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n                           
""")

