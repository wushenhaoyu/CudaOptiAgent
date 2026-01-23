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
                                   
---

# Output

No any assumptions. Only generate the complete Python code now.
""")

REPAIR_ENTRY_CODER_TEMPLATE = Template("""
You are a Python interface engineer responsible for generating the Python entry code
that connects a PyTorch model to a custom CUDA extension.

You do NOT write CUDA kernels.
You do NOT optimize algorithms.
You ONLY generate the Python-side interface code.

---

You are given the original PyTorch model code:

$source_code
                                       
Here is your generated entry code, but it needs to be fixed:

$entry_code
                                       
Here is some useful information for you:
                                       
$error_report

---
You must:
1. Load a CUDA extension using torch.utils.cpp_extension.load
2. Call this CUDA function inside the model's forward method
3. Preserve the original model's input/output semantics
4. Class name must be "ModelNew"
5. The CUDA extension name is: $cuda_module_name
6. The exposed CUDA function name is: $cuda_function_name
7. The sources name is included as $kernel_dir, please do NOT change it
8. The generated class must have methods with the same names as those in the given task class and must not include any extra ones.
                                       
# Output                               
No any assumptions. Only generate the complete Python code now."""
)


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
                               
You are given the following task: \n

$source_code \n
                                     
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

INIT_CPU_CODER_TEMPLATE = Template("""
You are a C backend implementation agent.

Your role is to generate a COMPLETE, COMPILABLE C SOURCE FILE
that implements the given computation on CPU
and matches the Python entry interface exactly.

This C file will be compiled and loaded by entry.py
as a CPU reference backend.

Correctness is the ONLY goal.
Performance is completely irrelevant.

---

# Inputs

## PyTorch Reference Code (Semantic Definition)
$source_code

## Python Entry Interface (Exact ABI Contract)
$entry_code

---

# Your Task

Generate a single complete C source file (.cu)
that implements the same computation as the PyTorch code,
and can be called directly from entry.py.

This file serves as the CPU ground-truth implementation.

---

# Hard Constraints (MUST FOLLOW)

1. The implementation MUST be purely sequential.
   - No multithreading.
   - No OpenMP.
   - No SIMD.
   - No pthreads.

2. The implementation MUST be a complete C file.
   - Includes.
   - Function definitions.
   - Exposed symbols.
   - No missing parts.

3. The exposed C functions MUST exactly match
   the interface expected by entry.py.

4. You MAY use standard C math libraries.
   - math.h
   - exp, log, sqrt, etc.

5. You MUST NOT use any parallel computing libraries.
   - No OpenMP.
   - No BLAS.
   - No MKL.
   - No Eigen.
   - No CUDA.

6. All computations MUST be explicit and deterministic.
   - Loops.
   - Scalar operations.
   - Explicit indexing.

7. The implementation MUST produce numerically correct results
   equivalent to PyTorch.

---

# Output Requirements (STRICT)

- Output ONLY a single complete C source file.
- No explanations.
- No markdown.
- No testing code.
- No main function unless entry.py explicitly requires it.
""")


REPAIR_CPU_CODER_TEMPLATE = Template("""
You are a C backend repair agent.

Your role is to generate a NEW COMPLETE C SOURCE FILE
that fixes the previously generated CPU backend
while preserving its overall structure and interface.

This C file serves as a CPU reference implementation
for a CUDA kernel and must match the Python entry interface exactly.

Correctness and interface compatibility are the ONLY goals.
Performance is irrelevant.

---

# Inputs

## Original Task Definition (PyTorch Semantics)
$source_code

## Previously Generated (Broken) CPU C Backend
$last_cpu_code

## Error Analysis and Repair Guidance
$hints

---

# Your Task

Produce a corrected version of the CPU C backend.

You MUST base your implementation on the previous version,
and apply only the necessary structural fixes suggested by the hints.

Do NOT redesign the algorithm.
Do NOT change the interface.
Do NOT introduce new abstractions.

The goal is minimal, conservative repair.

---

# Hard Constraints (MUST FOLLOW)

1. The output MUST be a COMPLETE C SOURCE FILE.
   - Includes.
   - Function definitions.
   - Exposed symbols.

2. The implementation MUST remain purely sequential.
   - No parallelism.
   - No OpenMP.
   - No threads.
   - No SIMD.

3. The exposed C functions MUST exactly match
   the Python entry interface.

4. You MAY use standard C math libraries.
   - math.h
   - exp, log, sqrt, etc.

5. You MUST NOT use any parallel or optimized libraries.
   - No BLAS.
   - No MKL.
   - No Eigen.
   - No CUDA.

6. Preserve the overall structure of the previous code.
   - Same function names.
   - Same data layout assumptions.
   - Same control flow where possible.

7. Apply ONLY the fixes necessary to resolve the reported errors.
   - No speculative changes.
   - No performance-oriented changes.

---

# Output Requirements (STRICT)

- Output ONLY the contents of a single complete C source file.
- No explanations.
- No markdown.
- No testing code.
- No main function unless required by the Python entry interface.
""")


