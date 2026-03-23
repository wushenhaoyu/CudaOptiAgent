from string import Template


#FUSE_ANALYZER_BASE_TEMPLATE = Template("""
#You are a CUDA Kernel Architect specialized in INFERENCE-ONLY optimization.
#Goal: Partition the graph into "High-Potential" fusion groups that are correct-by-design and optimization-ready.
#
#Rules for fusion analysis:
#
#1. One Kernel, One Pattern: Maintain single parallel pattern (GEMM, Conv, Reduction, or Elementwise).
#2. Max One Heavy Op: Only ONE Heavy Op (GEMM/Conv/Large Reduction) per kernel. Fuse elementwise ops (Bias, ReLU) as epilogue.
#3. No Cross-Block Sync: If Op B requires full output of Op A across blocks (e.g., Softmax denominator), SPLIT.
#4. Memory Access Safety: SPLIT if memory access patterns mismatch (e.g., Sequential Write vs. Strided Read, NCHW vs. NHWC). 
#5. Physical Structural Pruning: Dropout etc. MUST be physically removed. The input of the preceding op must connect directly to the succeeding op. Fold BatchNorm into Conv/Linear weights.
#6. Implementation Intent: For each group, define a "Physical Mapping Strategy" (e.g., Tiled, Grid-Stride, or Warp-Reduction) based on data reuse.
#7. Memory Continuity Protocol: Check if input tensors are non-contiguous (e.g., post-permute/slice). If the kernel assumes a linear index, explicitly mandate ".contiguous()" in the Entry call to avoid indexing bugs.
#8. Identify repeated operator sequences (e.g., Conv → ReLU → BatchNorm) and create a single fusion template per unique sequence.  
#   - Mark repeated sequences to indicate they can reuse the same CUDA kernel template.  
#   - List all occurrences of the sequence in the model for reference.
#OUTPUT FORMAT:
#{
#  "task_description": "...",
#  "removed_ops": ["Dropout_0", "..."],
#  "fusion_groups": [
#    {
#      "group_id": 0,
#      "kernel_name": "...",
#      "operators": ["Conv2d", "ReLU"],
#      "dispatch_requirements": {
#         "enforce_contiguous": ["input_tensor_name"],
#         "meta_data_passing": ["h", "w", "stride_info_if_needed"]
#      },
#      "implementation_scaffold": {
#        "paradigm": "Tiled | Grid-Stride | Warp-Reduction",
#        "mapping_logic": "e.g., '2D tiles for HW locality' or '1D mapping for elementwise'",
#        "indexing_variables": ["h_base", "w_base", "..."],
#        "optimization_potential": "Ready for shared memory tiling or vectorized access."
#      },
#      "justification": "..."
#    }
#  ]
#}
#
#------------------------------------------------------------
#ANALYZE THIS INFERENCE MODEL:
#$source_code
#""")

FUSE_ANALYZER_BASE_TEMPLATE = Template("""
You are a CUDA Kernel Architect specialized in INFERENCE-ONLY optimization.
Goal: Partition the graph into "High-Potential" fusion groups that are correct-by-design and optimization-ready.

Rules for fusion analysis:

1. One Kernel, One Pattern: Maintain single parallel pattern (GEMM, Conv, Reduction, or Elementwise).
2. Max One Heavy Op: Only ONE Heavy Op (GEMM/Conv/Large Reduction) per kernel. Fuse elementwise ops (Bias, ReLU) as epilogue.
3. No Cross-Block Sync: If Op B requires full output of Op A across blocks (e.g., Softmax denominator), SPLIT.
4. Memory Access Safety: SPLIT if memory access patterns mismatch (e.g., Sequential Write vs. Strided Read, NCHW vs. NHWC). 
5. Physical Structural Pruning: Dropout etc. MUST be physically removed. The input of the preceding op must connect directly to the succeeding op. Fold BatchNorm into Conv/Linear weights.
6. Memory Continuity Protocol: Check if input tensors are non-contiguous (e.g., post-permute/slice). If the kernel assumes a linear index, explicitly mandate ".contiguous()" in the Entry call to avoid indexing bugs.
7. Identify repeated operator sequences (e.g., Conv → ReLU → BatchNorm) and create a single fusion template per unique sequence.  
   - Mark repeated sequences to indicate they can reuse the same CUDA kernel template.  
   - List all occurrences of the sequence in the model for reference.


OUTPUT FORMAT:
```json
{
  "task_description": "...",
  "fusion_groups": [
    {
      "group_id": 0,
      "kernel_name": "...",
      "operators": ["Conv2d", "ReLU"],
      "justification": "..."
    }
  ]
}```

ANALYZE THIS INFERENCE MODEL:
$source_code
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
entry.py is the Python entry code,
kernel.cu contains CUDA kernels and pybind bindings,
ref.py is the original PyTorch reference implementation (read only),
there may be additional .cu/.h files inside the kernel directory
$file_list

- Selected files with content:
$selected_files_content


Output strictly in the following JSON format:
```json
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
```
Rules:
- Strictly Error-Fix Only: Solve the reported $error_message. Ignore any performance optimizations (Tiling, Vectorization) or refactoring unless they are the only way to fix the bug.
- Minimal Invasive Changes: Correct logic, indexing, or configuration overflows without altering the kernel's basic architecture.
- JSON Compliance: Each file appears once. file_name must include 'kernel/' for .cu files. related_files must be from Available files. 
- Output Format: NO commentary. NO full file rewrites. Return valid JSON only.
- Specificity: suggested_fix must state exactly WHAT to change and HOW to resolve the mismatch or crash.
""")



OPTIMIZE_PLAN_TEMPLATE_FIRST = Template("""
You are a CUDA Kernel Optimizer.

Goal: Analyze the kernel with NCU report and identify the main bottleneck. 
Propose only the most impactful optimizations.

You may consider the following optimization techniques:

[Memory]
- Memory coalescing (consecutive access patterns)
- Shared memory tiling (data reuse)
- Bank conflict avoidance (padding shared memory)
- Vectorized loads (float2/float4)
- Cache utilization

[Compute]
- Instruction-level parallelism (ILP)
- Loop unrolling
- Tensor Cores (WMMA/MMA)
- Mixed precision (FP16/TF32)

[Parallelism]
- Occupancy tuning (block size, register usage)
- Warp efficiency
- Grid-stride loops

[Latency Hiding]
- Prefetching
- Double buffering
- Persistent kernels

[Control Flow]
- Boundary checks (tid < size)
- Warp divergence reduction

[Advanced]
- Warp-level primitives (__shfl_sync, __ballot_sync)
- Kernel fusion
- CUDA Graphs

Rules:
- DO NOT list all techniques
- SELECT only 1–3 most relevant optimizations
- Each optimization MUST be tied to a real issue from NCU
- Prefer high-impact structural changes over minor tweaks
- Be specific about what to change in code

kernel:
```cuda
$cuda_code
```

ncu_report:
```json
$ncu_report
```
device_info:
```json
$device_info
```
Output:

{
  "bottleneck": "memory | compute | latency | occupancy | divergence",
  "reason": "based on NCU metrics",
  "optimizations": [
    {
      "technique": "...",
      "action": "... (concrete code-level change)"
    }
  ],
  "description":"describe the optimize direction"
}

""")