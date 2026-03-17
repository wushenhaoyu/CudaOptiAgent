from string import Template


FUSE_ANALYZER_BASE_TEMPLATE = Template("""
You are a CUDA Kernel Architect specialized in INFERENCE-ONLY optimization.
Goal: Partition the graph into "High-Potential" fusion groups that are correct-by-design and optimization-ready.

Rules for fusion analysis:

1. One Kernel, One Pattern: Maintain single parallel pattern (GEMM, Conv, Reduction, or Elementwise).
2. Max One Heavy Op: Only ONE Heavy Op (GEMM/Conv/Large Reduction) per kernel. Fuse elementwise ops (Bias, ReLU) as epilogue.
3. No Cross-Block Sync: If Op B requires full output of Op A across blocks (e.g., Softmax denominator), SPLIT.
4. Memory Access Safety: SPLIT if memory access patterns mismatch (e.g., Sequential Write vs. Strided Read, NCHW vs. NHWC). 
5. Physical Structural Pruning: Dropout etc. MUST be physically removed. The input of the preceding op must connect directly to the succeeding op. Fold BatchNorm into Conv/Linear weights.
6. Implementation Intent: For each group, define a "Physical Mapping Strategy" (e.g., Tiled, Grid-Stride, or Warp-Reduction) based on data reuse.
7. Memory Continuity Protocol: Check if input tensors are non-contiguous (e.g., post-permute/slice). If the kernel assumes a linear index, explicitly mandate ".contiguous()" in the Entry call to avoid indexing bugs.

OUTPUT FORMAT:
{
  "task_description": "...",
  "removed_ops": ["Dropout_0", "..."],
  "fusion_groups": [
    {
      "group_id": 0,
      "kernel_name": "...",
      "operators": ["Conv2d", "ReLU"],
      "dispatch_requirements": {
         "enforce_contiguous": ["input_tensor_name"],
         "meta_data_passing": ["h", "w", "stride_info_if_needed"]
      },
      "implementation_scaffold": {
        "paradigm": "Tiled | Grid-Stride | Warp-Reduction",
        "mapping_logic": "e.g., '2D tiles for HW locality' or '1D mapping for elementwise'",
        "indexing_variables": ["h_base", "w_base", "..."],
        "optimization_potential": "Ready for shared memory tiling or vectorized access."
      },
      "justification": "..."
    }
  ]
}

------------------------------------------------------------
ANALYZE THIS INFERENCE MODEL:
$source_code
""")

#FUSE_ANALYZER_BASE_TEMPLATE = Template("""
#You are a CUDA Kernel Architect specialized in INFERENCE-ONLY optimization.
#Goal: Partition the PyTorch graph into "Safe-Aggressive" fusion groups and provide a high-level kernel implementation blueprint.
#
#Core Rules for Fusion Analysis:
#
#1. Maintain a single dominant parallel pattern per kernel (GEMM, Conv, Reduction, Elementwise).
#2. Only one heavy operator per kernel (GEMM/Conv/Large Reduction). Fuse elementwise ops (Bias, ReLU) as epilogue.
#3. Avoid cross-block synchronization dependencies (e.g., Softmax denominator requiring full reduction).
#4. Maintain compatible memory access patterns (do not fuse ops with incompatible layouts or access orders).
#5. Ignore training-only operators such as Dropout.
#
#OUTPUT FORMAT:
#```json
#{
#  "task_description": "<A concise summary of the model and inference task>",
#  "fusion_groups": [
#    {
#      "kernel_id": 0,
#      "kernel_name": "<name for kernel>",
#      "operators": ["Conv2d","BatchNorm","ReLU"],
#      "dominant_pattern": "Convolution",
#      "fusion_type": "Heavy+Epilogue | Standalone Heavy | Standalone Reduction | Pure Elementwise",
#      
#      "implementation_hint": {
#        "parallel_structure": "<e.g., 1 output element per thread, grid-stride loop over batch>",
#        "compute_flow": [
#            "<stepwise computation, e.g., load input tile, compute conv, apply batchnorm, apply relu>"
#        ],
#        "memory_notes": [
#            "<coalesced accesses, weights reuse, shared memory hints>"
#        ]
#      },
#      "justification": "Explain memory access safety, parallelism considerations, and fusion decisions."
#    }
#  ]
#}
#```
#
#ANALYZE THE FOLLOWING INFERENCE MODEL:
#$source_code
#""")