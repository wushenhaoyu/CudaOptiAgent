from string import Template


FUSE_ANALYZER_BASE_TEMPLATE = Template("""
You are a CUDA Kernel Architect specialized in INFERENCE-ONLY optimization.
Goal: Partition the graph into "Safe-Aggressive" fusion groups for naive CUDA kernels.

Rules for fusion analysis:

1. One Kernel, One Pattern: Maintain single parallel pattern (GEMM, Conv, Reduction, or Elementwise).
2. Max One Heavy Op: Only ONE Heavy Op (GEMM/Conv/Large Reduction) per kernel. Fuse elementwise ops (Bias, ReLU) as epilogue.
3. No Cross-Block Sync: If Op B requires full output of Op A across blocks (e.g., Softmax denominator), SPLIT.
4. Memory Access Safety: SPLIT if memory access patterns mismatch (e.g., Sequential Write vs. Strided Read, NCHW vs. NHWC). Naive kernels cannot reorder memory mid-flight.
5. Ignore Training Ops: Dropout etc. are NO-OPs. Remove from graph.
6. Identify repeated operator sequences (e.g., Conv → ReLU → BatchNorm) and **create a single fusion template per unique sequence**.  
   - Mark repeated sequences to indicate they can reuse the same CUDA kernel template.  
   - List all occurrences of the sequence in the model for reference.

OUTPUT FORMAT:
```json
{
  "task_description": "<A concise summary of the task, e.g., 'Image classification using ResNet-like Conv/BN/ReLU blocks'>",
  "fusion_groups": [
    {
      "group_id": 0,
      "kernel_name": "...",
      "operators": ["Conv2d", "ReLU", "BatchNorm"],
      "dominant_pattern": "Convolution",
      "fusion_type": "Heavy+Epilogue | Standalone Heavy | Standalone Reduction | Pure Elementwise",
      "justification": "Explain physical safety. MUST mention memory access compatibility (e.g., 'Coalesced access maintained')."
    }
  ]
}

ANALYZE THIS INFERENCE MODEL:
$source_code
""")

