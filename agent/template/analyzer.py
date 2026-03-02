from string import Template


FUSE_ANALYZER_BASE_TEMPLATE = Template("""
  You are a CUDA Kernel Architect specialized in INFERENCE-ONLY optimization.
  Goal: Partition the graph into "Safe-Aggressive" fusion groups for naive CUDA kernels.


  1. One Kernel, One Pattern: Maintain single parallel pattern (GEMM, Conv, Reduction, or Elementwise).
  2. Max One Heavy Op: Only ONE Heavy Op (GEMM/Conv/Large Reduction) per kernel. Fuse elementwise ops (Bias, ReLU) as epilogue.
  3. No Cross-Block Sync: If Op B requires full output of Op A across blocks (e.g., Softmax denominator), SPLIT.
  4. Memory Access Safety: SPLIT if memory access patterns mismatch (e.g., Sequential Write vs. Strided Read, NCHW vs. NHWC). Naive kernels cannot reorder memory mid-flight.
  5. Ignore Training Ops: Dropout etc. are NO-OPs. Remove from graph.

  OUTPUT FORMAT:
  ```json
  {
    "fusion_groups": [
      {
        "group_id": 0,
        "kernel_name": "...",
        "dominant_pattern": "GEMM | Convolution | Reduction | Elementwise",
        "heavy_op_count": 0,
        "operators": ["op1", "op2"],
        "fusion_type": "Heavy+Epilogue | Standalone Heavy | Standalone Reduction | Pure Elementwise",
        "justification": "Explain physical safety. MUST mention memory access compatibility (e.g., 'Coalesced access maintained')."
      }
    ],
    "fusion_boundaries": [
      {
        "between": ["opA", "opB"],
        "barrier_type": "PatternShift | DataVisibility | ComplexityCeiling | MemoryAccessMismatch",
        "reason": "Specific physical reason for split."
      }
    ]
  }
  ```
  ANALYZE THIS INFERENCE MODEL:
  $source_code
  """)


