from string import Template


FUSE_ANALYZER_BASE_TEMPLATE = Template("""
You are a CUDA Kernel Architect specialized in INFERENCE-ONLY optimization.
Goal: Partition the PyTorch graph into "Safe-Aggressive" fusion groups and provide a high-level kernel implementation blueprint.

Core Rules for Fusion Analysis:

1. Maintain a single dominant parallel pattern per kernel (GEMM, Conv, Reduction, Elementwise).
2. Only one heavy operator per kernel (GEMM/Conv/Large Reduction). Fuse elementwise ops (Bias, ReLU) as epilogue.
3. Avoid cross-block synchronization dependencies (e.g., Softmax denominator requiring full reduction).
4. Maintain compatible memory access patterns (do not fuse ops with incompatible layouts or access orders).
5. Ignore training-only operators such as Dropout.

OUTPUT FORMAT:
```json
{
  "task_description": "<A concise summary of the model and inference task>",
  "fusion_groups": [
    {
      "kernel_id": 0,
      "kernel_name": "<name for kernel>",
      "operators": ["Conv2d","BatchNorm","ReLU"],
      "dominant_pattern": "Convolution",
      "fusion_type": "Heavy+Epilogue | Standalone Heavy | Standalone Reduction | Pure Elementwise",
      
      "implementation_hint": {
        "parallel_structure": "<e.g., 1 output element per thread, grid-stride loop over batch>",
        "compute_flow": [
            "<stepwise computation, e.g., load input tile, compute conv, apply batchnorm, apply relu>"
        ],
        "memory_notes": [
            "<coalesced accesses, weights reuse, shared memory hints>"
        ]
      },
      "justification": "Explain memory access safety, parallelism considerations, and fusion decisions."
    }
  ]
}
```

ANALYZE THE FOLLOWING INFERENCE MODEL:
$source_code
""")

