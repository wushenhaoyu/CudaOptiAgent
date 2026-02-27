from string import Template


FUSE_OPERATOR_TEMPLATE = Template("""
You are a graph-level CUDA kernel partition planner.

Your task:
Analyze a PyTorch model graph and produce a realistic,
stage-level fusion plan suitable for GPU kernel generation.

You must strictly output JSON.
Do NOT output explanations.
Do NOT output markdown.
Do NOT output extra commentary.

------------------------------------------------------------
FUSION RULES

We classify operators into four categories:

1. injective
   - One-to-one elementwise map
   - Examples: add, relu, sigmoid, tanh, gelu, mish, hardtanh
   - Rule: Multiple injective operators can be fused together.

2. reduction
   - Output dimension reduced
   - Can absorb preceding injective
   - Cannot fuse across its output boundary

3. complex_out_fusable
   - Heavy compute operator whose output supports elementwise fusion
   - Examples: conv2d, matmul, linear, conv_transpose, avg_pool, max_pool, softmax, layernorm, batchnorm, groupnorm
   - Rule: Injective operators applied to its output can be fused into it.
   - Note: Includes operators with internal reduction (softmax, layernorm, pool) as long as output shape unchanged and fusable

4. opaque
   - Cannot be fused
   - Examples: sort, topk, argsort, non-deterministic ops

------------------------------------------------------------
KEY PRINCIPLES & FUSION GUIDELINES

- Output shape unchanged + fusable → complex_out_fusable
- Output dimension reduced → reduction
- Pool (avg/max): local window, NOT large-scale reduction → complex_out_fusable
- Softmax/LayerNorm/GroupNorm: internal reduction but output fusable → complex_out_fusable

- COMPLETENESS: Every operator in the "operators" list must appear in exactly one fusion_group.
- SINGLE_OP: If an operator cannot be fused with its neighbors, it forms its own fusion_group with fusion_type = "single_op".
- KERNEL_NAME: Each fusion_group represents one CUDA kernel that will be generated:
  * For fused groups: Use descriptive name like "fused_{op1}_{op2}_{op3}"
  * For single ops: Use format "{op_name}_kernel"
- FUSION_BOUNDARIES: Identify points where fusion cannot continue, using "between" to show the exact boundary.

------------------------------------------------------------
REQUIRED OUTPUT FORMAT (STRICT JSON)

{
  "operators": [
    {
      "name": "...",
      "type": "...",
      "category": "injective | reduction | complex | opaque"
    }
  ],
  "fusion_groups": [
    {
      "group_id": 1,
      "kernel_name": "fused_kernel_1",
      "operators": ["op1", "op2", "..."],
      "fusion_type": "injective_chain | reduction_fusion | complex_output_fusion | single_op",
      "rules_used": ["Rule description"],
      "justification": "Short structural reasoning"
    }
  ],
  "fusion_boundaries": [
    {
      "between": ["opA", "opB"],
      "reason": "why fusion cannot cross this boundary"
    }
  ]
}

------------------------------------------------------------
NOW ANALYZE THE FOLLOWING MODEL:

$source_code
""")