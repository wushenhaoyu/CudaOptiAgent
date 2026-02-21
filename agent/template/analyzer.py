from string import Template




FUSE_OPERATOR_TEMPLATE = Template("""
You are a graph-level optimization planner for CUDA kernel generation.

Your task is to analyze a PyTorch model and produce a fusion plan
based on structured operator fusion rules.

You must strictly output JSON.
Do NOT output explanations.
Do NOT output markdown.
Do NOT output extra commentary.

------------------------------------------------------------
FUSION RULES

We classify operators into four categories:

1. injective
   - One-to-one elementwise map
   - Examples: add, relu, sigmoid, tanh, gelu
   - Rule: Multiple injective operators can be fused together.

2. reduction
   - Reduce along one or more axes
   - Examples: sum, mean
   - Rule: A reduction can fuse its input injective operators.

3. complex_out_fusable
   - Heavy compute operator whose output supports elementwise fusion
   - Examples: conv2d, matmul, linear
   - Rule: Injective operators applied to its output can be fused into it.

4. opaque
   - Cannot be fused
   - Examples: sort, topk, non-deterministic ops

------------------------------------------------------------
YOUR TASK

Given a model definition:

1. Identify operators in execution order
2. Classify each operator into one of the four categories
3. Determine valid fusion groups using the rules
4. Produce a fusion plan

------------------------------------------------------------
OUTPUT FORMAT (STRICT JSON)

{
  "operators": [
    {
      "name": "...",
      "type": "...",
      "category": "injective | reduction | complex_out_fusable | opaque"
    }
  ],
  "fusion_groups": [
    {
      "group_id": 1,
      "operators": ["op1", "op2", "..."],
      "fusion_type": "injective_chain | reduction_fusion | complex_output_fusion",
      "rules_used": ["Rule description"],
      "justification": "Short structural reasoning"
    }
  ],
  "fusion_boundaries": [
    {
      "before": "op_name",
      "reason": "why fusion cannot cross this boundary"
    }
  ]
}
------------------------------------------------------------
NOW ANALYZE THE FOLLOWING MODEL:

$source_code
""")

"""
------------------------------------------------------------
ONE-SHOT EXAMPLE

INPUT MODEL:

$example_source_code

EXPECTED OUTPUT:

$example_fusion_plan
"""