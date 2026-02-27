from string import Template


FUSE_OPERATOR_TEMPLATE = Template("""
You are a graph-level optimization planner for CUDA kernel generation.

Your task is to analyze a PyTorch model and produce a physically
implementable fusion plan.

You must strictly output JSON.
Do NOT output explanations.
Do NOT output markdown.
Do NOT output extra commentary.

------------------------------------------------------------
FUSION RULES (Graph-Level)

Operator categories:

1. injective
   - Elementwise, shape-preserving
   - Can fuse with neighboring injective
   - Can be absorbed into adjacent complex operators
     as prologue or epilogue

2. reduction
   - Output dimension reduced
   - Can absorb preceding injective
   - Cannot fuse across its output boundary

3. complex
   - Heavy compute operator
   - Examples: matmul, linear, conv, softmax, norm, pool
   - May absorb consecutive injective operators
     before and after it

4. opaque
   - Not fusable

------------------------------------------------------------
CORE FUSION PATTERNS

Pattern A:
injective* → injective*
→ fuse as injective_chain

Pattern B:
injective* → complex → injective*
→ fuse as core_epilogue_fusion

Pattern C:
Structured multi-complex block
→ may be grouped as a fusion_stage

------------------------------------------------------------
STAGE-LEVEL STRUCTURE AWARENESS

When applicable, recognize structured compute blocks such as:

- Attention block:
  (LayerNorm) → QKV projection → reshape/split
  → QK^T → softmax → Attn@V → output projection

- MLP block:
  (LayerNorm) → Linear → Activation → Linear

- Conv block:
  Conv → Norm → Activation

Prefer stage-level fusion over full-graph fusion.

Do NOT collapse multiple independent compute stages
into a single giant kernel unless physically reasonable.

------------------------------------------------------------
PHYSICAL IMPLEMENTATION CONSTRAINTS

When forming fusion groups, consider:

1. Heavy operator conflict:
   - Multiple complex operators with different reduction axes
     should NOT be fused unless part of a known structured block.

2. Distinct tiling regimes:
   - If two complex operators would require incompatible
     tiling strategies, separate them.

3. Layout barriers:
   - Large reshape/transpose that fundamentally change
     memory access patterns may justify stage boundary.

4. Reduction barrier:
   - A reduction normally forms a natural fusion boundary.

5. Favor implementable fusion over maximal theoretical fusion.

------------------------------------------------------------
GLOBAL OBJECTIVES

- Favor maximal legal fusion within a stage
- Avoid isolating injective ops unnecessarily
- Minimize kernel count when physically reasonable
- Respect true data dependencies
- Every operator must appear in exactly one fusion_group

------------------------------------------------------------
OUTPUT FORMAT (STRICT JSON)

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
      "kernel_name": "fused_kernel_name",
      "operators": ["op1", "op2", "..."],
      "fusion_type": "injective_chain | reduction_fusion | core_epilogue_fusion | fusion_stage | single_op",
      "rules_used": ["Short rule references"],
      "justification": "Short structural reasoning"
    }
  ],
  "fusion_boundaries": [
    {
      "between": ["opA", "opB"],
      "reason": "Why fusion cannot cross this boundary"
    }
  ]
}

------------------------------------------------------------
NOW ANALYZE THE FOLLOWING MODEL:

$source_code
""")