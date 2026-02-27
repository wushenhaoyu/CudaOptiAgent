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
OBJECTIVE

The goal is NOT maximal fusion.
The goal is realistic stage-level kernel partitioning
that reflects practical GPU implementation constraints.

Minimize kernel count when safe.
Avoid unrealistic multi-heavy-op fusion.
Respect real hardware execution structure.

------------------------------------------------------------
OPERATOR CATEGORIES

1. injective
   - Elementwise, shape-preserving
   - Can fuse with neighboring injective
   - Can be absorbed into adjacent complex operators (prologue or epilogue)

2. reduction
   - Output dimension reduced
   - Can absorb preceding injective
   - Cannot fuse across its output boundary

3. complex
   - Heavy compute operator
   - Examples: matmul, linear, conv, softmax, norm, pool
   - May absorb consecutive injective operators before/after
   - Usually defines a kernel core
   - Note: norm (LayerNorm, BatchNorm) and pooling operators contain
     internal reductions and define their own tile boundaries.
     Treat them as complex with embedded reduction — they can absorb
     injective epilogues but cannot be merged with other
     reduction-heavy operators.

4. opaque
   - Cannot be fused
   - Must form its own kernel

------------------------------------------------------------
HARD BARRIERS (MUST NOT CROSS)

Fusion MUST stop if any of the following occur:

- Output rank reduction
- Opaque operator
- True multi-parent data dependency
- Required tensor materialization
- Incompatible reduction domains

------------------------------------------------------------
SOFT BARRIERS (AVOID UNLESS STRUCTURED BLOCK)

Avoid crossing unless clearly part of a recognized structured compute block:

- Two independent GEMM / matmul operators
- Distinct reduction axes
- Major tiling strategy change
- Register lifetime explosion
- Layout requiring re-tiling

------------------------------------------------------------
TILE LIFETIME RULE

If two complex operators require independent tiling loops
with different reduction axes,
they should NOT be fused unless part of a known
streaming structured algorithm (e.g., FlashAttention-style).

------------------------------------------------------------
MEMORY MATERIALIZATION RULE

If an operator only changes view/stride without data movement,
it should NOT create a fusion boundary.

If an operator requires contiguous re-materialization
(e.g., reshape before conv, permute before matmul),
it SHOULD create a fusion boundary unless the reshape
is a pure view with no data movement.

------------------------------------------------------------
BOUNDARY ASSIGNMENT RULE

If an injective operator sits between two complex operators,
assign it as epilogue of the preceding complex operator,
not as prologue of the following one.

------------------------------------------------------------
KERNEL COST HEURISTIC

When deciding fusion:

- Prefer one heavy reduction core per kernel
- Allow injective* -> complex -> injective* fusion
- Avoid fusing unrelated heavy operators
- Favor producer-consumer streaming patterns
- Minimize global memory round-trips
- Avoid creating kernels with multiple unrelated reduction domains

------------------------------------------------------------
STRUCTURED BLOCK DETECTION

You MAY group multiple complex operators into a single
"fusion_stage" IF they form a well-known structured block:

Examples:
- Attention core (QK^T -> scale -> mask -> softmax -> @V)
- MLP block (Linear -> Act -> Linear)
- Norm + Linear prologue

Such grouping must respect TILE LIFETIME RULE.

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
      "kernel_name": "...",
      "operators": ["op1", "op2", "..."],
      "fusion_type": "injective_chain | prologue_fusion | epilogue_fusion | prologue_epilogue_fusion | reduction_fusion | fusion_stage | single_op",
      "rules_used": [
        // Must reference one or more of:
        // "HARD_BARRIER", "SOFT_BARRIER", "TILE_LIFETIME",
        // "MEMORY_MATERIALIZATION", "BOUNDARY_ASSIGNMENT",
        // "STRUCTURED_BLOCK", "KERNEL_COST_HEURISTIC",
        // "INJECTIVE_ABSORPTION"
      ],
      "justification": "Short structural reasoning"
    }
  ],
  "fusion_boundaries": [
    {
      "between": ["opA", "opB"],
      "reason": "Structural reason for boundary"
    }
  ]
}

------------------------------------------------------------
NOW ANALYZE THE FOLLOWING MODEL:

$source_code
""")