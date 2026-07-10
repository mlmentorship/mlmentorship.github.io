---
title: "Multi-head attention: why one head is not enough"
description: "Run h independent attention computations in parallel, then concatenate. Each head specializes in a different relation. The mechanism most senior candidates can write but few can motivate."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

**Multi-head attention** projects $Q$, $K$, $V$ into $h$ lower-dimensional subspaces, runs scaled dot-product attention independently in each, and concatenates the results before a final output projection. Same FLOPs as one large head; very different inductive bias.

## Why it matters

Single-head attention computes one weighted average per position. That single distribution has to encode every relation the model needs: syntactic, positional, semantic, coreferential. In practice it cannot, and ablations show that single-head transformers underperform multi-head transformers at matched parameter count ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).

Multiple heads let different attention patterns coexist. One head learns "previous token," another "matching bracket," another "this noun's modifier." Probing studies on BERT show many heads fire on syntactic dependencies that linguists recognize ([Clark et al., 2019](https://arxiv.org/abs/1906.04341)).

## The mechanism

Given input $X \in \mathbb{R}^{n \times d}$ and head count $h$ with per-head dimension $d_h = d / h$:

1. **Project**: $Q = X W_Q$, $K = X W_K$, $V = X W_V$, each shape $n \times d$. Reshape to $n \times h \times d_h$.
2. **Per-head attention**: for each head $i$,
$$
\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_h}}\right) V_i.
$$
3. **Concatenate**: stack the $h$ heads back into shape $n \times d$.
4. **Output projection**: $\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \, W_O$.

Total parameters: $4 d^2$ (the four $d \times d$ projection matrices). FLOPs: $O(n^2 d + n d^2)$. Identical to single-head; the heads share the budget.

## Why split the dimension

If you keep $d_h = d$ per head and run $h$ heads, you multiply parameters and compute by $h$. Splitting $d$ across heads keeps the cost matched to a single-head baseline, so any gain is attributable to the multiplicity itself, not extra capacity. This is the design choice that makes the comparison meaningful.

## Variants

- **Multi-query attention (MQA)**: share $K$ and $V$ across all heads; only $Q$ is per-head. KV-cache shrinks by $h$x. See [GQA and MQA](/concepts/grouped-query-attention/).
- **Grouped-query attention (GQA)**: share $K, V$ across groups of heads. Compromise between full MHA and MQA. The Llama 2/3 default.
- **Cross-attention**: $Q$ from one sequence, $K, V$ from another. See [self-attention vs cross-attention](/concepts/self-attention-vs-cross-attention/).
- **Sliding-window / sparse**: restrict each head to a local window or learned sparse pattern.

## Tradeoffs

- **Head count**: 8 to 32 is typical. More heads with smaller $d_h$ can hurt expressiveness; fewer heads with larger $d_h$ loses specialization. $d_h = 64$ to $128$ is the modern sweet spot.
- **KV-cache memory** scales linearly with $h$ in vanilla MHA. The motivation for MQA and GQA at long context.

## Common pitfalls

- **Equating "more heads" with "more capacity."** Splitting fixes the parameter budget; it is a structural choice, not a scale-up.
- **Reading the post-softmax weights as "what the model attends to."** Heads are mixed in $W_O$. Single-head probes can be misleading.
- **Treating MHA as the bottleneck.** In long-context LLMs, the FFN is usually larger; attention compute scales with $n^2$ but FFN compute scales with $n d^2$.

## Related

- [The attention mechanism](/concepts/attention-mechanism/).
- [GQA and MQA](/concepts/grouped-query-attention/).
- [Self-attention vs cross-attention](/concepts/self-attention-vs-cross-attention/).
