---
title: "Multi-head attention: why one head is not enough"
description: "Run h independent attention computations in parallel, then concatenate. Each head specializes in a different relation. The mechanism most senior candidates can write but few can motivate."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Multi-head attention** projects $Q$, $K$, $V$ into $h$ lower-dimensional subspaces, runs scaled dot-product attention independently in each, and concatenates the results before a final output projection. Same FLOPs as one large head; very different inductive bias.

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

Ignoring optional biases, total parameters are $4 d^2$ (the four $d \times d$ projection matrices). FLOPs are $O(n^2 d + n d^2)$, the same asymptotic cost as single-head attention; the heads share the dimension budget.

<!-- visual:multi-head-attention-split-recombine -->
<figure class="learning-figure plot-panel" aria-labelledby="multi-head-visual-title">
	<p class="visual-kicker">Tensor flow</p>
	<p class="visual-title" id="multi-head-visual-title">Heads split the feature axis, not the token sequence.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 510" role="img" aria-labelledby="multi-head-svg-title multi-head-svg-desc">
			<title id="multi-head-svg-title">Q, K, and V projection into parallel attention heads and recombination</title>
			<desc id="multi-head-svg-desc">A full input sequence X with n tokens and d features is projected into Q, K, and V, each n by d. The feature axis is split across h parallel heads. Representative head 1 and head h each receive all n tokens through Q i, K i, and V i tensors of shape n by d sub h, and each produces an n by d sub h output. The h head outputs concatenate into n by h d sub h, equal to n by d, then a d by d output matrix W O produces the final n by d tensor. The diagram explicitly states that heads do not receive disjoint token subsets.</desc>
			<defs><marker id="arrow-forward" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
			<text class="viz-axis-label" x="16" y="18">1 · SHARED SEQUENCE</text>
			<rect class="viz-node viz-node--input" x="104" y="28" width="152" height="42" rx="3"></rect>
			<text class="viz-node-label" x="180" y="46">X</text>
			<text class="viz-node-value" x="180" y="62">all n tokens × d features</text>
			<path class="viz-forward" d="M180 70V84"></path>
			<text class="viz-axis-label" x="16" y="99">2 · PROJECT Q, K, V</text>
			<rect class="viz-node viz-node--focus" x="28" y="110" width="88" height="42" rx="3"></rect>
			<rect class="viz-node viz-node--focus" x="136" y="110" width="88" height="42" rx="3"></rect>
			<rect class="viz-node viz-node--focus" x="244" y="110" width="88" height="42" rx="3"></rect>
			<text class="viz-node-label" x="72" y="128">Q</text><text class="viz-node-value" x="72" y="144">n × d</text>
			<text class="viz-node-label" x="180" y="128">K</text><text class="viz-node-value" x="180" y="144">n × d</text>
			<text class="viz-node-label" x="288" y="128">V</text><text class="viz-node-value" x="288" y="144">n × d</text>
			<path class="viz-forward" d="M72 152V160H49V184"></path>
			<path class="viz-forward" d="M180 152V166H94V184"></path>
			<path class="viz-forward" d="M288 152V172H139V184"></path>
			<path class="viz-forward" d="M72 152V172H221V184"></path>
			<path class="viz-forward" d="M180 152V166H266V184"></path>
			<path class="viz-forward" d="M288 152V160H311V184"></path>
			<rect class="viz-node viz-node--input" x="20" y="188" width="148" height="142" rx="4"></rect>
			<rect class="viz-node viz-node--output" x="192" y="188" width="148" height="142" rx="4"></rect>
			<text class="viz-callout" x="94" y="207" text-anchor="middle">HEAD 1 · all n tokens</text>
			<text class="viz-callout" x="266" y="207" text-anchor="middle">HEAD h · all n tokens</text>
			<rect class="viz-node viz-node--focus" x="30" y="218" width="38" height="38" rx="2"></rect>
			<rect class="viz-node viz-node--focus" x="75" y="218" width="38" height="38" rx="2"></rect>
			<rect class="viz-node viz-node--focus" x="120" y="218" width="38" height="38" rx="2"></rect>
			<text class="viz-callout" x="49" y="233" text-anchor="middle">Q₁</text><text class="viz-label" x="49" y="249" text-anchor="middle">n×dₕ</text>
			<text class="viz-callout" x="94" y="233" text-anchor="middle">K₁</text><text class="viz-label" x="94" y="249" text-anchor="middle">n×dₕ</text>
			<text class="viz-callout" x="139" y="233" text-anchor="middle">V₁</text><text class="viz-label" x="139" y="249" text-anchor="middle">n×dₕ</text>
			<rect class="viz-node viz-node--focus" x="202" y="218" width="38" height="38" rx="2"></rect>
			<rect class="viz-node viz-node--focus" x="247" y="218" width="38" height="38" rx="2"></rect>
			<rect class="viz-node viz-node--focus" x="292" y="218" width="38" height="38" rx="2"></rect>
			<text class="viz-callout" x="221" y="233" text-anchor="middle">Qₕ</text><text class="viz-label" x="221" y="249" text-anchor="middle">n×dₕ</text>
			<text class="viz-callout" x="266" y="233" text-anchor="middle">Kₕ</text><text class="viz-label" x="266" y="249" text-anchor="middle">n×dₕ</text>
			<text class="viz-callout" x="311" y="233" text-anchor="middle">Vₕ</text><text class="viz-label" x="311" y="249" text-anchor="middle">n×dₕ</text>
			<text class="viz-label" x="94" y="274" text-anchor="middle">scaled dot-product attention</text>
			<text class="viz-label" x="266" y="274" text-anchor="middle">scaled dot-product attention</text>
			<path class="viz-forward" d="M94 278V288"></path>
			<path class="viz-forward" d="M266 278V288"></path>
			<rect class="viz-node viz-node--output" x="53" y="292" width="82" height="28" rx="2"></rect>
			<rect class="viz-node viz-node--output" x="225" y="292" width="82" height="28" rx="2"></rect>
			<text class="viz-callout" x="94" y="310" text-anchor="middle">head₁ · n×dₕ</text>
			<text class="viz-callout" x="266" y="310" text-anchor="middle">headₕ · n×dₕ</text>
			<text class="viz-callout" x="180" y="244" text-anchor="middle">⋯</text>
			<path class="viz-forward" d="M94 330V358H180V374"></path>
			<path class="viz-forward" d="M266 330V358H180V374"></path>
			<text class="viz-axis-label" x="16" y="354">3 · CONCATENATE h FEATURE SLICES</text>
			<rect class="viz-node viz-node--focus" x="72" y="378" width="216" height="42" rx="3"></rect>
			<text class="viz-node-label" x="180" y="396">[ head₁ | ··· | headₕ ]</text>
			<text class="viz-node-value" x="180" y="412">n × (h dₕ) = n × d</text>
			<path class="viz-forward" d="M180 420V440"></path>
			<rect class="viz-node viz-node--focus" x="124" y="446" width="112" height="28" rx="3"></rect>
			<text class="viz-callout" x="180" y="464" text-anchor="middle">W<tspan baseline-shift="sub">O</tspan> · d × d</text>
			<path class="viz-forward" d="M180 474V484"></path>
			<rect class="viz-node viz-node--output" x="112" y="486" width="136" height="22" rx="3"></rect>
			<text class="viz-callout" x="180" y="502" text-anchor="middle">output · n × d</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> every head receives Q, K, and V for all n tokens; only the d-wide feature axis is split into dₕ = d/h slices. The h outputs concatenate back to n × d, then W<sub>O</sub> mixes information across heads.</figcaption>
</figure>

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
