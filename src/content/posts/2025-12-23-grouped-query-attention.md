---
title: "Grouped-query and multi-query attention (GQA, MQA)"
description: "Share K and V heads across query heads to shrink the KV cache 4-8x with negligible quality loss. Standard in modern decoder LLMs."
date: "2025-12-23"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["GQA", "MQA", "multi-query attention", "grouped query attention"]
---

## Summary

GQA and MQA reduce the number of distinct K/V projection heads while keeping the full set of Q heads, so multiple query heads share the same key and value tensors. MQA is the extreme case: one K/V head total. GQA picks an intermediate number of K/V groups.

The KV cache dominates LLM serving memory at long contexts (see [KV cache](/concepts/kv-cache/)). Cutting the number of K/V heads cuts cache size proportionally:

- **MHA** (standard, e.g. GPT-3): K and V have the same number of heads as Q.
- **MQA** [(Shazeer, 2019)](https://arxiv.org/abs/1911.02150): 1 K and 1 V head shared across all Q heads. ~`num_heads`× smaller cache.
- **GQA** [(Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245): G groups, each shared across `num_heads / G` Q heads. Tunable midpoint.

Llama 2 70B uses GQA with 8 K/V groups for 64 query heads (8× cache reduction). Llama 3, Mistral, Qwen, and most modern decoders default to GQA.

## The mechanism

In standard multi-head attention, for each head $h$:
$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d}}\right) V_h
$$
with $Q_h, K_h, V_h \in \mathbb{R}^{n \times d}$ and $h \in \{1, \dots, H\}$.

In GQA with $G$ groups, the $H$ query heads are partitioned into $G$ contiguous groups of size $H/G$. All query heads in the same group attend to the same shared $K_g, V_g$. MQA is GQA with $G = 1$.

Implementation: project K and V to dimension $G \cdot d$ instead of $H \cdot d$, then broadcast (repeat) across the matching Q heads before the matmul.

<!-- visual:gqa-eight-queries-two-kv-groups -->
<figure class="learning-figure" aria-labelledby="gqa-sharing-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="gqa-sharing-title">Which heads are shared when 8 query heads use 2 K/V heads?</p>
	<div class="visual-grid--two" role="group" aria-label="Worked grouped-query attention example and KV-cache comparison">
		<section class="visual-panel" aria-labelledby="gqa-map-title">
			<h4 id="gqa-map-title">The 8 query heads stay distinct</h4>
			<p>Each group reuses one key head and one value head; it does not merge its queries.</p>
			<table class="cm-grid" aria-label="Mapping from eight query heads to two shared key and value head pairs">
				<thead><tr><th scope="col">Group</th><th scope="col">Query heads</th><th scope="col">Shared pair</th></tr></thead>
				<tbody>
					<tr><th scope="row">1</th><td>Q<sub>1</sub>, Q<sub>2</sub>, Q<sub>3</sub>, Q<sub>4</sub></td><td class="cm-selected"><strong>K<sub>1</sub> + V<sub>1</sub></strong>reused 4×</td></tr>
					<tr><th scope="row">2</th><td>Q<sub>5</sub>, Q<sub>6</sub>, Q<sub>7</sub>, Q<sub>8</sub></td><td class="cm-selected"><strong>K<sub>2</sub> + V<sub>2</sub></strong>reused 4×</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">H/G = 8 queries ÷ 2 groups = 4 queries per pair</p>
		</section>
		<section class="visual-panel" aria-labelledby="gqa-cache-title">
			<h4 id="gqa-cache-title">Only cached K/V head count shrinks</h4>
			<p>Holding sequence length and head dimension fixed, cache size scales with the number of K/V pairs.</p>
			<table class="cm-grid" aria-label="Comparison of query heads, cached key and value head pairs, and relative KV-cache size">
				<thead><tr><th scope="col">Variant</th><th scope="col">Q heads</th><th scope="col">K/V pairs</th><th scope="col">Cache</th></tr></thead>
				<tbody>
					<tr><th scope="row">MHA</th><td>8</td><td>8</td><td>8/8 = 1×</td></tr>
					<tr><th scope="row">GQA</th><td><strong>8</strong></td><td class="cm-selected"><strong>2</strong></td><td class="cm-selected"><strong>2/8 = ¼×</strong>4× smaller</td></tr>
					<tr><th scope="row">MQA</th><td>8</td><td>1</td><td>1/8 = ⅛×</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">Q count: unchanged · cached K/V pairs: 8 → 2</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> read across the left table first: Q<sub>1</sub> through Q<sub>4</sub> remain four separate query heads but all use K<sub>1</sub> and V<sub>1</sub>; the next four queries use K<sub>2</sub> and V<sub>2</sub>. Then compare counts on the right: caching 2 K/V pairs instead of 8 makes the head-dependent cache one quarter as large without removing any query heads. MQA takes the same sharing idea to one K/V pair. Original comparison checked against the primary <a href="https://arxiv.org/abs/2305.13245">GQA paper</a> and <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html">PyTorch GQA documentation</a>.</figcaption>
</figure>

## Tradeoffs

| Variant | KV heads | Cache size | Quality | Used by |
|---------|---------|-----------|---------|---------|
| MHA | $H$ | 1× | baseline | GPT-3, original Llama |
| GQA-8 | 8 | $H/8$× | ~baseline | Llama 2/3 70B, Mistral |
| MQA | 1 | $1/H$× | small drop | PaLM, Falcon |

GQA recovers nearly all MHA quality while keeping most of MQA's cache savings. The dominant choice in 2026.

## Common pitfalls

- **Confusing K/V heads with Q heads.** GQA shrinks K/V only; Q stays full-rank.
- **Assuming the speedup is in compute.** GQA mostly saves *memory* (cache + bandwidth), not FLOPs. The matmul cost barely changes.
- **Re-training cost.** You generally cannot convert MHA → GQA post-hoc; the K/V projections were trained per-head. Distillation or partial re-training is required.
