---
title: "Mixture of Experts (MoE)"
description: "Replace one large feed-forward block with N smaller experts and a router that activates only k of them per token. Trades parameter count for compute."
date: "2026-03-27"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A Mixture-of-Experts layer replaces a single dense feed-forward network with $N$ parallel "expert" FFNs and a router that sends each token to the top-$k$ experts (typically $k=1$ or $k=2$). Total parameters scale with $N$; per-token compute scales with $k$.

The defining tradeoff: a $k$-of-$N$ MoE has roughly the same per-token FLOPs as a dense model with $k/N$ of the parameters, but the *capacity* of all $N$ experts. Mixtral 8×7B [(Jiang et al., 2023)](https://arxiv.org/abs/2401.04088) has 47B total parameters and uses ~13B per token. Quality close to a 70B dense model at ~5× lower inference compute.

MoE is the dominant strategy for scaling parameter count beyond what dense training and inference can afford. GPT-4, Mixtral, DeepSeek-V3, Grok 1, and many other 2024-2026 frontier models are MoE.

## The mechanism

For each transformer block, replace the single FFN with:

1. **Router**: a small linear layer $W_r \in \mathbb{R}^{d \times N}$. For each token, compute logits $W_r x$, take top-$k$ experts, and softmax-normalize over those $k$.
2. **Experts**: $N$ independent FFN blocks $\{E_1, \dots, E_N\}$, each the same shape as the dense FFN it replaces.
3. **Combine**: output is $\sum_{i \in \text{top-}k} g_i \cdot E_i(x)$ where $g_i$ is the router weight.

Attention layers are typically *not* MoE (shared across all tokens).

<!-- visual:moe-top-two-routing-accounting -->
<figure class="learning-figure" aria-labelledby="moe-routing-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="moe-routing-title">How can one token use only two experts while the model stores all four?</p>
	<div class="visual-grid--two" role="group" aria-label="Top-2 expert routing and resource accounting">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 280" role="img" aria-labelledby="moe-route-svg-title moe-route-svg-desc">
				<title id="moe-route-svg-title">One token routed to experts one and three</title>
				<desc id="moe-route-svg-desc">Token x enters a router that scores four experts. After top-2 selection and softmax, solid paths send x to expert one with weight 0.65 and expert three with weight 0.35. Dashed paths to experts two and four are marked skipped with gate weight zero.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="245" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">1 · ROUTE THIS TOKEN</text>
				<rect class="viz-node viz-node--input" x="20" y="112" width="48" height="42" rx="5"></rect>
				<text class="viz-node-label" x="44" y="131">x</text>
				<text class="viz-node-value" x="44" y="146">token</text>
				<rect class="viz-node viz-node--focus" x="91" y="88" width="78" height="90" rx="7"></rect>
				<text class="viz-node-label" x="130" y="113">router</text>
				<text class="viz-node-value" x="130" y="132">score → top 2</text>
				<text class="viz-node-value" x="130" y="150">softmax selected</text>
				<path d="M68 133H91" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
				<rect class="viz-node viz-node--focus" x="222" y="35" width="58" height="38" rx="5"></rect>
				<rect class="viz-node" x="222" y="94" width="58" height="38" rx="5"></rect>
				<rect class="viz-node viz-node--focus" x="222" y="153" width="58" height="38" rx="5"></rect>
				<rect class="viz-node" x="222" y="212" width="58" height="38" rx="5"></rect>
				<text class="viz-node-label" x="251" y="59">E1</text>
				<text class="viz-node-label" x="251" y="118">E2</text>
				<text class="viz-node-label" x="251" y="177">E3</text>
				<text class="viz-node-label" x="251" y="236">E4</text>
				<path d="M169 104L222 57M169 145L222 172" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3"></path>
				<path d="M169 126L222 113M169 163L222 231" style="fill:none;stroke:var(--c-muted);stroke-width:1.5;stroke-dasharray:4 4"></path>
				<text class="viz-callout" x="184" y="67">0.65</text>
				<text class="viz-label" x="184" y="108">0 · skip</text>
				<text class="viz-callout" x="184" y="166">0.35</text>
				<text class="viz-label" x="184" y="220">0 · skip</text>
				<text class="viz-axis-label" x="150" y="263" text-anchor="middle">SOLID = EXECUTE · DASHED = BYPASS</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 280" role="img" aria-labelledby="moe-account-svg-title moe-account-svg-desc">
				<title id="moe-account-svg-title">Two active experts but four resident parameter sets</title>
				<desc id="moe-account-svg-desc">All four expert parameter blocks remain resident in memory. Expert one and expert three are labeled active and feed a weighted sum. Expert two and expert four are labeled resident but skipped. The accounting states that compute uses two of four experts while memory stores four of four.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="245" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">2 · ACCOUNT FOR COMPUTE AND MEMORY</text>
				<rect class="viz-node viz-node--focus" x="18" y="44" width="58" height="54" rx="5"></rect>
				<rect class="viz-node" x="87" y="44" width="58" height="54" rx="5"></rect>
				<rect class="viz-node viz-node--focus" x="156" y="44" width="58" height="54" rx="5"></rect>
				<rect class="viz-node" x="225" y="44" width="58" height="54" rx="5"></rect>
				<text class="viz-node-label" x="47" y="67">E1</text>
				<text class="viz-node-value" x="47" y="86">active</text>
				<text class="viz-node-label" x="116" y="67">E2</text>
				<text class="viz-node-value" x="116" y="86">skipped</text>
				<text class="viz-node-label" x="185" y="67">E3</text>
				<text class="viz-node-value" x="185" y="86">active</text>
				<text class="viz-node-label" x="254" y="67">E4</text>
				<text class="viz-node-value" x="254" y="86">skipped</text>
				<path d="M47 98C47 123 102 121 122 143M185 98C185 123 161 121 143 143" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3"></path>
				<rect class="viz-node viz-node--output" x="76" y="143" width="148" height="49" rx="7"></rect>
				<text class="viz-node-label" x="150" y="163">weighted sum</text>
				<text class="viz-node-value" x="150" y="181">0.65 E1(x) + 0.35 E3(x)</text>
				<rect class="viz-node viz-node--focus" x="25" y="214" width="113" height="40" rx="5"></rect>
				<rect class="viz-node" x="162" y="214" width="113" height="40" rx="5"></rect>
				<text class="viz-node-label" x="81" y="231">compute</text>
				<text class="viz-node-value" x="81" y="246">2 of 4 execute</text>
				<text class="viz-node-label" x="218" y="231">memory</text>
				<text class="viz-node-value" x="218" y="246">4 of 4 resident</text>
				<text class="viz-axis-label" x="150" y="267" text-anchor="middle">SPARSE FLOPs · DENSE PARAMETER STORAGE</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> top-2 routing makes only E1 and E3 run for this token, and their normalized weights combine the two outputs. E2 and E4 do no work for this token, but their parameters still occupy memory because another token may select them.</figcaption>
</figure>

## Load balancing

The router will collapse to a few favorite experts unless penalized. Standard fix: an **auxiliary load-balancing loss** that penalizes uneven expert usage within a batch. Alternatives include expert-choice routing (each expert picks its top tokens, [Zhou et al., 2022](https://arxiv.org/abs/2202.09368)) and noise injection.

If experts go unused for many steps their parameters drift; a few production systems "reset" dead experts.

## Capacity and expert parallelism

With $N=8$ experts, an intuitive serving setup is **expert parallelism**: each GPU holds one expert. Tokens are routed via all-to-all communication. This works but introduces:

- **Communication overhead**: all-to-all is bandwidth-bound and stalls when the routing is imbalanced.
- **Capacity factor**: each expert has a max number of tokens it can process per batch; overflow tokens are dropped (or sent to a fallback path). Capacity factor of 1.25 is common.

## Tradeoffs vs. dense

- **Memory**: MoE needs $N \times$ FFN parameters in HBM even though only $k$ are used per token. Inference VRAM is dominated by all experts being loaded, not just active ones.
- **Throughput**: MoE wins per-FLOP. Throughput per VRAM-byte is worse than dense.
- **Quality at fixed FLOPs**: MoE generally beats dense at matched per-token FLOPs.
- **Fine-tuning**: MoE is harder to fine-tune cleanly; routing can drift, and small datasets exacerbate load imbalance.

## Common pitfalls

- **Quoting "total parameters" as if they were active.** A 47B MoE with 13B active is a 13B-FLOPs model with 47B-VRAM cost.
- **Ignoring the routing loss.** Without it, training collapses to using a few experts.
- **Assuming MoE always wins.** At small scale or with limited compute for routing experimentation, dense is simpler and competitive.
