---
title: "Speculative decoding"
description: "Break the autoregressive serial bottleneck without changing the output distribution. 2-3× inference speedup, free."
date: "2026-02-13"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

A small "draft" model proposes K candidate tokens cheaply; a single parallel forward pass of the large target model verifies them using a rejection-sampling rule that **provably preserves the target model's output distribution**.

LLM decoding is autoregressive: each token depends on the previous, so the GPU sits idle most of the time waiting for the next sequential step. Each forward pass on a single token is **memory-bound**: you read all of the model's weights from HBM but only do one matrix-vector multiplication. Tensor cores are barely used.

Speculative decoding turns this serial work into batched work. The large model runs *one* forward pass per K-token chunk instead of K forward passes. Wall-clock latency drops 2-3&times; with no quality change.

Speculative decoding is a standard optimization for LLM serving in 2026. Major serving systems (vLLM, TGI, SGLang, TensorRT-LLM) support it.

<!-- visual:speculative-decoding-first-rejection-boundary -->
<figure class="learning-figure plot-panel visual-wide" aria-labelledby="speculative-rejection-visual-title">
	<p class="visual-kicker">Learning objective · one cycle with K = 5</p>
	<p class="visual-title" id="speculative-rejection-visual-title">Trace what survives when the target rejects the third draft token.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 760 330" role="img" aria-labelledby="speculative-rejection-svg-title speculative-rejection-svg-desc">
			<title id="speculative-rejection-svg-title">First-rejection boundary in one speculative-decoding cycle</title>
			<desc id="speculative-rejection-svg-desc">The draft model proposes five ordered tokens. One parallel target-model pass scores all five positions. Drafts one and two are accepted and committed. Draft three is rejected and replaced by a sample from the corrected residual distribution q. Drafts four and five are discarded because they follow the rejected draft prefix. The cycle commits the two accepted drafts and the replacement, then restarts after that replacement.</desc>
			<rect class="viz-plot-bg" x="12" y="12" width="736" height="306" rx="5"></rect>
			<text class="viz-axis-label" x="24" y="38">DRAFT MODEL PROPOSES SERIALLY</text>
			<text class="viz-label" x="24" y="92">draft</text>
			<text class="viz-axis-label" x="138" y="67" text-anchor="middle">position 1</text>
			<text class="viz-axis-label" x="258" y="67" text-anchor="middle">position 2</text>
			<text class="viz-axis-label" x="378" y="67" text-anchor="middle">position 3</text>
			<text class="viz-axis-label" x="498" y="67" text-anchor="middle">position 4</text>
			<text class="viz-axis-label" x="618" y="67" text-anchor="middle">position 5</text>
			<rect class="viz-node viz-node--input" x="93" y="76" width="90" height="42" rx="4"></rect>
			<rect class="viz-node viz-node--input" x="213" y="76" width="90" height="42" rx="4"></rect>
			<rect class="viz-node viz-node--input" x="333" y="76" width="90" height="42" rx="4"></rect>
			<rect class="viz-node" x="453" y="76" width="90" height="42" rx="4" style="stroke-dasharray:5 3"></rect>
			<rect class="viz-node" x="573" y="76" width="90" height="42" rx="4" style="stroke-dasharray:5 3"></rect>
			<text class="viz-node-label" x="138" y="102">draft 1</text>
			<text class="viz-node-label" x="258" y="102">draft 2</text>
			<text class="viz-node-label" x="378" y="102">draft 3</text>
			<text class="viz-node-label" x="498" y="102">draft 4</text>
			<text class="viz-node-label" x="618" y="102">draft 5</text>
			<path d="M81 132H675" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
			<text class="viz-axis-label" x="378" y="148" text-anchor="middle">ONE TARGET PASS SCORES ALL FIVE POSITIONS IN PARALLEL</text>
			<path d="M138 119V165" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2"></path>
			<path d="M258 119V165" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2"></path>
			<path d="M378 119V165" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:3"></path>
			<path d="M498 119V165" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:4 3"></path>
			<path d="M618 119V165" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:4 3"></path>
			<text class="viz-callout" x="138" y="181" text-anchor="middle">ACCEPT</text>
			<text class="viz-callout" x="258" y="181" text-anchor="middle">ACCEPT</text>
			<text class="viz-callout" x="378" y="181" text-anchor="middle">REJECT</text>
			<text class="viz-label" x="498" y="181" text-anchor="middle">DISCARD</text>
			<text class="viz-label" x="618" y="181" text-anchor="middle">DISCARD</text>
			<path d="M318 55V294" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.5;stroke-dasharray:5 4"></path>
			<text class="viz-gradient-label" x="326" y="213">FIRST REJECTION</text>
			<text class="viz-label" x="24" y="252">commit</text>
			<rect class="viz-node viz-node--output" x="93" y="226" width="90" height="42" rx="4"></rect>
			<rect class="viz-node viz-node--output" x="213" y="226" width="90" height="42" rx="4"></rect>
			<rect class="viz-node viz-node--focus" x="333" y="226" width="90" height="42" rx="4"></rect>
			<text class="viz-node-label" x="138" y="252">draft 1</text>
			<text class="viz-node-label" x="258" y="252">draft 2</text>
			<text class="viz-node-label" x="378" y="245">replacement</text>
			<text class="viz-node-value" x="378" y="260">sample from q</text>
			<text class="viz-axis-label" x="498" y="240" text-anchor="middle">cycle stops here;</text>
			<text class="viz-axis-label" x="498" y="256" text-anchor="middle">restart after replacement</text>
			<path d="M108 286H408" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2"></path>
			<path d="M408 281L422 286L408 291Z" style="fill:var(--viz-output-stroke)"></path>
			<text class="viz-axis-label" x="258" y="307" text-anchor="middle">COMMITTED OUTPUT STILL FOLLOWS TARGET M</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> scan left to right. The accepted prefix stays. At the first rejection, replace that draft from the corrected distribution q; discard every later draft because each was conditioned on a prefix that no longer exists.</figcaption>
</figure>

## The mechanism

Setup: large target model **M** (the one whose distribution we want to sample from), small draft model **m** (cheap; e.g., a 1B distilled version of a 70B M).

Per cycle:

1. **Draft.** Run m autoregressively for K steps to propose tokens x&#770;&#8321; ... x&#770;&#7548;. This is cheap because m is small.
2. **Verify.** Run M *once* on the K-token prefix in parallel. This gives M's distribution at every position: p&#7704;(&middot; | prefix, x&#770;&#8321; ... x&#770;&#7522;) for i = 0 ... K-1.
3. **Accept/reject.** Sweep i = 1 ... K:
   - Accept x&#770;&#7522; with probability **&alpha; = min(1, p&#7704;(x&#770;&#7522;) / p&#7521;(x&#770;&#7522;))**.
   - On the first reject at position i\*: resample a new token from the corrected distribution **q(x) = normalize(max(0, p&#7704;(x) &minus; p&#7521;(x)))**. Discard x&#770;&#7522;\*&#8330;&#8321; ... x&#770;&#7548; and continue from the new token.
4. **Bonus token.** If all K drafts are accepted, sample one extra token from p&#7704;(&middot; | full prefix). So a perfect cycle yields K+1 accepted tokens for the cost of one M forward pass.

## Why it's lossless

The accept/reject rule is a special case of rejection sampling chosen specifically so that the marginal distribution of the output token at each position is *exactly* p&#7704;. The proof is one page of careful algebra; the upshot is **the output stream is statistically indistinguishable from sampling from M directly**.

This is the key selling point. Unlike quantization or distillation, speculative decoding is not a quality/speed trade-off, it's pure free speedup. (At least in theory; in practice, numerical issues, KV-cache subtleties, and tokenizer mismatches can introduce tiny deviations.)

## Speedup analysis

Let &alpha;&#770; = expected acceptance rate per token (a property of how well m mimics M).

Average accepted tokens per cycle: roughly **(1 &minus; &alpha;&#770;&#7585;&#8314;&#185;) / (1 &minus; &alpha;&#770;)**: plus the bonus token.

Cost per cycle: K &times; cost(m) + 1 &times; cost(M).

If cost(m) &laquo; cost(M), the wall-clock speedup approximately equals the average accepted tokens per cycle. In practice:

- &alpha;&#770; &asymp; 0.6-0.8 with a well-distilled draft model → 2-3&times; speedup typical.
- &alpha;&#770; &asymp; 0.85+ for code generation (high agreement on syntax) → 4-5&times; possible.
- &alpha;&#770; &asymp; 0.4 with a poorly-matched draft → speedup &lt; 1.5&times;, sometimes negative.

The choice of K matters: too small, you don't amortize the M forward pass; too large, late-position drafts almost always get rejected. K = 4-8 is the typical sweet spot.

## Variants

- **Self-speculation / Medusa.** M itself produces K drafts via extra prediction heads attached to its top layer. No separate draft model needed. Lower &alpha;&#770; than a dedicated draft, but no extra model to maintain.
- **EAGLE.** Trains a small "feature regressor" on top of M's hidden states that predicts the next token's hidden state cheaply. Better &alpha;&#770; than vanilla self-speculation.
- **Tree speculation.** m proposes a *tree* of candidates (multiple branches at each step), M verifies all branches in one batched pass, the longest accepted prefix is kept. Higher per-cycle yield at the cost of more verifier work.
- **Lookahead decoding.** No draft model at all; uses parallel n-gram speculation. Lower speedup but trivially deployable.

In 2026, EAGLE-2 / EAGLE-3 are SOTA; Medusa is the simplest to implement; tree speculation is what high-end serving systems use under the hood.

## What an interviewer expects you to say

If asked about speculative decoding:

1. Frame the problem: decoding is memory-bound, GPUs are idle, the n in the matmul is 1.
2. Explain draft + verify + accept/reject. **M's verify pass adds little cost** when weight loading dominates the extra matrix multiplication.
3. State that it's **lossless** (preserves M's distribution) and explain why this is non-obvious and important.
4. Quote a realistic speedup number (2-3&times;).
5. Bonus: mention Medusa, EAGLE, or tree speculation as variants.

## Common confusions

- **"It's an approximation."** No. The output distribution is exactly p&#7704;. (Modulo floating-point.)
- **"It only helps for greedy decoding."** No, it works for sampling too, the rejection rule is *defined* in terms of probabilities precisely because sampling is the general case.
- **"It needs the draft model to be fine-tuned to match the target."** Helpful but not required. Even a much smaller off-the-shelf model can give &alpha;&#770; &asymp; 0.6.
- **"You can use any small model as the draft."** The draft must use the **same tokenizer** as the target. Tokenizer mismatch is a common deployment pitfall.
- **"It saves FLOPs."** No, it does *more* FLOPs (the wasted draft tokens that get rejected, plus the K-token verification pass). The wins are wall-clock and GPU utilization.

## Why interviewers care

This question tests whether you understand:
1. Why decoding is memory-bound (the most important fact about LLM inference).
2. The difference between batched and sequential workloads on GPU.
3. Lossless vs. lossy optimizations (a common confusion).
4. That you've kept up with serving developments since 2023.

If you can also discuss how speculative decoding interacts with KV-cache, batching, and continuous batching, you're at a level the interviewer probably wants to hire.

## Reading list

- *Fast Inference from Transformers via Speculative Decoding* [(Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192)
- *Accelerating Large Language Model Decoding with Speculative Sampling* [(Chen et al., 2023)](https://arxiv.org/abs/2302.01318)
- *Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads* [(Cai et al., 2024)](https://arxiv.org/abs/2401.10774)
- *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* [(Li et al., 2024)](https://arxiv.org/abs/2401.15077)

---

*Related: [FlashAttention](/concepts/flashattention/) and [production LLM inference design](/questions/design-production-llm-inference-service/).*
