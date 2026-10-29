---
title: "Sequence packing with block-diagonal masks"
description: "Pack short examples without cross-example attention. Separate padding savings, position handling, loss masking, and measured throughput."
date: "2026-01-25"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Sequence packing concatenates multiple training examples back-to-back into a single fixed-length sequence and uses a block-diagonal attention mask so each example only attends within itself, eliminating the FLOPs and memory wasted on padding tokens.

Most NLP corpora have heavily skewed length distributions: many short examples, few long ones. With naive padding to the longest example in the batch, the wasted-token ratio is

$$
1 - \frac{\bar{\ell}}{\ell_{\max}}
$$

This ratio counts padded token positions, not total attention FLOPs or end-to-end throughput.
Dense padded implementations can still compute on masked positions; efficient variable-length kernels can avoid that work.
How much packing helps depends on lengths, bucketing, kernels, and the rest of the training step.

## The mechanism

1. Pick a fixed packed length $L$ (e.g., 8192).
2. Concatenate examples $e_1, e_2, \dots$ until adding the next would exceed $L$. Record the boundaries (cumulative sequence lengths, often called `cu_seqlens`).
3. Build a **block-diagonal attention mask**: token $i$ in example $e_a$ cannot attend to any token in $e_b \neq e_a$.
4. Compute attention with a kernel that respects `cu_seqlens` (FlashAttention-2 supports this natively via the `varlen` API).
5. Apply the loss only on response tokens within each example (mask the boundaries and any prompt tokens for SFT).

The worked construction resets position IDs at each boundary.
The block-diagonal mask provides attention isolation; position handling must separately match the model and kernel contract.

**Learning objective:** map packed-example boundaries to reset position IDs and a causal block-diagonal attention mask that permits no cross-example attention.

<!-- visual:packed-sequence-mask-boundaries -->
<figure class="learning-figure" aria-labelledby="packed-sequence-mask-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="packed-sequence-mask-title">Earlier physical position does not grant attention across example boundaries</p>
	<div class="visual-grid--two" role="group" aria-label="A six-token packed row and its six-by-six causal block-diagonal attention mask">
		<section class="visual-panel" aria-labelledby="packed-row-title">
			<h4 id="packed-row-title">1 · PACK TOKENS, KEEP BOUNDARIES</h4>
			<table class="cm-grid" aria-label="Six packed tokens from examples e1, e2, and e3">
				<thead><tr><th scope="col">slot</th><th scope="col">0</th><th scope="col">1</th><th scope="col">2</th><th scope="col">3</th><th scope="col">4</th><th scope="col">5</th></tr></thead>
				<tbody>
					<tr><th scope="row">example</th><td>e1</td><td>e1</td><td>e1</td><td>e2</td><td>e2</td><td>e3</td></tr>
					<tr><th scope="row">position</th><td>0</td><td>1</td><td>2</td><td>0</td><td>1</td><td>0</td></tr>
				</tbody>
			</table>
			<p><strong>Boundaries: `cu_seqlens = [0, 3, 5, 6]`</strong><br />The intervals [0, 3), [3, 5), and [5, 6) recover the three examples without padding.</p>
		</section>
		<section class="visual-panel" aria-labelledby="packed-mask-title">
			<h4 id="packed-mask-title">2 · MASK BY EXAMPLE AND TIME</h4>
			<table class="cm-grid" aria-label="Causal attention mask with query tokens as rows and key tokens as columns; A means allowed, F means a future token in the same example, and X means a different example">
				<thead><tr><th scope="col">q \ k</th><th scope="col">e1:0</th><th scope="col">e1:1</th><th scope="col">e1:2</th><th scope="col">e2:0</th><th scope="col">e2:1</th><th scope="col">e3:0</th></tr></thead>
				<tbody>
					<tr><th scope="row">e1:0</th><td class="cm-selected">A</td><td>F</td><td>F</td><td>X</td><td>X</td><td>X</td></tr>
					<tr><th scope="row">e1:1</th><td class="cm-selected">A</td><td class="cm-selected">A</td><td>F</td><td>X</td><td>X</td><td>X</td></tr>
					<tr><th scope="row">e1:2</th><td class="cm-selected">A</td><td class="cm-selected">A</td><td class="cm-selected">A</td><td>X</td><td>X</td><td>X</td></tr>
					<tr><th scope="row">e2:0</th><td>X</td><td>X</td><td>X</td><td class="cm-selected">A</td><td>F</td><td>X</td></tr>
					<tr><th scope="row">e2:1</th><td>X</td><td>X</td><td>X</td><td class="cm-selected">A</td><td class="cm-selected">A</td><td>X</td></tr>
					<tr><th scope="row">e3:0</th><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td class="cm-selected">A</td></tr>
				</tbody>
			</table>
			<p><strong>A = attend · F = future · X = other example</strong><br />Only the three lower-triangular diagonal blocks participate in attention.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> first use the cumulative boundaries to split the physical row into logical examples and restart positions at each boundary. Then read each mask row as one query: it may attend backward inside its own example (A), never forward in causal training (F), and never across an example boundary (X). Packing removes padding; boundaries preserve the same attention problem each example had before concatenation. Loss masking is a separate per-token decision. Original worked example informed by <a href="https://arxiv.org/abs/2107.02027">Krell et al. (2021)</a> and the <a href="https://github.com/Dao-AILab/flash-attention">FlashAttention variable-length interface</a>. <span class="visual-reader-check" data-reader-check="packed-sequence-mask-boundaries"><strong>Check:</strong> Why may e2:1 attend to e2:0 but not to the earlier physical slot e1:2?</span></figcaption>
</figure>

## Numbers

For the illustrated lengths `[3,2,1]`, padding three examples to length three allocates nine token positions for six real tokens.
The six-token packed row has no slack, reducing physical positions from nine to six.
That 1.5x position ratio is not a measured GPU speedup.

Compare packed and unpacked steps with matched examples, labels, position semantics, and loss weighting.
Benchmark tokens per second, peak memory, and task quality on the actual model, precision, hardware, and kernel.
Packing can otherwise change per-example weighting even when the attention mask is correct.

## Common implementation pitfalls

- **Confusing positions with isolation.** A correct block-diagonal mask forbids cross-example attention regardless of numbering. Standard fixed-frequency RoPE preserves within-example attention under a common position shift; absolute embeddings or other scaling rules may not. Follow the specific model/API contract.
- **Wrong target boundaries.** Mask any next-token target that would train one example to predict the first token of another. Attention masking and label masking solve different problems.
- **Mixing prompt and response in SFT without masking.** For SFT, mask out prompt tokens from the loss within each packed example.
- **Assuming the mask saves computation.** A dense implementation can still materialize the full $L \times L$ matrix. Use a boundary-aware kernel and measure end-to-end savings rather than inferring them from zero attention weights.

**Predict before revealing:** keep the causal block-diagonal mask, but change positions from `[0,1,2,0,1,0]` to `[0,1,2,3,4,5]`.
With standard fixed-frequency RoPE, does this create cross-example attention?

<details><summary>Identify which contract changed</summary>

No. The mask still excludes other examples.
Each example received only a common position shift, so relative rotary positions and its attention weights are unchanged.
Removing the mask would break isolation even with reset positions.
Do not extend this invariance claim to arbitrary positional embeddings or scaling variants.

</details>

## When not to pack

- Very long single examples that fill or exceed $L$ on their own (no concatenation possible; padding is already minimal).
- When examples have inter-document context that should attend across boundaries (rare).
- During inference, use the serving engine's multi-request batching and KV-cache contracts; training-style concatenation is not a drop-in replacement.

## Related

- [FlashAttention](/concepts/flashattention/). The underlying kernel that makes varlen attention efficient.
- [Gradient accumulation](/concepts/gradient-accumulation/). Orthogonal way to grow effective batch size.
