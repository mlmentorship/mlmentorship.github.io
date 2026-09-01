---
title: "Sequence packing with block-diagonal masks"
description: "Concatenate multiple short examples into one fixed-length sequence to eliminate padding waste. The single largest throughput win for training on skewed-length corpora."
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

For C4-like web text this is often 50–80%. Padded positions cost full FLOPs and memory but contribute nothing to the loss. Sequence packing recovers nearly all of that throughput.

## The mechanism

1. Pick a fixed packed length $L$ (e.g., 8192).
2. Concatenate examples $e_1, e_2, \dots$ until adding the next would exceed $L$. Record the boundaries (cumulative sequence lengths, often called `cu_seqlens`).
3. Build a **block-diagonal attention mask**: token $i$ in example $e_a$ cannot attend to any token in $e_b \neq e_a$.
4. Compute attention with a kernel that respects `cu_seqlens` (FlashAttention-2 supports this natively via the `varlen` API).
5. Apply the loss only on response tokens within each example (mask the boundaries and any prompt tokens for SFT).

Position IDs reset at each example boundary so position 0 is the start of each packed example.

**Learning objective:** map packed-example boundaries to reset position IDs and a causal block-diagonal attention mask that permits no cross-example attention.

<!-- visual:packed-sequence-mask-boundaries -->
<figure class="learning-figure" aria-labelledby="packed-sequence-mask-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="packed-sequence-mask-title">How can one token row still behave like three separate examples?</p>
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
	<figcaption><strong>Read it this way:</strong> first use the cumulative boundaries to split the physical row into logical examples and restart positions at each boundary. Then read each mask row as one query: it may attend backward inside its own example (A), never forward in causal training (F), and never across an example boundary (X). Packing removes padding; boundaries preserve the same attention problem each example had before concatenation. Loss masking is a separate per-token decision. Original worked example informed by <a href="https://arxiv.org/abs/2107.02027">Krell et al. (2021)</a> and the <a href="https://github.com/Dao-AILab/flash-attention">FlashAttention variable-length interface</a>.</figcaption>
</figure>

## Numbers

For pretraining on web text packed at $L = 8192$:

- Wasted tokens drop from ~50% (naive batching) to <2% (just the slack at the end of the packed sequence).
- Throughput per GPU roughly doubles.
- Quality is unchanged when the mask is correct.

Most modern training stacks pack by default. Llama, Mistral, Qwen, and major SFT toolkits (axolotl, TRL) all support it.

## Common implementation pitfalls

- **Forgetting to reset positions.** If position IDs continue across boundaries, attention learns to treat packed boundaries as long-range dependencies and quality drops.
- **Wrong loss masking.** Loss must not flow from one example to another; mask boundaries explicitly.
- **Mixing prompt and response in SFT without masking.** For SFT, mask out prompt tokens from the loss within each packed example.
- **Using a kernel that doesn't support varlen.** Without FlashAttention-2 varlen (or equivalent), the block-diagonal mask materializes the full $L \times L$ matrix and you lose the speedup.

## When not to pack

- Very long single examples that fill or exceed $L$ on their own (no concatenation possible; padding is already minimal).
- When examples have inter-document context that should attend across boundaries (rare).
- During inference, where examples come one at a time.

## Related

- [FlashAttention](/concepts/flashattention/). The underlying kernel that makes varlen attention efficient.
- [Gradient accumulation](/concepts/gradient-accumulation/). Orthogonal way to grow effective batch size.
