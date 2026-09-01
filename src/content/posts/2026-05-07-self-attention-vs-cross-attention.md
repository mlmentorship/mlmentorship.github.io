---
title: "Self-attention vs cross-attention"
description: "Self-attention reads from one sequence; cross-attention reads from another. This input choice determines encoder-only, decoder-only, and encoder-decoder structures."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Self-attention** computes $\text{softmax}(QK^\top / \sqrt{d}) V$ where $Q, K, V$ are all derived from the same sequence. **Cross-attention** uses $Q$ from one sequence and $K, V$ from another. Same kernel, different routing.

This input choice separates encoder-only models (BERT), decoder-only models (GPT), and encoder-decoder models (T5, the original Transformer, and Whisper). Multimodal architectures also use cross-attention to connect image, text, or audio representations.

If you can write the matrix multiplications and explain why a layer uses one form, you understand the attention structure of common transformer architectures.

## Self-attention

Inputs: a single sequence $X \in \mathbb{R}^{n \times d}$.

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V,
$$

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V.
$$

Each token attends to every other token (subject to masking). Used in:

- **BERT encoder layers**: bidirectional self-attention.
- **GPT decoder layers**: causal self-attention. The mask zeros out positions $j > i$ so token $i$ cannot attend to future tokens.

## Cross-attention

Inputs: a query sequence $X$ and a key-value sequence $Y$.

$$
Q = X W_Q, \quad K = Y W_K, \quad V = Y W_V.
$$

Same softmax, same scaling. The shape of the attention matrix is now $|X| \times |Y|$.

Used wherever the model needs to "look up" information from a different source:

- **Encoder-decoder transformers**: decoder layers cross-attend to the encoder output. Translation, summarization, speech-to-text.
- **Diffusion models with text conditioning**: image-side latents cross-attend to text embeddings. Stable Diffusion, DiT.
- **Perceiver / Q-Former**: a small set of learned latent queries cross-attend to a large input (image patches, audio frames) to compress it.
- **RAG architectures with separate memory**: model states can cross-attend to retrieved document representations. Many decoder-only RAG systems instead concatenate retrieved text into the prompt and use self-attention.

<!-- visual:attention-source-routing -->
<figure class="learning-figure plot-panel" aria-labelledby="attention-source-routing-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="attention-source-routing-title">Which sequence supplies Q, K, and V?</p>
	<svg viewBox="0 0 360 430" role="img" aria-labelledby="attention-source-routing-svg-title attention-source-routing-svg-desc">
		<title id="attention-source-routing-svg-title">Self-attention uses one source while cross-attention routes queries and key-values from different sources</title>
		<desc id="attention-source-routing-svg-desc">The upper panel shows one three-token sequence X supplying query, key, and value projections, producing a square three-by-three attention matrix whose rows and columns both refer to X. The lower panel shows a two-token sequence X supplying queries while a three-token sequence Y supplies keys and values, producing a rectangular two-by-three matrix. In both panels the same scaled dot-product attention kernel maps one output row to each query token.</desc>
		<defs><marker id="attention-routing-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path class="viz-arrow-forward" d="M0 0L10 5L0 10Z"></path></marker></defs>
		<rect class="viz-plot-bg" x="5" y="5" width="350" height="195" rx="4"></rect>
		<text class="viz-axis-label" x="16" y="27">SELF-ATTENTION · ONE SOURCE</text>
		<rect class="viz-node viz-node--input" x="17" y="73" width="69" height="55" rx="4"></rect><text class="viz-node-label" x="51" y="96">X</text><text class="viz-node-value" x="51" y="114">3 tokens</text>
		<path d="M86 88H120M86 100H120M86 112H120" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;marker-end:url(#attention-routing-arrow)"></path>
		<rect class="viz-node" x="122" y="50" width="55" height="30" rx="3"></rect><text class="viz-callout" x="149" y="69" text-anchor="middle">Q = XWQ</text>
		<rect class="viz-node" x="122" y="86" width="55" height="30" rx="3"></rect><text class="viz-callout" x="149" y="105" text-anchor="middle">K = XWK</text>
		<rect class="viz-node" x="122" y="122" width="55" height="30" rx="3"></rect><text class="viz-callout" x="149" y="141" text-anchor="middle">V = XWV</text>
		<path d="M177 65H215M177 101H202M177 137H215" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;marker-end:url(#attention-routing-arrow)"></path>
		<text class="viz-label" x="268" y="48" text-anchor="middle">keys: X (3 columns)</text><text class="viz-label" x="213" y="96" text-anchor="end">queries:</text><text class="viz-label" x="213" y="109" text-anchor="end">X</text><text class="viz-label" x="213" y="122" text-anchor="end">(3 rows)</text>
		<g style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1"><rect x="228" y="61" width="24" height="24"></rect><rect x="252" y="61" width="24" height="24"></rect><rect x="276" y="61" width="24" height="24"></rect><rect x="228" y="85" width="24" height="24"></rect><rect x="252" y="85" width="24" height="24"></rect><rect x="276" y="85" width="24" height="24"></rect><rect x="228" y="109" width="24" height="24"></rect><rect x="252" y="109" width="24" height="24"></rect><rect x="276" y="109" width="24" height="24"></rect></g>
		<text class="viz-callout" x="264" y="154" text-anchor="middle">scores: 3 × 3</text><text class="viz-axis-label" x="180" y="183" text-anchor="middle">each X token reads from X</text>
		<rect class="viz-plot-bg" x="5" y="210" width="350" height="215" rx="4"></rect>
		<text class="viz-axis-label" x="16" y="232">CROSS-ATTENTION · TWO SOURCES</text>
		<rect class="viz-node viz-node--input" x="17" y="254" width="69" height="48" rx="4"></rect><text class="viz-node-label" x="51" y="274">X</text><text class="viz-node-value" x="51" y="290">2 queries</text>
		<rect class="viz-node viz-node--focus" x="17" y="325" width="69" height="48" rx="4"></rect><text class="viz-node-label" x="51" y="345">Y</text><text class="viz-node-value" x="51" y="361">3 memory tokens</text>
		<path d="M86 278H121M86 341H121M86 357H121" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;marker-end:url(#attention-routing-arrow)"></path>
		<rect class="viz-node" x="122" y="263" width="55" height="30" rx="3"></rect><text class="viz-callout" x="149" y="282" text-anchor="middle">Q = XWQ</text>
		<rect class="viz-node" x="122" y="320" width="55" height="30" rx="3"></rect><text class="viz-callout" x="149" y="339" text-anchor="middle">K = YWK</text>
		<rect class="viz-node" x="122" y="356" width="55" height="30" rx="3"></rect><text class="viz-callout" x="149" y="375" text-anchor="middle">V = YWV</text>
		<path d="M177 278H215M177 335H202M177 371H215" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;marker-end:url(#attention-routing-arrow)"></path>
		<text class="viz-label" x="268" y="268" text-anchor="middle">keys: Y (3 columns)</text><text class="viz-label" x="213" y="305" text-anchor="end">queries:</text><text class="viz-label" x="213" y="318" text-anchor="end">X</text><text class="viz-label" x="213" y="331" text-anchor="end">(2 rows)</text>
		<g style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1"><rect x="228" y="281" width="24" height="24"></rect><rect x="252" y="281" width="24" height="24"></rect><rect x="276" y="281" width="24" height="24"></rect><rect x="228" y="305" width="24" height="24"></rect><rect x="252" y="305" width="24" height="24"></rect><rect x="276" y="305" width="24" height="24"></rect></g>
		<text class="viz-callout" x="264" y="350" text-anchor="middle">scores: 2 × 3</text><text class="viz-axis-label" x="264" y="372" text-anchor="middle">output: 2 rows</text>
		<text class="viz-axis-label" x="180" y="408" text-anchor="middle">each X query reads from Y · same attention kernel</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> follow the projection inputs, not the operation name. In self-attention, one sequence <code>X</code> fans out to Q, K, and V, so three query tokens scored against three key tokens form a square <code>3 × 3</code> matrix. In cross-attention, <code>X</code> supplies two queries while <code>Y</code> supplies three key-value pairs, so the same kernel forms a rectangular <code>2 × 3</code> matrix and returns one row per X query. Source labels, arrows, and matrix dimensions carry the distinction without color. Original schematic, checked against <a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a>.</figcaption>
</figure>

## Where each lives in a transformer block

Encoder block (BERT, T5 encoder):

1. Self-attention.
2. FFN.

Decoder block (GPT):

1. Causal self-attention.
2. FFN.

Encoder-decoder block (T5 decoder, original Transformer decoder):

1. Causal self-attention.
2. **Cross-attention** to encoder output.
3. FFN.

The decoder reads its own past tokens (self-attention) and the encoder's output (cross-attention) at every layer.

## Tradeoffs

- **Compute**: self-attention is $O(n^2 d)$. Cross-attention is $O(n_Q \cdot n_{KV} \cdot d)$, which can be much cheaper if $n_{KV}$ is small (compressed conditioning) or much larger (cross-attending to a long context).
- **KV-cache**: at encoder-decoder inference, the decoder can cache both forms. Cross-attention K/V are computed once from the fixed encoder output and reused for every decoded token; self-attention K/V grow with the decoded sequence.

## Variants

- **Masked cross-attention**: padding masks hide invalid K/V positions. A triangular mask is meaningful only when query and K/V positions share an ordered alignment; separate source and target sequences do not imply one automatically.
- **Cross-attention with caching**: precompute $K, V$ from a fixed conditioning sequence (system prompt, retrieved docs) and reuse across decoding steps.
- **Asymmetric cross-attention**: in Perceiver, the queries are a small learned set (e.g. 256), the K/V are massive (e.g. all image patches). The model compresses high-dim input into a fixed-size latent.

## Common pitfalls

- **Calling decoder self-attention "cross-attention."** They are different. Self-attention reads from the same sequence (the previously generated tokens); cross-attention reads from another sequence (encoder output, retrieved docs).
- **Forgetting that decoder-only LLMs do not use cross-attention.** Their conditioning is the prompt prefix, attended via self-attention, not a separate cross-attention path.
- **Conflating attention masks with attention types.** The mask shape differs (causal mask is square and triangular; cross-attention is rectangular and usually unmasked) but the operation is the same.

## Related

- [The attention mechanism](/concepts/attention-mechanism/).
- [Multi-head attention](/concepts/multi-head-attention/).
- [Transformer architecture](/concepts/transformer-architecture/).
