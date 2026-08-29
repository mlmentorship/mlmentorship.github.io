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
- **RAG-style architectures**: model output cross-attends to retrieved document representations.

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
- **KV-cache**: at decoder inference, both self- and cross-attention have a KV-cache. The cross-attention cache is computed once from the encoder output and reused for every decoded token, which makes it cheaper than the self-attention cache that grows with the decoded sequence.

## Variants

- **Causal cross-attention**: rare. The query side is causally masked but $K, V$ are not from the same sequence, so causality is moot.
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
