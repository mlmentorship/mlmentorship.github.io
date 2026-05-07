---
title: "RoPE, ALiBi, and the modern positional encoding landscape"
description: "Sinusoidal positional encoding is in the original transformer paper and not in any modern LLM. Here's what replaced it and why."
date: "2026-03-15"
draft: false
tags: ["reference"]
category: "reference"
---


## One-line definition

Positional encoding gives a transformer information about token *order*, since attention itself is permutation-invariant. Modern LLMs use **rotary position embeddings (RoPE)** or **ALiBi** instead of the original sinusoidal scheme, primarily for better long-context behavior.

## Why it matters

The choice of positional encoding determines long-context performance, extrapolation beyond training length, and relative position representation. It's a small choice with outsized impact on production inference.

## The lineup

### Sinusoidal (original Transformer, 2017)

Add a fixed sinusoidal pattern to the input embeddings:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

- Encodes absolute position.
- Allows some implicit relative-position learning through linear combinations of sinusoids.
- Limited extrapolation: at positions much beyond training length, behavior degrades.

### Learned absolute positional embeddings

A learnable embedding per position. Used in BERT, GPT-2.

- Each position gets its own embedding learned during training.
- Cannot extrapolate at all to lengths beyond training.
- Larger memory cost (one embedding per position).

### Relative positional encoding

Replace absolute positions with relative offsets. T5 introduced a simple bucketed bias added to attention scores.

- Better generalization than absolute.
- Several variants (T5, Transformer-XL, Shaw et al.).

### RoPE (Rotary Position Embeddings), the modern default

Don't add positional info to embeddings. Instead, *rotate* Q and K by position-dependent angles before the attention dot product.

For position `m`, the rotation matrix `R_m` is block-diagonal with 2x2 rotations of angles `m * theta_i` for each pair of dimensions. Then:
```
Q' = R_m * Q   (Q rotated by query position)
K' = R_n * K   (K rotated by key position)
```

The dot product `Q'^T * K' = Q^T * R_{n-m} * K` depends only on the *relative* position `n - m`, not on absolute positions. So RoPE is implicitly relative.

- Encodes relative position naturally.
- Plays well with FlashAttention (rotation is an element-wise op).
- Better long-context extrapolation than sinusoidal.
- Used in: LLaMA family, Mistral, Qwen, Gemma, most modern open LLMs.

Variants exist for context extension: **NTK-aware RoPE**: **YaRN**: **PI (Position Interpolation)**: rescale the rotation frequencies to handle longer-than-trained contexts.

### ALiBi (Attention with Linear Biases)

Add a position-dependent *bias* directly to attention scores:
```
attention(Q, K) = softmax(QK^T / sqrt(d) + m * |i - j|)
```
where `m` is a per-head slope and `|i - j|` is the absolute distance between positions.

- No additional parameters; no embedding modification.
- Implicitly relative.
- Excellent extrapolation to longer sequences than seen at training time.
- Used in: MPT, BLOOM, some BERT variants.

### Position-free / implicit position

Some recent architectures (some MoE variants, certain SSM-based models) avoid explicit positional encoding by relying on the recurrence or state dynamics to inject position. Less common in transformers proper.

## What an interviewer expects you to say

If asked about positional encoding:

1. Explain *why* attention needs positional info (permutation-invariance).
2. Mention the original sinusoidal scheme as a starting point.
3. State that modern LLMs use **RoPE or ALiBi**: not sinusoidal.
4. Explain RoPE's mechanism (rotation in 2D blocks; gives relative position implicitly).
5. Discuss the long-context extrapolation issue and the techniques (NTK-aware, YaRN, PI) for extending RoPE-trained models to longer contexts.

If you describe positional encoding in 2026 using only sinusoidal, you signal your knowledge stops in 2020.

## Common confusions

- **"Sinusoidal is the standard."** It was the standard in 2017-2019 and is now obsolete in production. RoPE is the standard since ~2021.
- **"Absolute vs relative positional encoding."** A meaningful distinction. RoPE and ALiBi are both relative.
- **"Positional encoding extends the context window."** No, the model architecture and training data extend the context window. PE choices affect *how gracefully* the model handles long contexts and whether it can extrapolate.
- **"YaRN is a different positional encoding."** YaRN is a specific *adaptation* of RoPE that extends context, not a separate scheme.

## Long-context extension techniques

A practical concern: most LLMs are pretrained at modest context (e.g., 8K) but production wants much longer (32K, 128K, 1M). Three families of fix:

1. **Position interpolation (PI)**: scale position indices down so that positions in the longer context map into the trained range. Requires fine-tuning. Simple, works well up to ~4&times; extension.
2. **NTK-aware scaling**: scale RoPE frequencies non-uniformly so high-frequency dimensions are preserved (which matter for short-distance precision) and low-frequency dimensions are stretched. Better than naive PI.
3. **YaRN**: a refinement of NTK-aware that further improves long-context extrapolation, often with no fine-tuning needed.
4. **Continued pretraining at long context**: the most reliable but expensive option. Used by Anthropic, OpenAI, etc.

Long-context extension is now standard interview territory because it's a commercial differentiator in 2026.

## Why interviewers ask

Positional encoding tests:
1. Whether you've kept up with transformer evolution since 2020.
2. Whether you understand attention's permutation-invariance and why position info is needed.
3. Whether you've handled long-context concerns in production.

A senior follow-up: "How would you extend a RoPE-trained model from 8K to 128K context?" Answer: NTK-aware scaling or YaRN, possibly followed by continued pretraining on long-context data; evaluate retrieval quality (needle-in-haystack) at the new length to verify. This is a standard 2026 problem and the answer signals production fluency.

---

*Related: [Transformer architecture](/reference/transformer-architecture/), [FlashAttention](/reference/flashattention/), [KV cache](/reference/kv-cache/).*
