---
title: "Transformer architecture: a senior-level mental model"
description: "Strip away the diagram clutter. A transformer is a stack of (residual + LayerNorm + (attention or FFN)) blocks. Understanding why each piece is there is more important than memorizing the diagram."
date: "2025-11-05"
draft: false
tags: ["concepts"]
category: "concepts"
---


## One-line definition

A transformer stacks blocks of (1) self-attention layers mixing information across positions, and (2) feed-forward layers mixing information across features, bound by residual connections and normalization.

## Why it matters

The transformer is the dominant ML architecture of 2017-2026. It powers nearly every modern NLP system, most computer vision systems, all current LLMs, and increasingly recommendation, biology, and reinforcement learning systems. Understanding *why* it works (not just what it does) is foundational.

## The minimum mental model

A transformer block is:

```
x = x + Attention(LayerNorm(x))   # mix across positions
x = x + FFN(LayerNorm(x))         # mix across features
```

Stack L of these. Add an embedding layer at the bottom and an output projection at the top. That's it.

The two information-mixing operations:
- **Attention**: a learned, content-dependent weighted sum across positions. The weights come from a (query, key) dot product.
- **FFN**: a position-wise MLP, typically `Linear → activation (GELU/SwiGLU) → Linear`. Same weights applied at every position.

Plus the operational glue:
- **Residual connections**: `x = x + Sublayer(x)`. Without these, transformers don't train at depth.
- **LayerNorm (pre-norm or post-norm)**: keeps activations in a sane range across layers.
- **Positional encoding**: attention is permutation-invariant; the model needs position information injected somewhere. Original was sinusoidal; modern is RoPE (rotary), ALiBi, or learned.

## What an interviewer expects you to articulate

If asked "explain transformers":

1. The block structure: attention + FFN, both wrapped in residual + LayerNorm.
2. Self-attention math: Q, K, V from input projections; attention = softmax(QK^T / sqrt(d)) V.
3. Multi-head: split d into H heads, run attention per head in parallel, concatenate.
4. Positional encoding: attention is permutation-invariant, so position information is added; mention RoPE for modern systems.
5. Pre-norm vs post-norm; mention pre-norm is the modern default.
6. Why each piece is there (residual for gradient flow, LayerNorm for stability, multi-head for diverse attention patterns).

If you can additionally discuss the *scale story* (more parameters + more data + more compute = monotonically better, per Kaplan/Hoffmann/Chinchilla scaling laws) and the production reality (FlashAttention, KV cache, RoPE, GQA), you're at strong-senior depth.

## The why behind each piece

### Why attention?

The naive way to build a sequence model is RNNs (sequential) or convolutions (local). Both have limits: RNNs can't parallelize across time, CNNs have limited receptive field per layer. Attention is *parallelizable across positions* (everything is a matmul) and has *unlimited receptive field per layer* (every position attends to every other). These two properties are why transformers won.

### Why multi-head?

A single attention head can attend to one pattern at a time. Multi-head allows the model to attend to multiple patterns simultaneously, one head might track syntactic dependencies, another might track topic, another might track entity references. Empirically, even with no labels, different heads end up learning different patterns.

The d_model dimension is split across H heads, so multi-head adds no parameters or FLOPs over single-head with the same total d. It's purely a structural inductive bias.

### Why residual connections?

Two reasons. (1) Gradient flow: residuals create direct paths from any layer to the loss, preventing vanishing gradients in deep networks. (2) Identity preservation: the model can learn to "do nothing" by setting the sublayer output to zero, which means adding more layers is at worst a no-op, you can always train deeper.

### Why LayerNorm (not BatchNorm)?

BatchNorm normalizes across the batch dimension, which doesn't work for variable-length sequences and breaks at small batch sizes. LayerNorm normalizes per-token across the feature dimension, independently of other tokens or other examples. This works for transformers because tokens within a sequence are already a "batch" of features for normalization purposes. See [BatchNorm vs LayerNorm](/concepts/batchnorm-vs-layernorm/).

### Why pre-norm vs post-norm?

The original transformer used post-norm: `LN(x + Sublayer(x))`. This is unstable to train at depth without careful warmup.

Pre-norm: `x + Sublayer(LN(x))`. The norm is *inside* the residual, so the residual passes the raw activation through. Easier to train, more stable at scale. Slightly worse final quality in some settings, but the training stability win dominates at scale. Almost all modern transformers (GPT, LLaMA, Mistral, etc.) use pre-norm.

### Why FFN?

Attention mixes across positions but doesn't mix across features within a position (each output dim is a weighted sum of input dims with the same weights for all positions). FFN does the per-position feature mixing. The two operations are complementary.

FFN intermediate dimension is typically 4&times; d_model (the "expansion factor"). Modern variants use SwiGLU instead of GELU, which uses three matrices instead of two and effectively gives a multiplicative gating mechanism.

### Why positional encoding?

Attention with no position info is permutation-invariant: shuffling the input tokens gives the same outputs (in different positions). For sequence modeling, position matters.

Solutions:
- **Sinusoidal** (original): fixed sinusoidal patterns added to embeddings. Works, but limited extrapolation to longer sequences.
- **Learned**: learnable position embeddings. Works, but doesn't extrapolate at all beyond training length.
- **Rotary (RoPE)**: rotate Q and K by position-dependent angles before the attention dot product. Encodes *relative* position naturally. Better long-context extrapolation than learned. Used in LLaMA, Mistral, and most modern open LLMs.
- **ALiBi**: bias the attention scores by a function of relative position. No additional parameters, good extrapolation. Used in MPT and some other models.

Modern default in 2026: **RoPE**. If you're discussing transformers and mention sinusoidal positional encoding without acknowledging RoPE, you're showing your knowledge stops in 2020.

## The scale story

The transformer's biggest property isn't any one design choice. It's that **performance scales smoothly with parameters, data, and compute** ([Kaplan et al. 2020](https://arxiv.org/abs/2001.08361), [Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)). This is what made GPT-3, GPT-4, Claude, Gemini, etc. possible.

Roughly: loss `L ~ A * N^-alpha + B * D^-beta` where N is parameters, D is tokens, with empirically-determined alpha, beta. Optimal allocation: scale N and D roughly proportionally (Chinchilla compute-optimal scaling).

This scaling property is the most important fact about transformers that wasn't obvious in 2017 and is foundational to the modern field.

## What's changed since 2017

The original architecture has evolved:

- **Pre-norm** instead of post-norm.
- **RoPE / ALiBi** instead of sinusoidal positional encoding.
- **GQA / MQA** instead of standard MHA (saves KV cache memory).
- **SwiGLU** instead of GELU (better quality at marginal compute cost).
- **RMSNorm** instead of LayerNorm in some models (LLaMA family).
- **No bias terms** in linear layers (small simplification, no quality cost).
- **No dropout** during pretraining at scale.

If you describe a "transformer" in 2026 using only the 2017 paper, you're behind. Reference the modern stack.

## Common confusions

- **"Transformers don't have inductive bias."** They have *less* inductive bias than CNNs/RNNs, but they have plenty (residuals, LayerNorm, multi-head, position encoding). The right framing is "scale-friendly architecture with weak structural priors."
- **"Attention is the most expensive part."** For long sequences, yes. For typical transformer training (sequence ~1K-4K), the FFN is comparable or larger.
- **"All transformers are GPT-style."** No. Encoder-only (BERT), encoder-decoder (T5), decoder-only (GPT). Different objectives, different mask patterns.
- **"You need positional encoding."** True for vanilla attention; some recent models (eg. with implicit RoPE) blur this.

---

*Related: [BatchNorm vs LayerNorm](/concepts/batchnorm-vs-layernorm/), [FlashAttention](/concepts/flashattention/), [KV cache](/concepts/kv-cache/).*
