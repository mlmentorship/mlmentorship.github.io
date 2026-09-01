---
title: "Transformer architecture: a senior-level mental model"
description: "Strip away the diagram clutter. A transformer is a stack of (residual + LayerNorm + (attention or FFN)) blocks. Understanding why each piece is there is more important than memorizing the diagram."
date: "2025-11-05"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

A transformer stacks blocks of (1) self-attention layers mixing information across positions, and (2) feed-forward layers mixing information across features, bound by residual connections and normalization.

The transformer is the dominant ML architecture of 2017-2026. It powers nearly every modern NLP system, most computer vision systems, all current LLMs, and increasingly recommendation, biology, and reinforcement learning systems. Understanding *why* it works (not just what it does) is foundational.

## The minimum mental model

A transformer block is:

```
x = x + Attention(LayerNorm(x))   # mix across positions
x = x + FFN(LayerNorm(x))         # mix across features
```

Stack L of these. Add an embedding layer at the bottom and an output projection at the top. That's it.

**Learning objective:** distinguish attention's cross-position communication from the FFN's independent, per-position feature transformation while tracing the unchanged token-by-feature tensor shape through a pre-norm block.

<!-- visual:transformer-two-mixing-axes -->
<figure class="learning-figure plot-panel" aria-labelledby="transformer-mixing-title">
	<p class="visual-kicker">One tensor, two dependency patterns</p>
	<p class="visual-title" id="transformer-mixing-title">Which dimension does each transformer sublayer connect?</p>
	<svg viewBox="0 0 360 420" role="img" aria-labelledby="transformer-mixing-svg-title transformer-mixing-svg-desc">
		<title id="transformer-mixing-svg-title">Attention connects token positions while the feed-forward network transforms each token independently</title>
		<desc id="transformer-mixing-svg-desc">The input is a token-by-feature tensor with three token rows and four feature columns. In the unmasked encoder example in stage one, arrows from token rows one, two, and three converge on the updated representation of token two, showing that self-attention can gather information across positions allowed by its mask. In stage two, three separate horizontal paths apply the same feed-forward network to each token row, with no path between rows, showing independent feature transformation at every position. Both sublayers preserve the token-by-feature shape and add their updates through pre-norm residual connections.</desc>
		<defs>
			<marker id="transformer-mixing-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" style="fill:var(--viz-edge)"></path></marker>
			<marker id="transformer-mixing-focus-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" style="fill:var(--viz-focus-stroke)"></path></marker>
		</defs>
		<rect class="viz-plot-bg" x="8" y="30" width="344" height="176" rx="5"></rect>
		<text class="viz-axis-label" x="14" y="18">1 · ATTENTION COMMUNICATES ACROSS TOKEN POSITIONS</text>
		<text class="viz-label" x="22" y="49">unmasked encoder rows</text>
		<text class="viz-label" x="262" y="49">updated token 2</text>
		<g aria-label="Three input token rows, each with four feature cells">
			<rect class="viz-node viz-node--input" x="22" y="60" width="118" height="30" rx="3"></rect>
			<path d="M51.5 60V90M81 60V90M110.5 60V90" class="viz-gridline"></path>
			<rect class="viz-node viz-node--focus" x="22" y="106" width="118" height="30" rx="3"></rect>
			<path d="M51.5 106V136M81 106V136M110.5 106V136" class="viz-gridline"></path>
			<rect class="viz-node viz-node--input" x="22" y="152" width="118" height="30" rx="3"></rect>
			<path d="M51.5 152V182M81 152V182M110.5 152V182" class="viz-gridline"></path>
			<text class="viz-callout" x="10" y="79">t₁</text>
			<text class="viz-callout" x="10" y="125">t₂</text>
			<text class="viz-callout" x="10" y="171">t₃</text>
		</g>
		<path d="M140 75C190 75 199 108 247 118M140 121H247M140 167C190 167 199 134 247 124" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;marker-end:url(#transformer-mixing-focus-arrow)"></path>
		<rect class="viz-node viz-node--output" x="249" y="106" width="90" height="30" rx="3"></rect>
		<path d="M271.5 106V136M294 106V136M316.5 106V136" class="viz-gridline"></path>
		<text class="viz-callout" x="294" y="157" text-anchor="middle">mask permits all rows</text>
		<text class="viz-label" x="180" y="194" text-anchor="middle">shape stays T × d; attention returns an update for every token row</text>
		<rect class="viz-plot-bg" x="8" y="228" width="344" height="140" rx="5"></rect>
		<text class="viz-axis-label" x="14" y="218">2 · THE FFN TRANSFORMS FEATURES WITHIN EACH TOKEN</text>
		<text class="viz-label" x="20" y="247">one row at a time</text>
		<text class="viz-label" x="253" y="247">same T × d shape</text>
		<g aria-label="Three isolated token paths through the same feed-forward network">
			<rect class="viz-node viz-node--input" x="20" y="258" width="76" height="24" rx="3"></rect>
			<rect class="viz-node viz-node--input" x="20" y="295" width="76" height="24" rx="3"></rect>
			<rect class="viz-node viz-node--input" x="20" y="332" width="76" height="24" rx="3"></rect>
			<path d="M96 270H128M96 307H128M96 344H128" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;marker-end:url(#transformer-mixing-arrow)"></path>
			<rect class="viz-node viz-node--focus" x="130" y="256" width="92" height="28" rx="3"></rect>
			<rect class="viz-node viz-node--focus" x="130" y="293" width="92" height="28" rx="3"></rect>
			<rect class="viz-node viz-node--focus" x="130" y="330" width="92" height="28" rx="3"></rect>
			<text class="viz-callout" x="176" y="274" text-anchor="middle">same FFN</text>
			<text class="viz-callout" x="176" y="311" text-anchor="middle">same FFN</text>
			<text class="viz-callout" x="176" y="348" text-anchor="middle">same FFN</text>
			<path d="M222 270H250M222 307H250M222 344H250" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;marker-end:url(#transformer-mixing-arrow)"></path>
			<rect class="viz-node viz-node--output" x="252" y="258" width="76" height="24" rx="3"></rect>
			<rect class="viz-node viz-node--output" x="252" y="295" width="76" height="24" rx="3"></rect>
			<rect class="viz-node viz-node--output" x="252" y="332" width="76" height="24" rx="3"></rect>
			<text class="viz-callout" x="10" y="274">t₁</text>
			<text class="viz-callout" x="10" y="311">t₂</text>
			<text class="viz-callout" x="10" y="348">t₃</text>
		</g>
		<text class="viz-label" x="180" y="384" text-anchor="middle">no row-to-row path: each position is processed independently</text>
		<text class="viz-callout" x="180" y="405" text-anchor="middle">x′ = x + Attention(LN(x))  →  x″ = x′ + FFN(LN(x′))</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> hold the <code>T × d</code> tensor shape fixed. In this unmasked encoder example, attention lets each token row gather context from every row; a causal decoder would omit future-token links. The same FFN then transforms features inside each row independently. Each sublayer contributes an update through its own residual addition. The primary communication roles, not every internal projection, are shown. Original schematic checked against <a href="https://arxiv.org/abs/1706.03762">Vaswani et al. (2017)</a> and the <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html">PyTorch encoder-layer documentation</a>.</figcaption>
</figure>

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
