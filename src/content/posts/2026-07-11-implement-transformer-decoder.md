---
title: "Implement a pre-norm Transformer decoder block"
description: "The code tests tensor contracts, causal masking, stable attention, residual structure, and whether you can connect a toy block to production kernels."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> Implement a pre-norm Transformer decoder block from projections and tensor operations. Do not call a built-in attention module.

Get the contract right before writing code:

- input and output: `[batch, time, d_model]`;
- `d_model` divisible by `num_heads`;
- each position may attend only to itself and earlier positions;
- softmax runs stably;
- residual paths preserve shape;
- gradients reach input and parameters.

## The minimal structure

A pre-norm block is:

$$
\begin{aligned}
h' &= h + \operatorname{Attention}(\operatorname{LN}(h)), \\
y &= h' + \operatorname{MLP}(\operatorname{LN}(h')).
\end{aligned}
$$

**Learning objective:** trace the unchanged residual stream through a pre-norm decoder block while normalized branches compute and add attention and MLP updates.

<!-- visual:pre-norm-decoder-residual-stream -->
<figure class="learning-figure plot-panel" aria-labelledby="pre-norm-decoder-title">
	<p class="visual-kicker">Pre-norm block topology</p>
	<p class="visual-title" id="pre-norm-decoder-title">Normalize the update branches, not the identity stream.</p>
	<svg viewBox="0 0 360 430" role="img" aria-labelledby="pre-norm-decoder-svg-title pre-norm-decoder-svg-desc">
		<title id="pre-norm-decoder-svg-title">Residual stream through a pre-norm Transformer decoder block</title>
		<desc id="pre-norm-decoder-svg-desc">A vertical solid identity rail carries input h with shape batch by time by model dimension through two additions. At stage one, a dashed branch copies h through LayerNorm and causal self-attention, then rejoins the rail at a plus node to produce h prime. At stage two, another dashed branch copies h prime through LayerNorm and the position-wise MLP, then rejoins at a second plus node to produce output y. LayerNorm appears only on the branches, so the raw residual states bypass both normalization and sublayers.</desc>
		<defs>
			<marker id="decoder-solid-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="var(--c-text-soft)"></path></marker>
			<marker id="decoder-branch-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="var(--viz-input-stroke)"></path></marker>
		</defs>
		<text class="viz-axis-label" x="18" y="20">RESIDUAL STREAM · [B, T, C] THROUGHOUT</text>
		<rect class="viz-plot-bg" x="16" y="32" width="328" height="348" rx="4"></rect>
		<text class="viz-node-label" x="55" y="57" text-anchor="middle">h</text>
		<text class="viz-node-value" x="55" y="73" text-anchor="middle">[B,T,C]</text>
		<path d="M55 79V163" style="fill:none;stroke:var(--c-text-soft);stroke-width:3" marker-end="url(#decoder-solid-arrow)"></path>
		<circle cx="55" cy="174" r="13" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></circle>
		<text class="viz-node-label" x="55" y="179" text-anchor="middle">+</text>
		<path d="M55 187V278" style="fill:none;stroke:var(--c-text-soft);stroke-width:3" marker-end="url(#decoder-solid-arrow)"></path>
		<circle cx="55" cy="289" r="13" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></circle>
		<text class="viz-node-label" x="55" y="294" text-anchor="middle">+</text>
		<path d="M55 302V345" style="fill:none;stroke:var(--c-text-soft);stroke-width:3" marker-end="url(#decoder-solid-arrow)"></path>
		<text class="viz-node-label" x="55" y="367" text-anchor="middle">y</text>
		<text class="viz-node-value" x="55" y="383" text-anchor="middle">[B,T,C]</text>
		<path d="M55 88H110" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-dasharray:6 4" marker-end="url(#decoder-branch-arrow)"></path>
		<rect class="viz-node viz-node--input" x="116" y="68" width="82" height="40" rx="4"></rect>
		<text class="viz-node-label" x="157" y="85" text-anchor="middle">LayerNorm 1</text>
		<text class="viz-node-value" x="157" y="100" text-anchor="middle">LN(h)</text>
		<path d="M198 88H214" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-dasharray:6 4" marker-end="url(#decoder-branch-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="220" y="58" width="108" height="60" rx="4"></rect>
		<text class="viz-node-label" x="274" y="79" text-anchor="middle">Causal self-attn</text>
		<text class="viz-node-value" x="274" y="96" text-anchor="middle">normalized Q, K, V</text>
		<text class="viz-node-value" x="274" y="111" text-anchor="middle">future keys blocked</text>
		<path d="M274 118V158H78V174H70" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-dasharray:6 4" marker-end="url(#decoder-branch-arrow)"></path>
		<text class="viz-axis-label" x="82" y="153">UPDATE 1</text>
		<text class="viz-callout" x="82" y="190">h′ = h + Attention(LN(h))</text>
		<text class="viz-node-label" x="32" y="224" text-anchor="middle">h′</text>
		<text class="viz-node-value" x="32" y="240" text-anchor="middle">[B,T,C]</text>
		<path d="M55 218H110" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-dasharray:6 4" marker-end="url(#decoder-branch-arrow)"></path>
		<rect class="viz-node viz-node--input" x="116" y="198" width="82" height="40" rx="4"></rect>
		<text class="viz-node-label" x="157" y="215" text-anchor="middle">LayerNorm 2</text>
		<text class="viz-node-value" x="157" y="230" text-anchor="middle">LN(h′)</text>
		<path d="M198 218H214" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-dasharray:6 4" marker-end="url(#decoder-branch-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="220" y="198" width="108" height="40" rx="4"></rect>
		<text class="viz-node-label" x="274" y="215" text-anchor="middle">MLP</text>
		<text class="viz-node-value" x="274" y="230" text-anchor="middle">feature update</text>
		<path d="M274 238V273H78V289H70" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-dasharray:6 4" marker-end="url(#decoder-branch-arrow)"></path>
		<text class="viz-axis-label" x="82" y="268">UPDATE 2</text>
		<text class="viz-callout" x="82" y="318">y = h′ + MLP(LN(h′))</text>
		<path class="viz-gridline" d="M18 392H342"></path>
		<text class="viz-label" x="92" y="411" text-anchor="middle">Solid rail: unchanged identity path</text>
		<text class="viz-label" x="270" y="411" text-anchor="middle">Dashed: normalized learned update</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> follow the solid rail from <var>h</var> to <var>y</var>: neither LayerNorm sits on that identity path. At each numbered update, copy the current residual state into the dashed branch, normalize the copy, run one sublayer, and add the result back. The second branch must start from <var>h</var>′, not the original <var>h</var>. Structure checked against <a href="https://arxiv.org/abs/1706.03762"><cite>Attention Is All You Need</cite></a>, <a href="https://proceedings.mlr.press/v119/xiong20b.html">Xiong et al.'s Pre-LN analysis</a>, and the <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html">PyTorch decoder-layer contract</a>; the graphic is original.</figcaption>
</figure>

Inside attention:

1. project $h$ to $Q$, $K$, and $V$;
2. reshape `[B, T, 3C]` into heads;
3. compute $QK^T / \sqrt{d_h}$;
4. mask positions where key index is greater than query index;
5. apply softmax along the key dimension, preferably in FP32;
6. multiply by $V$;
7. reassemble heads and apply the output projection.

The causal-mask test is stronger than checking a triangular tensor. Change future input tokens and verify earlier outputs remain unchanged.

## Reference sketch

```python
qkv = self.qkv(hidden)
q, k, v = qkv.chunk(3, dim=-1)
q = q.view(B, T, H, D).transpose(1, 2)
k = k.view(B, T, H, D).transpose(1, 2)
v = v.view(B, T, H, D).transpose(1, 2)

scores = q @ k.transpose(-1, -2) / math.sqrt(D)
# True means block this query-key pair before softmax.
blocked = torch.triu(torch.ones(T, T, dtype=torch.bool, device=hidden.device), diagonal=1)
scores = scores.masked_fill(blocked, float("-inf"))
weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
context = weights @ v
context = context.transpose(1, 2).contiguous().view(B, T, C)
return self.output(context)
```

The sketch is not the entire interview. Tests and explanation determine level.

## What an L4 answer sounds like

The candidate produces the correct formula but loses track of shapes, applies softmax over the query axis, or uses a mask whose boolean convention is inverted. They validate only output shape.

## What an L5 answer adds

An L5 candidate writes shape comments, uses a causal-invariance test, checks gradients, and explains scaling. They know why `contiguous()` may be needed after transpose and why raw `view()` on a non-contiguous tensor can fail or misrepresent layout.

They test:

- one token;
- multiple heads;
- future-token invariance;
- finite output under large logits;
- backward propagation;
- invalid head dimensions.

## What an L6 answer adds

An L6 candidate connects the block to the real stack without derailing implementation. They explain:

- fused QKV projection;
- FlashAttention avoiding materialized $T \times T$ scores;
- rotary position encoding entering $Q$ and $K$;
- GQA or MQA reducing KV-cache size;
- KV caching changing inference from full self-attention to one-query incremental attention;
- tensor and sequence parallelism changing projection and activation ownership;
- dropout and deterministic behavior in training versus evaluation.

They distinguish algorithmic equivalence from kernel behavior. A mathematically correct implementation can still be unusable because it materializes attention or launches many tiny kernels.

## Tells that get you a strong-hire vote

- Shapes are explicit at every reshape and transpose.
- The mask convention is proved with future-token invariance.
- Scaling and softmax axis are correct.
- Softmax stability and low precision are discussed.
- Residual and normalization order matches the requested block.
- Tests include gradients and causality, not only shape.
- Production differences are concise and technically correct.

## Tells that get you down-leveled

- Copying a remembered snippet without shape reasoning.
- Building a lower-triangular mask but not knowing whether `True` means keep or block.
- Softmax over the wrong dimension.
- Ignoring non-contiguous layout after transpose.
- Claiming the toy implementation is FlashAttention-ready.
- Explaining every Transformer variant before producing working code.

## Common follow-up

"Why pre-norm instead of post-norm?"

Pre-norm gives the residual stream a cleaner identity path, which generally improves gradient flow and stability in deep Transformers. Post-norm can work and was used in the original Transformer, but deep modern stacks usually need more care with initialization and schedule. The choice changes block order, not the attention mechanism itself.

Use the [implementation starter](/prep/labs/implementation/) before copying the reference sketch.

*Related: [implement attention from scratch](/questions/implement-attention-from-scratch/), [Transformer architecture](/concepts/transformer-architecture/), and [FlashAttention](/concepts/flashattention/).*
