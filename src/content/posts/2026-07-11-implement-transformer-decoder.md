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
