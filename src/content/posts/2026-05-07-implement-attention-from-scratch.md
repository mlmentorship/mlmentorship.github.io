---
title: "Implement attention from scratch"
description: "The coding question that doubles as a depth check. The code is short; the conversation around it tells the level."
date: "2026-05-07"
draft: false
tags: ["interviews"]
category: "interviews"
---


> *Asked in: coding rounds at LLM-team interviews.*

The code is short and most candidates get it right. The signal is in the conversation around it: the 1/sqrt(d) scaling, the masking convention, FlashAttention, KV cache, FP32 softmax in mixed precision.

## The minimum correct answer

```python
import torch
import torch.nn.functional as F

def attention(Q, K, V, mask=None):
    """
    Q: (B, H, T, d)  -- queries
    K: (B, H, T, d)  -- keys
    V: (B, H, T, d)  -- values
    mask: (B, 1, T, T) or broadcastable -- True positions to keep
    Returns: (B, H, T, d) -- attended values
    """
    d = Q.size(-1)
    scores = (Q @ K.transpose(-1, -2)) / d ** 0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return attn @ V
```

If this is all you write and you stop talking, you're at L4. The interviewer now wants to see you bring up the things they were planning to probe.

## What the L5 candidate adds, unprompted

> "A few things to note about this implementation:
>
> **The 1/sqrt(d) scaling.** Without it, the dot products grow as d gets larger, pushing the softmax into saturation regions where gradients vanish. The square root is what keeps the variance of the dot product roughly constant.
>
> **The mask.** I implemented it as boolean keep-positions, masked-fill with `-inf`. After softmax this gives exactly zero attention to masked positions. Two common masks: causal (lower-triangular for autoregressive) and padding (true wherever a real token is, false on padding).
>
> **The matmul layout.** I'm assuming (batch, heads, time, dim) so that the head dimension is broadcasted naturally in the @ operation. The alternative (batch, time, heads&times;dim) is more memory-friendly for some operations but needs reshaping before attention.
>
> **For multi-head, this is the per-head computation.** A real implementation projects Q, K, V from the input via three linear layers, splits into heads, runs this attention, and concatenates."

This is L5. You've named the things in the code and explained them.

## What the L6 candidate adds

> "...and a few more things I'd want to discuss before considering this done:
>
> **Numerical stability of softmax.** Built into `F.softmax` (which subtracts the max before exponentiating), but easy to get wrong if you implement softmax by hand. With float16 / bfloat16, this matters because exp can overflow.
>
> **Precision.** In production this would run in BF16 or FP16 on GPU. The matmuls are fine in low precision, but the softmax is often kept in FP32 for stability, the standard recipe is to cast scores to FP32 before softmax and back to the lower precision after.
>
> **The mask in autoregressive models.** A causal mask of size (T, T) is shared across all heads and batch elements; you should construct it once and broadcast, not allocate per-batch. Some frameworks (like nn.MultiheadAttention) accept this as a separate `is_causal=True` flag and avoid the explicit mask tensor entirely.
>
> **Memory.** This implementation materializes the full T&times;T attention matrix in HBM. For long sequences (T &gt; ~2K), that's the dominant memory and latency cost. In production we'd use FlashAttention, which computes the same output without materializing the matrix, tiles Q, K, V into SRAM and uses streaming softmax. The signature looks the same; the kernel is different.
>
> **Inference vs training.** During autoregressive inference, we don't recompute attention over the full prefix at every step, we cache K and V from previous steps (the KV-cache) and only compute new K, V for the new token. This makes per-step attention O(T) instead of O(T^2), at the cost of memory proportional to T."

This is L6. You've connected the toy code to the production reality, named the systems concerns, and shown you understand what changes when you actually deploy this.

## A common follow-up: implement multi-head attention

If they push you to multi-head:

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        out = attention(Q, K, V, mask)               # (B, H, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, -1)  # (B, T, d_model)
        return self.W_o(out)
```

Things to note out loud:

- Three projection matrices for Q, K, V (often combined into one for efficiency).
- Reshape into heads, transpose so head is dim 1.
- After attention, transpose back and reshape to (B, T, d_model).
- Output projection W_o mixes the heads.

## The tells that get you a strong-hire vote

- You **discuss the 1/sqrt(d) scaling** without being asked.
- You **mention the mask** (causal vs padding) and use boolean + `-inf` correctly.
- You **bring up FlashAttention** as the production kernel.
- You **mention KV-cache** for inference.
- You **use FP32 softmax in mixed precision** as the right pattern.

## The tells that get you down-leveled

- Forget the 1/sqrt(d) scaling (very common; very telling).
- Use Python loops over the batch or head dimension, signals you don't think in tensor ops.
- Add `+ mask` instead of `masked_fill(-inf)` when mask is boolean (off-by-broadcasting; doesn't actually mask).
- Don't know what KV-cache is.
- Reach for `nn.MultiheadAttention` directly when asked to implement, the question is testing whether you can.

## Common bugs in this code

In order of frequency:

1. **Forgetting the scaling.** Easy fix; immediate down-level if not corrected.
2. **Wrong axis on softmax.** Should be `dim=-1` (over keys); easy to mistakenly do `dim=-2`.
3. **Mask broadcasting.** A (B, T) padding mask needs to become (B, 1, 1, T) to broadcast correctly across heads and queries.
4. **Using `0` instead of `-inf` for masking.** Multiplying by 0 doesn't prevent attention, the softmax will still allocate weight to those positions.
5. **Using `bool` directly in `masked_fill` without checking polarity.** `masked_fill(mask, ...)` fills *where mask is True*. So if your mask is "True = keep", you need `~mask`.

## Why interviewers ask this

The question is mechanically simple but probes:
1. Your tensor-ops fluency (do you think in shapes?).
2. Your understanding of the *why* (scaling, mask, softmax).
3. Your awareness of production reality (FlashAttention, KV-cache, precision).

A candidate who writes correct attention in 5 minutes and *then* spends 10 minutes discussing the production concerns has signaled L6 in 15 minutes. A candidate who writes the same code and waits silently for the next question has signaled L4.

The code is the entry ticket. The conversation around it is the interview.

---

*Related reference: [FlashAttention](/reference/flashattention/), [BatchNorm vs LayerNorm](/reference/batchnorm-vs-layernorm/).*
