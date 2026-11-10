---
title: "Implement attention from scratch"
description: "The coding question that doubles as a depth check. The code is short; the conversation around it tells the level."
date: "2025-11-27"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: coding rounds at LLM-team interviews.*

Implement scaled dot-product attention with explicit tensor shapes and mask semantics, then explain its numerical and serving implications.

## Problem

**Context.** Floating tensors `Q`, `K`, and `V` have shape `(B, H, T, d)`, sharing device and compatible dtype.
An optional Boolean mask broadcasts to `(B, H, T, T)`; `True` means a key is allowed.

**Task.** Return attended values shaped `(B, H, T, d)` using scaled query-key scores, key-axis normalization, and weighted values.
Explain a causal mask and verify that disallowed keys contribute no weight.

**Constraints.** Use tensor operations rather than a built-in attention module, and preserve input tensors.
Assume positive feature size, representable arithmetic, and at least one allowed key per query.
Fully masked rows require a separate policy because the reference softmax would produce invalid values.

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

<!-- visual:attention-causal-mask-row -->
<figure class="learning-figure" aria-labelledby="attention-causal-mask-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="attention-causal-mask-title">Scale scores, block disallowed keys before normalization, then sum their weighted values</p><div class="visual-argument" data-visual-argument="attention-causal-mask-row" data-argument-kind="matrix" data-argument-composition="causal-frontier"><svg viewBox="0 0 360 2220" role="img" aria-label="Scale scores, block disallowed keys before normalization, then sum their weighted values. Each query scores keys and normalizes only allowed positions before combining their values. Boolean true means allowed. Causal future scores become negative infinity, giving zero weight even if future values grow. Zeroing logits instead leaves positive softmax weight. Normalize across keys, not queries. Tensor axes and output width stay intact. Fully masked rows need an explicit policy; stable softmax and suitable precision remain necessary."><title>Scale scores, block disallowed keys before normalization, then sum their weighted values</title><desc>Each query scores keys and normalizes only allowed positions before combining their values. Boolean true means allowed. Causal future scores become negative infinity, giving zero weight even if future values grow. Zeroing logits instead leaves positive softmax weight. Normalize across keys, not queries. Tensor axes and output width stay intact. Fully masked rows need an explicit policy; stable softmax and suitable precision remain necessary.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Future values cannot cross the time</tspan><tspan x="20" dy="18">barrier</tspan></text><text x="20" y="86" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="20" dy="0">Toy query at position two; feature</tspan><tspan x="20" dy="18">width four</tspan></text><path d="M226 127L226 400" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="40" cy="176" r="17" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="40" y="182" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="40" dy="0">0</tspan></text><path d="M40 198L40 292M36 284L40 292L44 284" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3.0803668780445657;stroke-linejoin:round"></path><text x="40" y="350" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="40" dy="0">0.090</tspan></text><circle cx="109" cy="176" r="17" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="109" y="182" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="109" dy="0">1</tspan></text><path d="M109 198L109 292M105 284L109 292L113 284" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:4.936741652657572;stroke-linejoin:round"></path><text x="109" y="350" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="109" dy="0">0.245</tspan></text><circle cx="178" cy="176" r="17" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="178" y="182" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">2</tspan></text><path d="M178 198L178 292M174 284L178 292L182 284" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:9.982891469297861;stroke-linejoin:round"></path><text x="178" y="350" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="178" dy="0">0.665</tspan></text><circle cx="247" cy="176" r="17" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="247" y="182" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="247" dy="0">3</tspan></text><path d="M247 233L247 275" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M240 273L254 287" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M254 273L240 287" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="247" y="350" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="247" dy="0">0.000</tspan></text><circle cx="316" cy="176" r="17" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="316" y="182" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="316" dy="0">4</tspan></text><path d="M316 233L316 275" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M309 273L323 287" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M323 273L309 287" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="316" y="350" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="316" dy="0">0.000</tspan></text><text x="20" y="441" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Allowed weights sum to one; future</tspan><tspan x="20" dy="18">weights are zero.</tspan></text><text x="20" y="507" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Mask the logits before normalizing</tspan><tspan x="20" dy="18">the visible prefix.</tspan></text><text x="20" y="645" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Query components all one. Dot</tspan><tspan x="20" dy="18">products divided by the square root</tspan><tspan x="20" dy="18">of width give scores zero, one, two,</tspan><tspan x="20" dy="18">eighteen, twenty.</tspan></text><text x="20" y="795" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">True means allowed. Replace future</tspan><tspan x="20" dy="18">scores with negative infinity before</tspan><tspan x="20" dy="18">stable key-axis softmax; subtract the</tspan><tspan x="20" dy="18">allowed maximum before</tspan><tspan x="20" dy="18">exponentiating.</tspan></text><text x="20" y="965" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Toy scalar values two, four, eight,</tspan><tspan x="20" dy="18">one hundred, one thousand. Only the</tspan><tspan x="20" dy="18">first three have contribution routes.</tspan></text><text x="20" y="1150" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Key 0: value 2, contribution 0.180</tspan></text><rect x="20" y="1195" width="7.2024458536304365" height="25" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="20" y="1310" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Key 1: value 4, contribution 0.979</tspan></text><rect x="20" y="1355" width="39.15655536876762" height="25" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="20" y="1470" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Key 2: value 8, contribution 5.322</tspan></text><rect x="20" y="1515" width="212.87710584794297" height="25" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="20" y="1690" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Add contributions: scalar output</tspan><tspan x="20" dy="18">6.481. Vector values use the same</tspan><tspan x="20" dy="18">weights per feature.</tspan></text><text x="20" y="1840" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Raise future values to one hundred</tspan><tspan x="20" dy="18">million and one billion, or change</tspan><tspan x="20" dy="18">future scores: output is unchanged</tspan><tspan x="20" dy="18">through this mask.</tspan></text><text x="20" y="1990" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Zeroing future logits leaves positive</tspan><tspan x="20" dy="18">weight. Fully masked rows need an</tspan><tspan x="20" dy="18">explicit policy; this reference</tspan><tspan x="20" dy="18">assumes at least one allowed key.</tspan><tspan x="20" dy="18">Keep batch/head/time/feature axes and</tspan><tspan x="20" dy="18">suitable arithmetic precision.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-grid--two" role="group" aria-label="Causal attention mask and a worked softmax row">
		<section class="visual-panel" aria-labelledby="attention-mask-grid-title">
			<h4 id="attention-mask-grid-title">Rows are queries; columns are keys</h4>
			<p>A query may keep its own key and earlier keys. Future-key scores become negative infinity.</p>
			<table class="cm-grid" aria-label="Four-token lower-triangular causal mask where keep cells are on and below the diagonal">
				<thead><tr><th scope="col">query</th><th scope="col">k1</th><th scope="col">k2</th><th scope="col">k3</th><th scope="col">k4</th></tr></thead>
				<tbody>
					<tr><th scope="row">q1</th><td class="cm-selected"><strong>keep</strong></td><td>-inf</td><td>-inf</td><td>-inf</td></tr>
					<tr><th scope="row">q2</th><td class="cm-selected"><strong>keep</strong></td><td class="cm-selected"><strong>keep</strong></td><td>-inf</td><td>-inf</td></tr>
					<tr><th scope="row">q3</th><td class="cm-selected"><strong>keep</strong></td><td class="cm-selected"><strong>keep</strong></td><td class="cm-selected"><strong>keep</strong></td><td>-inf</td></tr>
					<tr><th scope="row">q4</th><td class="cm-selected"><strong>keep</strong></td><td class="cm-selected"><strong>keep</strong></td><td class="cm-selected"><strong>keep</strong></td><td class="cm-selected"><strong>keep</strong></td></tr>
				</tbody>
			</table>
			<p class="cm-equation">mask shape: (1, 1, T queries, T keys)</p>
		</section>
		<section class="visual-panel" aria-labelledby="attention-row-trace-title">
			<h4 id="attention-row-trace-title">Trace query q3 across key columns</h4>
			<p>The fourth raw score is largest, but q3 cannot use future key k4.</p>
			<table class="cm-grid" aria-label="Worked query row before masking, after masking, and after softmax over four key columns">
				<thead><tr><th scope="col">stage</th><th scope="col">values across k1, k2, k3, k4</th></tr></thead>
				<tbody>
					<tr><th scope="row">scores</th><td>0.2, 1.1, -0.4, 2.8</td></tr>
					<tr><th scope="row">filled</th><td>0.2, 1.1, -0.4, -inf</td></tr>
					<tr><th scope="row">softmax</th><td class="cm-selected"><strong>0.25, 0.61, 0.14, 0.00</strong></td></tr>
				</tbody>
			</table>
			<p class="cm-equation">dim = -1: weights across keys sum to 1.00</p>
			<p class="cm-equation">output for q3 = 0.25V1 + 0.61V2 + 0.14V3</p>
		</section>
	</div>
	<div class="visual-derivation-caption">choose one query row, replace every disallowed key score with negative infinity, then normalize left to right across key columns. For q3, k4 receives exactly zero weight even though its raw score was largest; the remaining weights sum to one and multiply the matching value rows. This is an original worked example checked against <a href="https://arxiv.org/abs/1706.03762">scaled dot-product attention</a> and <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html">PyTorch's boolean-mask semantics</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="attention-causal-mask-row"><strong>Check:</strong> Why does k4 contribute nothing despite having the largest raw score for q3?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="attention-causal-mask-row">Each query scores keys and normalizes only allowed positions before combining their values. Boolean true means allowed. Causal future scores become negative infinity, giving zero weight even if future values grow. Zeroing logits instead leaves positive softmax weight. Normalize across keys, not queries. Tensor axes and output width stay intact. Fully masked rows need an explicit policy; stable softmax and suitable precision remain necessary. <strong>Can a future key contribute a tiny positive amount through this mask?</strong></span></figcaption>
</figure>

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

*Related: [FlashAttention](/concepts/flashattention/) and [BatchNorm versus LayerNorm](/concepts/batchnorm-vs-layernorm/).*
