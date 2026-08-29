---
title: "Gradient clipping"
description: "Cap the norm of the gradient before each optimizer step. The simplest and most reliable defense against training instability."
date: "2025-08-17"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Gradient clipping rescales the gradient vector before the optimizer step so that its global norm does not exceed a fixed threshold $c$. If $\|g\| > c$, replace $g$ with $g \cdot c / \|g\|$; otherwise leave it unchanged.

Training instabilities. Loss spikes, NaN gradients, exploding updates. Are usually caused by a single batch with anomalous gradients. Without clipping, that one bad step can drive parameters into a region from which training never recovers. Clipping bounds the worst-case update and turns a divergence into a recoverable hiccup.

Standard in: transformer pretraining (always), most RL training (always), RNN training (originally proposed for RNNs by [Pascanu et al., 2013](https://arxiv.org/abs/1211.5063), where exploding gradients are intrinsic).

## Two flavors

### Global-norm clipping (the standard)

Compute the L2 norm of the concatenated gradient vector across **all** parameters:

$$
\|g\|_2 = \sqrt{\sum_p \|g_p\|_2^2}
$$

If $\|g\|_2 > c$, scale every parameter's gradient by $c / \|g\|_2$. This preserves the *direction* of the gradient (just shrinks magnitude). $c = 1.0$ is the dominant default for transformer training.

### Per-parameter clipping

Clip each parameter's gradient norm independently. Simpler but distorts the gradient direction; rarely used.

### Value clipping

Clip individual elements of $g$ to a range $[-c, +c]$. Distorts direction even more; mostly historical.

## How to pick the threshold

- **Transformers**: $c = 1.0$ is the universal default. Llama, Mistral, Qwen, GPT all use 1.0.
- **RNNs / LSTMs**: $c$ between 0.25 and 5; needs tuning.
- **RL**: depends on reward scale and policy parametrization; often $c = 0.5$.
- **Diagnostic**: log $\|g\|_2$ over training. If it almost never exceeds $c$, the clip is inactive (try lower); if it always does, the clip is destroying signal (try higher).

## Combined with mixed precision

In FP16/BF16 mixed precision, the loss is scaled before the backward pass to keep small gradients representable. Clipping must be applied **on the unscaled gradients** (after the scaler unscales them). PyTorch's `GradScaler` and similar tooling enforce this ordering.

## Common pitfalls

- **Clipping per parameter group instead of globally.** Gives a different effective clip for each parameter; rarely intended.
- **Forgetting to unscale before clipping under AMP.** The clip threshold is meaningless if applied to scaled gradients.
- **Setting clipping too aggressive.** $c = 0.01$ for a transformer cripples training; you'll see flat loss curves with the clip always active.
- **Treating clipping as a fix for a buggy data pipeline.** A consistent stream of large gradients usually indicates a data or initialization problem, not a clipping problem.
