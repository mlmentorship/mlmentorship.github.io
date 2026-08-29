---
title: "Cross-entropy and softmax"
description: "The pairing isn't arbitrary. Cross-entropy is the negative log-likelihood under a categorical distribution, and the softmax+CE gradient simplifies to (p − y), which is why it's stable."
date: "2025-09-05"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

**Cross-entropy loss** is the negative log-likelihood of the true class under a probability distribution predicted by the model. **Softmax** is the standard parameterization that turns logits into a categorical distribution. The two are nearly always paired because the math composes cleanly.

Almost every classification model uses softmax + cross-entropy. The reasons are not arbitrary:

1. **Cross-entropy is the right loss for classification under MLE**. If you assume your label is a sample from a categorical distribution and you want maximum likelihood, the loss is exactly cross-entropy.
2. **The gradient simplifies to (p − y)**. The composition of softmax and cross-entropy has the unique property that the gradient of the loss with respect to the logits is `softmax_output − one_hot_label`. Numerically stable, easy to compute.
3. **MSE on classification has vanishing gradients for confident-but-wrong predictions**. Cross-entropy doesn't; the gradient stays large precisely when the model is most wrong.

## The math, briefly

For C classes and a single example with true class y:

**Softmax**: `p_i = exp(z_i) / sum_j exp(z_j)` for logits z = (z_1, ..., z_C).

**Cross-entropy**: `L = -log p_y = -z_y + log(sum_j exp(z_j))`.

Note the second form (log-sum-exp): this is the numerically stable way to compute cross-entropy, you never explicitly form the softmax, you compute LSE on the logits directly. PyTorch's `nn.CrossEntropyLoss` does this.

**Gradient w.r.t. logits**: `dL/dz_i = p_i − y_i` where y is one-hot. Three lines of algebra; the cleanest gradient in deep learning.

## What an interviewer expects you to say

If asked "why softmax + cross-entropy":

1. State the MLE interpretation.
2. State the gradient simplification.
3. Mention the numerical stability of computing them jointly (log-sum-exp trick).
4. Mention the contrast with MSE on classification (vanishing gradients).

Bonus depth: temperature `tau` (dividing logits by `tau` before softmax) controls the sharpness of the distribution, high `tau` makes it more uniform, low `tau` makes it more peaked. Used in distillation (high `tau` to extract more information from the teacher) and in sampling from LLMs (low `tau` for greedy-like behavior).

## Common confusions

- **"Softmax" vs "softmax + cross-entropy" as separate operations.** Conceptually distinct, but in practice always computed jointly because the joint gradient is so much cleaner.
- **Computing cross-entropy on probabilities you already softmaxed.** PyTorch's `nn.CrossEntropyLoss` takes *logits*, not probabilities. Passing `softmax(logits)` will give wrong gradients and (often silently) bad training.
- **"Cross-entropy" vs "categorical cross-entropy" vs "binary cross-entropy".** All the same idea; "binary" is just C=2 (often parameterized with sigmoid instead of softmax).
- **MSE for classification "as a baseline".** Don't. Vanishing gradients for confident-wrong predictions; train slowly.

## Numerical stability: the LSE trick

The naive `log(sum_j exp(z_j))` overflows for large z_j. The fix:

```
log_sum_exp(z) = max(z) + log(sum_j exp(z_j - max(z)))
```

Subtract the max before exponentiating; add back outside the log. The maximum exponential argument is now 0; no overflow.

Every framework's `cross_entropy_loss` does this internally. If you write your own, you must too.

## Why MSE fails on classification

MSE gradient for a single output is proportional to `(p - y) * p * (1-p)` (when paired with sigmoid). When the model is *very* confident and wrong (`p ≈ 1` for the wrong class), the `p*(1-p)` term vanishes, the model can't learn its way out of the bad prediction. Cross-entropy's gradient is `(p - y)` directly, which stays large.

This is why no one uses MSE for classification despite the textbook deriving it for completeness.

---

*Related: [BatchNorm vs LayerNorm](/concepts/batchnorm-vs-layernorm/). Related interview: [Why does dropout work?](/questions/why-does-dropout-work/).*
