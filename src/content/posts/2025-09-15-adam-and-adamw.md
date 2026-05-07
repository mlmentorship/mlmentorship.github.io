---
title: "Adam, AdamW, and the modern optimizer landscape"
description: "Why Adam works, why AdamW is the version you actually want, and what's changed in the optimizer landscape since 2018."
date: "2025-09-15"
draft: false
tags: ["reference"]
category: "reference"
---


## One-line definition

**Adam** is an adaptive optimizer that combines momentum (running mean of gradients) with per-parameter learning rate scaling (running mean of squared gradients), giving robust default behavior across many problems. **AdamW** is a small but important fix that decouples weight decay from the gradient update, and is what most modern training pipelines actually use.

## Why it matters

Adam is the default optimizer since 2015 across transformers, LLMs, and modern deep learning. Understanding why it works and where it breaks is interview-canonical.

## The mechanism

For each parameter, Adam maintains two running averages:

- `m_t` = running mean of gradients (1st moment, like momentum)
- `v_t` = running mean of squared gradients (2nd moment, captures gradient magnitude)

The update is:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2

# bias correction (for early steps when m, v are biased toward 0)
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)

theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
```

Default `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`.

The intuition: divide the momentum-smoothed gradient by its running magnitude. Parameters with consistently large gradients get smaller effective updates; parameters with consistently small gradients get larger. This makes Adam robust to wildly different gradient scales across parameters, one of its biggest practical advantages.

## What an interviewer expects you to say

If asked "explain Adam":

1. Two running averages: 1st moment (momentum) and 2nd moment (squared gradient).
2. The per-parameter scaling by `1/sqrt(v_hat)` is the adaptive part.
3. Bias correction matters for early steps (without it, m and v are biased toward zero for the first ~1/(1-beta) steps).
4. Default hyperparameters (`beta1=0.9, beta2=0.999, eps=1e-8`) work surprisingly well across many problems.
5. AdamW is the version with decoupled weight decay; it's what you actually want.

## AdamW: the fix that matters

Standard Adam adds weight decay to gradients, so it gets scaled by `1/sqrt(v_hat)`. Parameters with large gradients receive *less* weight decay, backwards. AdamW decouples weight decay from the adaptive scaling.

AdamW fixes this by applying weight decay *directly to the parameters*, not to the gradient:

```
theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps) - lr * weight_decay * theta_{t-1}
```

Decoupled weight decay is independent of adaptive scaling. Though minor-sounding, it's a meaningful improvement for transformers [(Loshchilov & Hutter 2019)](https://arxiv.org/abs/1711.05101). Standard in modern transformer training.

If your code uses `optim.Adam` with `weight_decay > 0`, you probably mean to use `optim.AdamW`.

## When Adam goes wrong

Adam is robust but not bulletproof:

- **Without warmup, large transformers diverge.** The 2nd moment estimate `v_t` is unreliable in the first ~1000 steps. Without warmup, you get huge effective steps that destabilize the model. Standard recipe: linear warmup over the first 5% of steps.
- **Adam is biased toward sharp minima**. SGD with momentum (and small enough LR) is sometimes argued to find flatter minima that generalize better. The empirical picture is mixed; for transformers Adam is essentially always better, for some CNN architectures SGD is competitive.
- **Memory overhead is 3x.** Stores m and v alongside parameters. For 70B in BF16, optimizer state is 280 GB. ZeRO/FSDP shard across ranks.
- **Adam's adaptive scaling can dramatically inflate updates on noisy gradients**. For RL or other settings with high gradient variance, Adam can be unstable.

## The modern optimizer landscape

Since Adam (2014), several alternatives have emerged. Most have not displaced AdamW for general use, but each occupies a niche:

- **AdamW** (2017): default for transformers, LLMs, most deep learning.
- **LAMB** (2019): a variant of AdamW that scales better to very large batch sizes (~32K+); used in some BERT-large training runs.
- **AdaFactor** (2018): Adam with a factored second-moment matrix, uses way less memory. Used in T5 training; quality slightly worse than full Adam but memory savings are huge.
- **Lion** (2023): a momentum-only optimizer with sign-based updates. Uses less memory than Adam (no second moment), trains slightly faster on large vision and language models. Has shown promise but has not fully displaced AdamW.
- **Sophia** (2023): a second-order method (Hessian-aware) for LLM pretraining. Reports up to 2x speedup over AdamW on some benchmarks; less battle-tested.
- **Shampoo** (2018, refined 2023): full second-order optimizer using preconditioners. Can outperform AdamW but is much more memory-intensive; mainly used in research.
- **Muon** (2024): a recent optimizer using orthogonal momentum updates; gaining traction in LLM training in 2025-2026.

**AdamW remains default for production in 2026.** Gains from alternatives are small relative to engineering cost.

## Common confusions

- **"Adam is the same as RMSProp + momentum."** Approximately. RMSProp tracks the second moment but not the first; Adam tracks both. The bias correction is also Adam-specific.
- **"Adam doesn't need a learning rate."** It needs one, just less aggressively tuned than SGD. Default `1e-3` is a reasonable starting point but you should still do an LR sweep.
- **"AdamW is just Adam with weight decay."** No, it's Adam with *decoupled* weight decay. Adam with `weight_decay > 0` is *not* AdamW.
- **"Use Adam for everything."** Mostly correct in 2026, but with caveats: very large batch (consider LAMB), memory-constrained (consider AdaFactor or Lion), unusual gradient distributions (consider SGD or careful Adam tuning).

## Why interviewers ask

The Adam question tests whether you:
1. Understand adaptive vs non-adaptive optimization.
2. Know the warmup-and-bias-correction subtleties.
3. Have used the version (AdamW) that actually works for transformers.
4. Have kept up with the optimizer landscape post-2020.

Easy to ace; easy to fumble.

---

*Related interview: [How do you choose a learning rate?](/interviews/how-to-choose-learning-rate/), [How would you debug a model that's not learning?](/interviews/debug-model-not-learning/).*
