---
title: "Learning rate schedules: warmup and cosine decay"
description: "Why almost every modern training run linearly warms up the LR over a few hundred steps and then decays it on a cosine to near zero."
date: "2026-01-05"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

A learning-rate schedule is a function $\eta(t)$ that varies the optimizer's step size over training. The dominant 2026 default for LLMs is **linear warmup** for a few hundred to a few thousand steps, followed by **cosine decay** down to ~10% of the peak LR.

## Why warmup

Adam-family optimizers track running variance estimates ($v_t$ in Adam). At step 1, those estimates are noisy. The bias correction divides by a small denominator, and updates can be huge in magnitude. With a small LR for the first $W$ steps, the optimizer accumulates reliable second-moment statistics before the LR ramps up. Without warmup, transformer training routinely diverges in the first few hundred steps.

Typical warmup: $W = 1000$ to $4000$ steps for pretraining; ~100 steps is enough for fine-tuning.

## Why cosine

After warmup, hold near peak LR for most of training to make progress; near the end, decay so the optimizer settles into a flatter minimum. Cosine decay,

$$
\eta(t) = \eta_{\min} + \tfrac{1}{2} (\eta_{\max} - \eta_{\min}) \left(1 + \cos\!\left(\pi \cdot \tfrac{t - W}{T - W}\right)\right),
$$

spends most of its budget near the peak and slows late, which empirically beats linear or step decay across most workloads [(Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1608.03983).

Common practice: cosine to $\eta_{\min} = 0.1 \cdot \eta_{\max}$ over the full training horizon $T$.

## How to set the peak LR

For Adam/AdamW on transformer training, the default starting point is $\eta_{\max} = 3 \times 10^{-4}$ (Karpathy's "magic constant") for moderate batch sizes. Larger batches scale up roughly linearly until the LR-batch tradeoff breaks ($\sim 10^{-3}$ for very large batch).

For SFT or task-specific fine-tuning of a pretrained model: 10–100× lower than pretraining ($1 \times 10^{-5}$ to $5 \times 10^{-5}$).

A **LR range test** [(Smith, 2017)](https://arxiv.org/abs/1506.01186): sweep $\eta$ from $10^{-7}$ to $10^{-1}$ over a few hundred steps, plot loss vs. LR. Pick the point at the steepest descent (typically ~10× below where the loss diverges).

## Other schedules

- **Constant**: useful for online learning / RL where the data distribution shifts.
- **Inverse square root** (original transformer paper): $\eta(t) \propto 1/\sqrt{t}$ after warmup. Largely superseded by cosine.
- **One-cycle** [(Smith, 2018)](https://arxiv.org/abs/1803.09820): warmup to a high peak, then decay aggressively. Used in some vision training; uncommon in LLMs.
- **WSD** (Warmup-Stable-Decay): warmup, hold constant for most of training, decay sharply at the end. Used in some recent LLM training [(Hu et al., 2024)](https://arxiv.org/abs/2404.06395) for easier checkpoint resumption.

## Common pitfalls

- **Skipping warmup.** Adam without warmup on a transformer routinely diverges.
- **Decaying too fast.** Aggressive decay limits how far the model can move; cosine-to-10% is a reliable default.
- **Forgetting LR scales with batch.** Doubling the batch usually requires roughly doubling the LR.
- **Resuming a cosine schedule from a checkpoint.** If the cosine is parameterized over total steps, resuming with a different total breaks the schedule. Save schedule state with the checkpoint.
