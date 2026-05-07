---
title: "How do you choose a learning rate?"
description: "The right answer is a procedure, not a number. The wrong answers are 'use the default' and 'try a few values.'"
date: "2026-05-07"
draft: false
tags: ["interviews"]
category: "interviews"
---


> *Asked in: ML breadth at every level.*

The right answer is a procedure, not a number. The wrong answers are "use the framework default" and "try a few values."

## What an L4 answer sounds like

> "I'd start with 1e-3 for Adam or 0.1 for SGD, then try a few values and pick the one that works best on validation."

Right number ranges, no method. You've copy-pasted from tutorials.

## What an L5 answer sounds like

> "Learning rate is the single most impactful hyperparameter, so I'd be deliberate about it. My standard procedure:
>
> **For a new model on a new dataset**: run an LR range test. Train for 200-500 steps with LR linearly increasing from 1e-7 to 1e-1. Plot loss vs LR on a log axis. The right LR is roughly in the steepest descent region, with a safety margin (1/3 of the LR where loss starts blowing up).
>
> **For SGD with momentum**: typically 0.01-0.1 with cosine annealing or step decay.
>
> **For Adam / AdamW**: typically 1e-4 to 1e-3 for most tasks; lower (1e-5 to 5e-5) for fine-tuning a pretrained model.
>
> **Schedule matters more than initial value**. A linear warmup over the first ~5% of steps followed by cosine decay to ~10% of the peak is the standard transformer recipe. Without warmup, large transformers diverge in the first hundred steps; the warm-up gives Adam's moment estimates time to stabilize.
>
> **For LLM pretraining**: peak LRs are surprisingly small (1e-4 to 6e-4) at standard scales, with decay to ~10% of peak. The Chinchilla-style training recipes use cosine decay over the full training horizon."

This is L5. You've named the LR range test, the schedule structure, the per-optimizer ranges, and the LLM-specific recipe.

## What an L6 answer sounds like

> "...and a few practical things I've learned:
>
> **The right LR depends on batch size.** The linear scaling rule (LR proportional to batch size) holds approximately for SGD but not for Adam. For Adam, the LR-batch relationship is roughly square-root. Doubling batch size, you'd want roughly 1.4&times; the LR for Adam.
>
> **Warmup matters more for adaptive optimizers.** Adam's per-parameter step size depends on the running variance estimate, which is unreliable for the first few hundred steps. Without warmup, you can get huge effective steps that destabilize training. SGD doesn't have this issue but benefits from warmup for other reasons (the gradient direction is noisy at the start).
>
> **For fine-tuning vs from-scratch, the LR is dramatically different.** Pretrained models have well-conditioned weights; large LRs will destroy them. The standard recipe is 5-100&times; smaller than from-scratch.
>
> **For LLM SFT and RLHF, even smaller.** Standard SFT recipes use LRs in the 1e-6 to 5e-5 range. RLHF/DPO is even more delicate; some papers use 1e-7. The right LR here is empirically determined and varies a lot by base model.
>
> **Cyclical LR or one-cycle policy** can sometimes beat cosine decay, especially for shorter training runs. Worth trying if cosine isn't giving great results.
>
> **The LR is downstream of the loss landscape.** A lot of what looks like 'wrong LR' is actually 'wrong initialization' or 'bad loss surface' or 'gradient pathology elsewhere.' If you find yourself needing very strange LRs to make training work, the LR isn't the issue."

This is L6. You're past the recipe and into the operational reality.

## The tells that get you a strong-hire vote

- You describe **a procedure** (LR range test) for picking the LR, not a guess.
- You distinguish **SGD vs Adam** ranges and **from-scratch vs fine-tuning** ranges.
- You bring up **the schedule** (warmup + decay) as separately important.
- You mention **LR-batch-size scaling**: especially the Adam-specific square-root rule.
- You acknowledge that **the LR is often downstream of other issues**.

## The tells that get you down-leveled

- Single number answer ("1e-3").
- No mention of warmup for transformers.
- "Just use the default in the framework", no calibration.
- Same LR for from-scratch and fine-tuning.
- No mention of schedule.

---

*Related: [How would you debug a model that's not learning?](/interviews/debug-model-not-learning/), [Walk me through the bias-variance tradeoff](/interviews/bias-variance-tradeoff/).*
