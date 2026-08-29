---
title: "Why does Adam sometimes generalize worse than SGD?"
description: "Adam usually trains faster but in some settings finds sharper minima with worse generalization. The senior answer names the regimes where this happens and the modern fixes."
date: "2026-04-29"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth, especially in CV-leaning or research-track interviews.*

A nuanced question. The L4 candidate doesn't know there's a difference. The L6 candidate explains the implicit-regularization argument, the regimes where SGD wins, and how AdamW + careful schedules close most of the gap.

## What an L4 answer sounds like

> "Adam adapts the learning rate per parameter, so it converges faster than SGD but might not generalize as well."

True at a slogan level, no mechanism. You've heard the fact, not the explanation.

## What an L5 answer sounds like

> "There are two main explanations:
>
> 1. **Sharp vs flat minima.** SGD's per-mini-batch noise biases it toward flat minima, which empirically generalize better than sharp ones. Adam's adaptive scaling reduces this implicit noise, so it tends to converge to sharper minima with similar training loss but worse test performance.
>
> 2. **Weight decay coupling.** Original Adam couples weight decay with the adaptive learning rate, effectively reducing weight decay on parameters with large gradient magnitudes (the wrong direction). AdamW decouples weight decay and largely fixes this.
>
> The empirical picture in 2026:
> - **Vision (CNNs)**: SGD with momentum sometimes beats AdamW on classic ImageNet-style problems. The gap is small and getting smaller.
> - **NLP (transformers)**: AdamW is essentially always better. SGD doesn't even train large transformers well.
> - **LLMs**: AdamW is the default at all scales. No serious team uses SGD for LLM pretraining."

This is L5. You've named the two mechanisms and given the regime breakdown.

## What an L6 answer sounds like

> "...subtler points:
>
> **The 'sharp minima generalize worse' argument is contested.** Multiple papers [(Dinh et al. 2017)](https://arxiv.org/abs/1605.08803) showed that sharpness can be reparameterized away, undermining the simplest version of the argument. The empirical generalization gap exists; the precise reason is still debated. Don't bet a strong answer on the sharp/flat framing alone.
>
> **AdamW closes most of the gap with SGD on vision.** The [Loshchilov & Hutter 2017](https://arxiv.org/abs/1711.05101) paper showed AdamW + cosine schedule + careful warmup matches or beats SGD on most benchmarks. The 'SGD generalizes better' folk wisdom predates AdamW and is largely outdated.
>
> **For very large batch training, neither classic Adam nor SGD work well.** LAMB (layer-wise adaptive momentum) was designed for very large batch transformer training and outperforms both at batch size 16K+.
>
> **Optimizer choice rarely matters at scale.** With enough data, the optimizer choice is dominated by data quality, model architecture, and learning rate schedule. Don't over-tune the optimizer at the expense of the schedule."

## Tells that get you a strong-hire vote

- You name **flat-vs-sharp minima** but acknowledge it's contested.
- You distinguish **AdamW from Adam** and name the weight-decay coupling fix.
- You give the **regime breakdown**: SGD competitive on CNNs, AdamW dominant on transformers/LLMs.
- You mention **LAMB** for very large batch training.

## Tells that get you down-leveled

- Asserting "SGD always generalizes better."
- Confusing Adam with AdamW.
- No mention of weight decay coupling.
- Suggesting SGD for LLM training.

## Common follow-up

"What learning rate schedule do you use with AdamW?"

The L6 answer:

> "For transformers: linear warmup over the first 1-5% of steps to peak LR, then cosine decay to ~10% of peak over the rest of training. Warmup is critical: Adam's second moment v is unreliable in the first few hundred steps, and full LR causes divergence. For very long training (LLMs trained for trillions of tokens), some recipes use constant LR after a brief warmup, then a final decay phase ('infinite' LR schedule). The exact shape matters less than having warmup and not over-decaying."

---

*Related: [Adam, AdamW, and modern optimizer choices](/concepts/adam-and-adamw/), [How do you choose a learning rate?](/questions/how-to-choose-learning-rate/), [Regularization](/concepts/regularization/).*
