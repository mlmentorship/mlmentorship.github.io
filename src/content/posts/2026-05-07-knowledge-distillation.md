---
title: "Knowledge distillation"
description: "Train a small student to match a large teacher's outputs. The student gets richer signal than from hard labels because the teacher's soft probabilities encode similarity structure."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

**Knowledge distillation** trains a student model with a loss against a teacher's soft predictions, not the hard label. The student learns the teacher's full output distribution, which carries information about how classes relate ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)).

## Why it matters

Hard labels say "this is a 7." Teacher logits say "94 percent 7, 4 percent 1, 1 percent 9, everything else 0.01." That extra structure tells the student that 7 looks more like 1 than like 9. A small model trained against this signal usually beats the same model trained from scratch on hard labels at matched compute.

Distillation is the dominant technique for shrinking large models in production. DistilBERT, TinyBERT, MobileBERT, and most production LLMs ship distilled variants. Often combined with [pruning](/concepts/pruning/) and [quantization](/concepts/quantization-int8-int4-fp8-and-the-inference-cost-picture/).

## The mechanism

Given teacher logits $z^T$, student logits $z^S$, hard label $y$, temperature $\tau > 1$:

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{CE}}(y, \text{softmax}(z^S)) + (1 - \alpha) \cdot \tau^2 \cdot \text{KL}\!\left(\text{softmax}(z^T / \tau) \,\|\, \text{softmax}(z^S / \tau)\right).
$$

- **Temperature** $\tau$ softens both distributions. Higher $\tau$ exposes more of the teacher's "dark knowledge" about non-target classes. $\tau = 2$ to $5$ is typical.
- **$\tau^2$ scaling** is needed because softening reduces gradient magnitude by $1/\tau^2$.
- **$\alpha$** weights the hard-label loss. $\alpha = 0$ gives pure distillation; $\alpha \in [0.1, 0.5]$ is common.

## Variants

| Variant | What it matches |
|---|---|
| **Logit distillation** (above) | Teacher output logits |
| **Feature distillation** ([FitNets](https://arxiv.org/abs/1412.6550)) | Intermediate hidden states |
| **Attention distillation** ([TinyBERT](https://arxiv.org/abs/1909.10351)) | Teacher attention maps |
| **Sequence-level distillation** ([Kim & Rush, 2016](https://arxiv.org/abs/1606.07947)) | Teacher's most likely outputs (for autoregressive models) |
| **Self-distillation** | Teacher and student are the same architecture; sometimes the teacher is a previous training checkpoint |

For LLMs, sequence-level distillation against teacher samples (or rejection-sampled teacher outputs) is the dominant recipe. Logit distillation is impractical at vocab size 100k+.

## When it works and when it doesn't

Works well when:

- Teacher is significantly better than what the student could reach alone.
- Student capacity is at least 10 to 20 percent of the teacher.
- Training data overlaps the teacher's training distribution.

Fails when:

- Student is too small. Capacity gap is the dominant ceiling.
- Teacher is already small. The "dark knowledge" margin is thin.
- Distribution shift. Teacher predictions are unreliable on student's deployment data.

## Common pitfalls

- **Forgetting $\tau^2$ scaling.** Without it, the KL term has tiny gradients and the hard-label term dominates.
- **Distilling only logits when feature distillation would help.** For very small students, intermediate matching is often required.
- **Skipping the temperature.** $\tau = 1$ collapses the teacher's distribution to nearly one-hot for confident predictions; you lose most of the signal.
- **Training student on teacher-correct examples only.** The interesting signal is on examples where the teacher is uncertain. Use the full training set.

## Related

- [Pruning](/concepts/pruning/).
- [Quantization](/concepts/quantization-int8-int4-fp8-and-the-inference-cost-picture/).
- [Cross-entropy loss](/concepts/cross-entropy-and-negative-log-likelihood/).
