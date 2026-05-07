---
title: "Mixup and CutMix"
description: "Two data-augmentation schemes that train on convex combinations of pairs of inputs and their labels. Strong regularization for image classification; sometimes used in audio and tabular."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

**Mixup** [(Zhang et al., 2018)](https://arxiv.org/abs/1710.09412) trains the model on convex combinations of pairs of training examples: $\tilde x = \lambda x_i + (1 - \lambda) x_j$ and $\tilde y = \lambda y_i + (1 - \lambda) y_j$ with $\lambda \sim \text{Beta}(\alpha, \alpha)$. **CutMix** [(Yun et al., 2019)](https://arxiv.org/abs/1905.04899) instead pastes a rectangular patch from $x_j$ onto $x_i$ and mixes labels by the area ratio.

## Why it matters

Both techniques regularize by training on examples between the original training points. Empirically:

- Improve top-1 accuracy on ImageNet by ~1–2% over baseline.
- Improve calibration (predicted probabilities track accuracy better).
- Improve robustness to label noise and adversarial perturbations.
- Standard in modern image classification recipes (timm, ConvNeXt, ViT-style training).

Less common in NLP (token mixing is non-trivial) and in pretraining (large data already covers the input space well). Sometimes used in audio (mix waveforms or spectrograms) and tabular (interpolate features).

## Mechanism

### Mixup

For each batch, sample $\lambda \sim \text{Beta}(\alpha, \alpha)$ once (or per sample) with $\alpha$ small (typical 0.2–0.4). For paired examples $(x_i, y_i)$ and $(x_j, y_j)$:

$$
\tilde x = \lambda x_i + (1 - \lambda) x_j, \quad \tilde y = \lambda y_i + (1 - \lambda) y_j.
$$

Train normally on $(\tilde x, \tilde y)$ with cross-entropy. The label is a soft target.

### CutMix

Sample $\lambda \sim \text{Beta}(\alpha, \alpha)$. Pick a random rectangle in $x_i$ of area $1 - \lambda$ (e.g., width and height $\sqrt{1 - \lambda}$ times the image). Paste the corresponding region from $x_j$ into $x_i$. Mix labels by the area ratio $\lambda$.

The resulting image has a clear local boundary (no blending). Models trained on CutMix often produce more localized class activations.

## Choosing $\alpha$

| Setting | Mixup $\alpha$ | CutMix $\alpha$ |
|---------|---------------|----------------|
| ImageNet from scratch | 0.2 | 1.0 |
| Small datasets | 0.2 (more aggressive Mixup hurts) | 1.0 |
| ViT training | 0.2 + CutMix 1.0 (used together) |. |

$\alpha \to 0$ gives near-original samples (almost no mixing); $\alpha \to \infty$ gives $\lambda \approx 0.5$ (always equally mixed). $\alpha$ between 0.2 and 1 is the empirical sweet spot.

## Why it works (intuition)

- **Vicinal risk minimization** [(Chapelle et al., 2001)](https://papers.nips.cc/paper/2000/hash/ba9a56ce0a9bfa26e8ed9e10b2cc8f46-Abstract.html): training on a vicinity around each point regularizes the decision boundary.
- **Empirically**: smoother decision functions, better calibration, less overconfidence on out-of-distribution inputs.
- **Equivalent to** an implicit form of weight regularization in the linear case.

## Common pitfalls

- **Mixing labels but not inputs.** Some implementations mix targets without mixing inputs; this is just label noise, not Mixup.
- **Combining with strong cropping.** Mixup + RandomResizedCrop + label smoothing + AutoAugment is the modern recipe but can over-regularize small datasets.
- **Using on detection / segmentation directly.** Class labels mix easily; bounding boxes do not. Variants like Mosaic (YOLOv4) handle this.
- **Forgetting to disable for evaluation.** Eval should use clean images.

## Related

- [Label smoothing](/reference/label-smoothing/). Another way to soften targets.
- [Dropout](/reference/dropout/). Stochastic activation regularization.
- [Regularization](/reference/regularization/). Overview.
