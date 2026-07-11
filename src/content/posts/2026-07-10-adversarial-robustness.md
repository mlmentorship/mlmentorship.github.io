---
title: "Adversarial robustness"
description: "Small worst-case perturbations, threat models, adversarial training, robust evaluation, and the difference between security and ordinary distribution shift."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Definition

An adversarial example is an input intentionally perturbed to cause failure while remaining within a defined threat model. Robustness is meaningful only relative to attacker knowledge, capabilities, norm or semantic constraints, and the defended system.

## First-order attacks

For loss $L(\theta, x, y)$, FGSM uses

$$x' = x + \epsilon\,\text{sign}(\nabla_x L).$$

Projected gradient descent repeats gradient steps and projects back into the allowed set. Strong evaluation uses multiple restarts and adaptive attacks rather than one weak attack.

## Defenses

Adversarial training optimizes against generated worst-case perturbations and remains the strongest general baseline for norm-bounded attacks. It is expensive and often trades clean accuracy for robust accuracy. Certified defenses prove robustness within a limited region but may not scale to realistic semantic threats.

## Why evaluation fails

- Gradient masking makes weak attacks appear unsuccessful.
- The attack does not adapt to preprocessing or randomness.
- The threat model is irrelevant to the real attacker.
- Robustness is measured on average while rare failures are catastrophic.
- The deployed pipeline contains non-model attack surfaces.

## Interview answer

1. Specify threat model and acceptable failure.
2. Establish clean and attacked baselines.
3. Use adaptive, sufficiently strong attacks and independent tools.
4. Discuss adversarial training and its cost/accuracy trade-off.
5. Expand from model robustness to detection, rate limiting, human review, and incident response.

## Common confusions

- **“Noise augmentation gives adversarial robustness.”** Random noise does not approximate worst-case optimization reliably.
- **“High PGD accuracy means secure.”** Only within the tested threat model and attack implementation.
- **“Adversarial and natural robustness are the same.”** They can interact but test different failure processes.

*Related: [regularization](/concepts/regularization/), [model interpretability](/concepts/model-interpretability/), and [epistemic uncertainty](/concepts/epistemic-vs-aleatoric-uncertainty/).*
