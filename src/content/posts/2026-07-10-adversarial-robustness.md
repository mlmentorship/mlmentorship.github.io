---
title: "Adversarial robustness"
description: "Small worst-case perturbations, threat models, adversarial training, robust evaluation, and the difference between security and ordinary distribution shift."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A model can hit 99% test accuracy and still fail on inputs a human cannot tell apart from clean ones, because average-case accuracy says nothing about the worst case an attacker can search for. That gap matters wherever an adversary is present: fraud, content moderation, malware, authentication, autonomous perception. An adversarial example is an input deliberately perturbed to cause failure while staying inside a defined threat model, and robustness only means something relative to that threat model: the attacker's knowledge, capabilities, norm or semantic budget, and the system being defended.

## First-order attacks

For loss $L(\theta, x, y)$, FGSM takes a single signed-gradient step:

$$x' = x + \epsilon\,\text{sign}(\nabla_x L).$$

Projected gradient descent iterates gradient steps and projects back into the allowed set. A meaningful evaluation uses multiple restarts and attacks that adapt to the defense, not one weak attack.

## Defenses

Adversarial training optimizes against generated worst-case perturbations and remains the strongest general baseline for norm-bounded attacks. It is expensive and usually trades clean accuracy for robust accuracy. Certified defenses prove robustness within a limited region but may not scale to realistic semantic threats.

## Why evaluation fails

- Gradient masking makes weak attacks fail, which creates false confidence in the defense.
- The attack does not adapt to preprocessing or randomness.
- The threat model is irrelevant to the real attacker.
- Robustness is measured on average while the rare failures are the catastrophic ones.
- The deployed pipeline has non-model attack surfaces the eval ignores.

## In an interview

1. Specify the threat model and the failure you care about.
2. Establish clean and attacked baselines.
3. Use adaptive, sufficiently strong attacks and independent tools.
4. Discuss adversarial training and its cost/accuracy trade-off.
5. Widen from model robustness to detection, rate limiting, human review, and incident response.

## Common confusions

- **"Noise augmentation gives adversarial robustness."** Random noise does not reliably approximate worst-case optimization.
- **"High PGD accuracy means secure."** Only within the tested threat model and attack implementation.
- **"Adversarial and natural robustness are the same."** They can interact but test different failure processes.

*Related: [regularization](/concepts/regularization/), [model interpretability](/concepts/model-interpretability/), and [epistemic uncertainty](/concepts/epistemic-vs-aleatoric-uncertainty/).*
