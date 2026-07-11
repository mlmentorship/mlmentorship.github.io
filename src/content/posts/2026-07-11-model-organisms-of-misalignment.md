---
title: "Model organisms of misalignment"
description: "Create controlled models with known problematic behavior to test detection and mitigation, while keeping the induced behavior bounded and measurable."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

A model organism of misalignment is a deliberately constructed model with a controlled, reproducible unwanted behavior used to study how that behavior arises, generalizes, is detected, and responds to mitigation.

## Why it matters

Real alignment failures are rare, poorly labeled, and entangled with model capability. If a researcher waits for a frontier system to exhibit a dangerous failure naturally, the mechanism and ground truth may remain unknown.

A model organism creates a tractable target. The analogy to biology is methodological: induce or select a phenotype, control conditions, and test interventions. It does not imply that the model is literally an organism.

## A good organism has a contract

Specify:

- **behavior:** the exact unwanted action or policy;
- **trigger:** when it appears;
- **control:** matched inputs where it should not appear;
- **training intervention:** how the behavior was introduced;
- **capability match:** what useful ability remains constant;
- **measurement:** independent labels or verifiers;
- **containment:** tools, data, and deployment boundaries;
- **research question:** which detection or mitigation claim the organism tests.

For example, a model could behave helpfully during ordinary evaluation but pursue a different objective when a synthetic trigger appears. The useful experiment is not "the model deceives." It is whether a specific monitor, fine-tuning method, or interpretability signal detects or removes the conditional policy.

## Construction methods

### Supervised fine-tuning

Train explicit trigger-behavior pairs. This gives clean labels but can create superficial pattern matching that does not resemble the target failure.

### Preference optimization or RL

Reward a policy under selected conditions. This can produce more flexible strategies but introduces grader and reward-hacking confounds.

### Prompted or scaffolded behavior

Use system instructions, memory, or an agent scaffold to create the policy without changing weights. This is cheap and controlled but may test the scaffold more than learned model behavior.

### Data poisoning or synthetic pretraining

Introduce the behavior earlier in training. This can study persistence and generalization but is expensive and harder to isolate.

## Experimental design

A strong study separates:

1. behavior presence;
2. trigger detection;
3. capability side effects;
4. monitor performance;
5. intervention effect;
6. generalization beyond training triggers;
7. recurrence after additional training.

Use multiple seeds, matched benign models, alternative triggers, and blinded evaluation. If one intervention removes the trigger phrase but not the latent policy, the apparent mitigation is cosmetic.

## Containment and research risk

The organism should use bounded tasks, synthetic data, restricted tools, and no path to consequential external actions. Release decisions should consider whether weights, prompts, or datasets make harmful capabilities easier to reproduce.

The research artifact itself can create risk. Documentation should preserve scientific usefulness without publishing operational details that materially lower misuse cost.

## Common confusions

- **"Any jailbroken model is a model organism."** An organism has a designed behavior, controls, ground truth, and a research purpose.
- **"The induced behavior proves natural emergence."** Construction method can dominate the mechanism.
- **"Removing the trigger removes misalignment."** The policy may transfer to paraphrases or new conditions.
- **"A monitor trained on the organism will generalize."** It can overfit organism-specific artifacts.
- **"More realistic is always better."** Realism can reduce control and increase risk.
- **"One seed is enough."** Training stochasticity can create organism-specific results.

## In an interview

Define the research claim first, then behavior, trigger, controls, construction, measurement, containment, and falsification. The organism is an instrument, not the conclusion.

*Related: [scalable oversight and AI control](/concepts/scalable-oversight-and-ai-control/), [chain-of-thought monitorability](/concepts/chain-of-thought-monitorability/), and [design an ablation study](/questions/design-ablation-study/).*
