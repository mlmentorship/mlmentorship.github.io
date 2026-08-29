---
title: "Preference data and reward models"
description: "Preference optimization is a measurement system: sampling policy, annotator protocol, disagreement, calibration, and shift determine the signal."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Preference data records comparative judgments among model outputs, and a reward model learns a scoring function that predicts those judgments for training or evaluation.

RLHF diagrams often compress the hardest part into "collect preferences." The signal depends on the sampled prompts and responses, the judges, the rubric, and the treatment of disagreement. It also changes as the policy moves away from the reward model's training distribution.

A powerful optimizer amplifies measurement defects. Better preference-data design can matter more than another RL algorithm.

## Constructing comparisons

For prompt $x$, sample responses $y_a$ and $y_b$ from policies and decoding settings chosen to expose meaningful differences. Ask an annotator which response better satisfies an explicit rubric, allows a tie, or marks both unacceptable.

Pairwise judgments are often easier than absolute scores, but sampling determines value. Comparing one obviously broken answer with one strong answer teaches less than a near-boundary pair that isolates factuality, instruction following, style, or safety.

Useful sampling strategies include:

- policy checkpoints across training;
- multiple temperatures and seeds;
- uncertainty or disagreement sampling;
- known failure slices;
- adversarially generated pairs;
- human and model-written alternatives;
- active selection near the current reward boundary.

## Annotator protocol

Define the target construct before collecting labels. "Which is better?" mixes correctness, relevance, verbosity, style, and safety differently across annotators.

A sound protocol includes:

- rubric and precedence among criteria;
- examples and counterexamples;
- tie and both-bad options;
- qualification and calibration tasks;
- blind randomization of response order and model identity;
- duplicate items for consistency;
- escalation for high-impact ambiguity;
- slice-level agreement and quality monitoring;
- privacy and worker-welfare constraints.

Disagreement is data. It can reveal ambiguous prompts, plural user values, or a broken rubric rather than a bad annotator.

## Reward-model objective

A common Bradley-Terry form models:

$$
P(y_w \succ y_l \mid x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)).
$$

The loss encourages the preferred response to receive higher reward. Only differences are identified; the absolute reward scale is arbitrary without further calibration.

## Evaluate the reward model

Use held-out prompts and behavioral families. Measure:

- pairwise accuracy and calibration;
- performance by criterion and slice;
- agreement with independent humans;
- order, length, style, and verbosity bias;
- out-of-distribution behavior;
- adversarially optimized outputs;
- ranking stability across policy checkpoints;
- uncertainty and abstention quality.

High random-split accuracy can coexist with failure on a new policy because response distribution changes during optimization.

## Reward hacking and shift

Once a policy optimizes the reward model, it searches for outputs the model scores highly, including shortcuts absent from ordinary samples. Mitigations include on-policy data refresh, adversarial sampling, reward ensembles, uncertainty penalties, KL constraints, independent verifiers, and held-out audits.

None remove the need to inspect what the optimized policy discovers.

## Common confusions

- **"More preference pairs always help."** Redundant easy pairs add little signal.
- **"Annotator disagreement is noise."** It can expose underspecified values or prompts.
- **"Reward accuracy equals reward quality."** Calibration, slice behavior, and optimization robustness matter.
- **"The reward score is utility."** It is a learned proxy with arbitrary scale and bounded coverage.
- **"DPO removes preference-data problems."** It removes the explicit reward model, not bias or coverage in the comparisons.
- **"Model graders eliminate humans."** They inherit model biases and require independent calibration.

## In an interview

Describe prompt and response sampling, rubric, annotator quality, disagreement, objective, held-out families, calibration, policy shift, reward hacking, and the online outcome the proxy is meant to improve.

*Related: [RLHF and DPO](/concepts/rlhf-and-dpo/), [PPO](/concepts/ppo/), and [design post-training data and an RL environment](/questions/design-post-training-data-and-rl-environment/).*
