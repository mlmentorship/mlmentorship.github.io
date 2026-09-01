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

**Learning objective:** explain why reward-model validation on collected comparisons does not guarantee reliable rewards after the policy is optimized.

<!-- visual:reward-model-distribution-shift -->
<figure class="learning-figure" aria-labelledby="reward-shift-title">
	<p class="visual-kicker">Proxy shift</p>
	<p class="visual-title" id="reward-shift-title">Why can a validated reward model fail during policy optimization?</p>
	<div class="visual-grid--two" role="group" aria-label="Comparison between fitting a reward model on sampled responses and optimizing a policy into a shifted response distribution">
		<section class="visual-panel" aria-labelledby="reward-fit-panel-title">
			<h4 id="reward-fit-panel-title">1 · FIT ON SAMPLED RESPONSES</h4>
			<p><strong>Coverage is chosen</strong><br />Prompts, policy checkpoints, temperatures, and failure slices determine which response pairs appear.</p>
			<p><strong>Judgments define the target</strong><br />Annotators apply a rubric, including ties, both-bad labels, and disagreements.</p>
			<p><strong>The proxy learns locally</strong><br />The reward model learns which sampled response wins; held-out pairs test that same measurement process.</p>
			<p><strong>Known evidence</strong><br />Accuracy, calibration, and slice results describe behavior on represented comparisons.</p>
		</section>
		<section class="visual-panel" aria-labelledby="reward-optimize-panel-title">
			<h4 id="reward-optimize-panel-title">2 · OPTIMIZE INTO NEW RESPONSES</h4>
			<p><strong>The policy follows the proxy</strong><br />Updates increase outputs that the fixed reward model scores highly.</p>
			<p><strong>The response distribution moves</strong><br />New outputs can leave the comparisons on which the proxy was trained and validated.</p>
			<p><strong>Search exposes shortcuts</strong><br />The optimizer can find verbosity, style, or other features that raise predicted reward without improving the rubric's target.</p>
			<p><strong>Required response</strong><br />Audit optimized outputs, refresh on-policy comparisons, and revalidate by slice before trusting the next update.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> read the left panel as a bounded measurement claim, then move right. Validation says the reward model predicts judgments on represented comparisons; optimization changes which responses it sees and searches specifically for high scores. That is why held-out pair accuracy cannot close the loop: inspect optimized outputs, collect fresh comparisons, and recalibrate the proxy. This original synthesis is informed by <a href="https://arxiv.org/abs/1909.08593">Ziegler et al. (2019)</a>, <a href="https://arxiv.org/abs/2203.02155">Ouyang et al. (2022)</a>, and <a href="https://arxiv.org/abs/2210.10760">Gao et al. (2023)</a>.</figcaption>
</figure>

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
