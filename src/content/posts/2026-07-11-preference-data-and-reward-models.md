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
<figure class="learning-figure editorial-illustration" aria-labelledby="reward-model-distribution-shift-edition-title"><span class="visual-fragment-anchor" id="reward-shift-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="reward-fit-panel-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="reward-optimize-panel-title" aria-hidden="true"></span><p class="visual-title" id="reward-model-distribution-shift-edition-title">Optimization can move responses beyond the proxy's validated coverage</p><svg viewBox="0 0 360 589" role="img" aria-labelledby="reward-model-distribution-shift-edition-svg-title reward-model-distribution-shift-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="reward-model-distribution-shift-edition-svg-title">Optimization can move responses beyond the proxy's validated coverage</title><desc id="reward-model-distribution-shift-edition-svg-desc">Fictional outputs R1 and R2 have proxy/human scores four/four and six/five. Optimized R3 has proxy nine and fresh human score one, outside the represented proxy-score band three to six. It is an illustrative possible shortcut, not an inevitable optimization result. New on-policy comparisons must revalidate the proxy.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional outputs; possible failure, not an</tspan><tspan x="180" dy="19">inevitable curve</tspan></text><rect x="132" y="91" width="81" height="210" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3;stroke-dasharray:5 3"></rect><text x="180" y="61" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Fresh human rubric score (0 to 5)</tspan></text><path d="M51 91L51 301" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M51 301L321 301" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="35" y="305" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="35" dy="0">0</tspan></text><text x="35" y="263" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="35" dy="0">1</tspan></text><text x="35" y="179" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="35" dy="0">3</tspan></text><text x="35" y="95" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="35" dy="0">5</tspan></text><text x="51" y="325" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="51" dy="0">0</tspan></text><text x="132" y="325" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="132" dy="0">3</tspan></text><text x="213" y="325" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="213" dy="0">6</tspan></text><text x="294" y="325" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="294" dy="0">9</tspan></text><text x="321" y="325" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="321" dy="0">10</tspan></text><circle cx="159" cy="133" r="6" style="fill:var(--viz-input-stroke)"></circle><text x="159" y="118" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="159" dy="0">R1</tspan></text><circle cx="213" cy="91" r="6" style="fill:var(--viz-input-stroke)"></circle><text x="231" y="76" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="231" dy="0">R2</tspan></text><circle cx="294" cy="259" r="6" style="fill:var(--viz-warning-stroke)"></circle><text x="294" y="244" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="294" dy="0">R3</tspan></text><text x="180" y="359" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Proxy reward (arbitrary units, 0 to 10)</tspan></text><text x="180" y="397" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Dashed band: represented proxy-score range, 3 to</tspan><tspan x="180" dy="19">6</tspan></text><g data-editorial-node><rect x="25" y="436" width="310" height="80" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="180" y="452.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Optimized R3: proxy 9, fresh human</tspan><tspan x="180" dy="19">score 1</tspan><tspan x="180" dy="19">possible style shortcut; inspect it</tspan><tspan x="180" dy="19">new on-policy pairs -&gt; revalidate</tspan></text></g><text x="180" y="553" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">A high proxy score outside coverage is not</tspan><tspan x="180" dy="19">validation.</tspan></text></svg><figcaption><strong>Read it this way:</strong> R3 improves the proxy while failing the fresh rubric in this toy example. A score-range band is not full distributional coverage. Inspect on-policy outputs and revalidate; do not assume every run diverges. <span class="visual-reader-check" data-reader-check="reward-model-distribution-shift"><strong>Check:</strong> Why can high held-out pair accuracy coexist with poor judgments on newly optimized responses?</span></figcaption></figure>

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
