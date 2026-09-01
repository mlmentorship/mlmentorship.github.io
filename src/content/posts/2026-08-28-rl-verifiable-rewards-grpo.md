---
title: "RL with verifiable rewards and GRPO"
description: "Train a model from outcomes that can be checked, using grouped samples to estimate relative advantage without a separate value model."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["RLVR", "GRPO", "group relative policy optimization", "verifiable rewards"]
---

## Summary

Reinforcement learning with verifiable rewards, often called RLVR, trains a policy using outcomes that a program or trusted process can check. Group Relative Policy Optimization, or GRPO, compares several outputs for the same input and uses their relative rewards to update the policy.

## Why AI labs care

Preference labels are useful and subjective. Some tasks have direct checks:

- a math answer matches a solver;
- code passes protected tests;
- a tool task reaches the correct state;
- a structured proof passes a checker;
- a database task satisfies exact constraints.

These checks can provide large amounts of training signal without asking humans to rate every output.

## The RLVR loop

1. Sample tasks from a training distribution.
2. Generate several outputs from the current policy.
3. Run a verifier for each output.
4. Convert verifier evidence into reward.
5. Update the policy while limiting large policy changes.
6. Evaluate on held-out tasks and verifier families.
7. Inspect new failure and exploit patterns.

The verifier defines what training rewards. An incomplete verifier creates an incomplete objective.

## GRPO in simple terms

For one task, sample a group of $G$ outputs with rewards $r_1,\ldots,r_G$.

A simple relative advantage is:

$$
A_i = \frac{r_i - \bar{r}}{s_r + \varepsilon},
$$

where $\bar{r}$ is the group mean and $s_r$ is the group standard deviation.

Outputs above the group average get positive advantage. Outputs below it get negative advantage.

<!-- visual:grpo-group-relative-signal -->
<figure class="learning-figure visual-wide plot-panel" aria-labelledby="grpo-group-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="grpo-group-title">When does a group of verified outputs create a learning direction?</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 760 390" role="img" aria-labelledby="grpo-group-svg-title grpo-group-svg-desc">
			<title id="grpo-group-svg-title">GRPO relative advantages for a mixed-reward group and an equal-reward group</title>
			<desc id="grpo-group-svg-desc">Two panels each contain four outputs sampled for one prompt. In the first panel, two outputs fail with reward zero and two pass with reward one. Their group mean is one half, so the failed outputs have negative relative advantage and the passing outputs have positive relative advantage. In the second panel, all four outputs pass with reward one. The mean is one and every reward equals it, so every relative advantage is zero and the group supplies no preference direction.</desc>
			<defs>
				<marker id="grpo-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="viz-arrow-forward" d="M0 0L10 5L0 10Z"></path></marker>
			</defs>
			<rect class="viz-plot-bg" x="20" y="42" width="350" height="300" rx="4"></rect>
			<rect class="viz-plot-bg" x="390" y="42" width="350" height="300" rx="4"></rect>
			<text class="viz-callout" x="195" y="25" text-anchor="middle">Prompt A · mixed verifier outcomes</text>
			<text class="viz-callout" x="565" y="25" text-anchor="middle">Prompt B · every output gets the same reward</text>
			<text class="viz-axis-label" x="42" y="72">1. Verify each sampled output</text>
			<text class="viz-axis-label" x="412" y="72">1. Verify each sampled output</text>
			<rect class="viz-node viz-node--input" x="42" y="88" width="66" height="54" rx="4"></rect>
			<rect class="viz-node viz-node--input" x="122" y="88" width="66" height="54" rx="4"></rect>
			<rect class="viz-node viz-node--output" x="202" y="88" width="66" height="54" rx="4"></rect>
			<rect class="viz-node viz-node--output" x="282" y="88" width="66" height="54" rx="4"></rect>
			<text class="viz-node-label" x="75" y="111">Fail</text><text class="viz-node-value" x="75" y="130">reward 0</text>
			<text class="viz-node-label" x="155" y="111">Fail</text><text class="viz-node-value" x="155" y="130">reward 0</text>
			<text class="viz-node-label" x="235" y="111">Pass</text><text class="viz-node-value" x="235" y="130">reward 1</text>
			<text class="viz-node-label" x="315" y="111">Pass</text><text class="viz-node-value" x="315" y="130">reward 1</text>
			<rect class="viz-node viz-node--output" x="412" y="88" width="66" height="54" rx="4"></rect>
			<rect class="viz-node viz-node--output" x="492" y="88" width="66" height="54" rx="4"></rect>
			<rect class="viz-node viz-node--output" x="572" y="88" width="66" height="54" rx="4"></rect>
			<rect class="viz-node viz-node--output" x="652" y="88" width="66" height="54" rx="4"></rect>
			<text class="viz-node-label" x="445" y="111">Pass</text><text class="viz-node-value" x="445" y="130">reward 1</text>
			<text class="viz-node-label" x="525" y="111">Pass</text><text class="viz-node-value" x="525" y="130">reward 1</text>
			<text class="viz-node-label" x="605" y="111">Pass</text><text class="viz-node-value" x="605" y="130">reward 1</text>
			<text class="viz-node-label" x="685" y="111">Pass</text><text class="viz-node-value" x="685" y="130">reward 1</text>
			<path class="viz-forward" style="marker-end:url(#grpo-arrow)" d="M195 150V174"></path>
			<path class="viz-forward" style="marker-end:url(#grpo-arrow)" d="M565 150V174"></path>
			<rect class="viz-node viz-node--focus" x="105" y="178" width="180" height="48" rx="4"></rect>
			<rect class="viz-node viz-node--focus" x="475" y="178" width="180" height="48" rx="4"></rect>
			<text class="viz-node-label" x="195" y="199">2. Group baseline</text><text class="viz-node-value" x="195" y="216">mean reward = 0.5</text>
			<text class="viz-node-label" x="565" y="199">2. Group baseline</text><text class="viz-node-value" x="565" y="216">mean reward = 1</text>
			<path class="viz-forward" style="marker-end:url(#grpo-arrow)" d="M195 234V258"></path>
			<path class="viz-forward" style="marker-end:url(#grpo-arrow)" d="M565 234V258"></path>
			<text class="viz-axis-label" x="42" y="278">3. Relative advantages</text>
			<text class="viz-axis-label" x="412" y="278">3. Relative advantages</text>
			<text class="viz-callout" x="75" y="306" text-anchor="middle">A₁ &lt; 0</text>
			<text class="viz-callout" x="155" y="306" text-anchor="middle">A₂ &lt; 0</text>
			<text class="viz-callout" x="235" y="306" text-anchor="middle">A₃ &gt; 0</text>
			<text class="viz-callout" x="315" y="306" text-anchor="middle">A₄ &gt; 0</text>
			<path class="viz-baseline" d="M42 318H348"></path>
			<text class="viz-label" x="195" y="334" text-anchor="middle">decrease failed outputs · increase passing outputs</text>
			<text class="viz-callout" x="445" y="306" text-anchor="middle">A₁ = 0</text>
			<text class="viz-callout" x="525" y="306" text-anchor="middle">A₂ = 0</text>
			<text class="viz-callout" x="605" y="306" text-anchor="middle">A₃ = 0</text>
			<text class="viz-callout" x="685" y="306" text-anchor="middle">A₄ = 0</text>
			<path class="viz-baseline" d="M412 318H718"></path>
			<text class="viz-label" x="565" y="334" text-anchor="middle">no within-group preference direction</text>
			<text class="viz-axis-label" x="195" y="368" text-anchor="middle">Variation creates a relative signal</text>
			<text class="viz-axis-label" x="565" y="368" text-anchor="middle">Equal rewards collapse the relative signal</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> compare each reward only with the mean for its own prompt. Prompt A has outcomes on both sides of 0.5, so GRPO gets negative and positive advantages without a learned value model. For Prompt B, every reward equals the mean; all numerators are zero, so this group says nothing about which output to prefer. The original example is checked against <a href="https://arxiv.org/abs/2402.03300">DeepSeekMath’s GRPO formulation</a> and the <a href="https://huggingface.co/docs/trl/main/en/grpo_trainer">TRL GRPO documentation</a>.</figcaption>
</figure>

GRPO then uses a clipped policy update, similar to PPO, and often adds a penalty that keeps the new policy near a reference policy. The exact loss differs across implementations.

The group reward provides a local baseline for the outputs sampled for one task. This removes the need to train a separate value function. A reference-policy penalty may still limit how far the policy changes.

## Why group sampling helps

Task difficulty varies. A reward of 1 may be common for an easy task and rare for a hard task. Comparing outputs for the same task reduces some of this variation.

Group samples also expose useful training cases:

- some outputs pass and some fail;
- several valid strategies have different lengths;
- one output exploits the verifier;
- all outputs fail, showing that the task may be too hard at the current stage.

If every output in a group receives the same reward, the relative signal is weak or zero.

## Reward design

Keep reward components clear:

- task correctness;
- format validity;
- policy or permission compliance;
- resource use;
- optional process evidence.

Do not let a soft style reward compensate for a failed hard constraint.

For code, passing visible tests is not enough. Protect hidden tests and prevent the policy from editing the grader. For tool agents, verify final state and unauthorized side effects.

## Outcome reward and process quality

A correct final answer does not prove that every reasoning step is correct. Outcome-only reward can reinforce lucky guesses, copied answers, or invalid reasoning that ends correctly.

Possible additions:

- intermediate checks with known validity;
- tool-state verification;
- process constraints;
- human review of a sample;
- tests where shortcuts fail;
- evaluation on changed task formats.

Do not add process rewards that are harder to validate than the outcome.

## Main failure modes

### Reward hacking

The policy finds a way to pass the check without doing the intended task.

### Sparse reward

Most outputs fail, so the update has little useful direction. Use easier tasks, partial checks, better sampling, or a curriculum.

### Narrow verifier coverage

Training improves only the formats and task families the verifier understands.

### Length growth

Longer outputs may find a correct answer more often and consume much more compute. Track accuracy against tokens and latency.

### Distribution collapse

The policy may learn one common strategy and lose useful diversity.

### Unstable updates

High-variance rewards or large policy changes can damage base capabilities. Track policy divergence and held-out behavior.

## Small example: math reasoning

A training task has a numeric answer that a solver can check.

- Generate eight candidate solutions.
- Check each final answer.
- Give correctness reward only when parsing succeeds and the answer matches.
- Compute relative advantages within the group.
- Update the policy with clipping and a reference-policy penalty.
- Evaluate on unseen problem families.
- Review correct answers with invalid reasoning to measure shortcut risk.

If all candidates fail, the batch provides little relative signal. A curriculum or better initial policy may be needed.

## In an interview

Use this order:

1. Define the task and trusted verifier.
2. Explain group sampling and relative advantage.
3. Explain why GRPO can avoid a value model.
4. State the policy constraint or reference penalty.
5. Cover sparse reward, reward hacking, and task coverage.
6. Separate final-answer checks from process claims.
7. Evaluate held-out task and verifier families.
8. Track quality, tokens, latency, and base-capability regressions.

## Common mistakes

- Saying verifiable reward means perfect reward.
- Treating passing tests as proof of a safe code change.
- Ignoring groups where every reward is equal.
- Assuming GRPO removes the need for careful reward design.
- Training and testing with the same task templates.
- Reporting only final-answer accuracy.
- Ignoring compute growth and capability regressions.

*Related: [RLHF and DPO](/concepts/rlhf-and-dpo/), [PPO](/concepts/ppo/), and [design post-training data and an RL environment](/questions/design-post-training-data-and-rl-environment/).*