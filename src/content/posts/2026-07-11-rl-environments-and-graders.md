---
title: "RL environments and graders for language-model agents"
description: "The environment defines what behavior is possible; the grader defines what optimization values. Both need versioning, adversarial tests, and evidence."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

An agent RL environment defines observations, actions, transitions, tools, and terminal conditions; a grader converts the resulting trajectory into structured evidence and a training signal.

For language-model agents, the environment is part simulator, part benchmark, and part specification. If it omits a permission boundary, rewards only the final answer, or leaks the expected solution, training can improve the score while degrading the intended behavior.

The grader is not a passive metric. It is an objective the policy will search for exploits.

**Learning objective:** trace one agent episode across the environment and grader contracts, then explain why a hard policy failure must remain separate from quality scores.

<!-- visual:environment-grader-contract-handoff -->
<figure class="learning-figure" aria-labelledby="environment-grader-handoff-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="environment-grader-handoff-title">Where does an episode become a training signal?</p>
	<div class="visual-grid--two" role="group" aria-label="Six-step handoff from an environment contract that produces a trajectory to a grader contract that produces gated component evidence">
		<section class="visual-panel" aria-labelledby="environment-contract-lane-title">
			<h4 id="environment-contract-lane-title">ENVIRONMENT CONTRACT · WHAT CAN HAPPEN</h4>
			<p><strong>1 · Reset the episode</strong><br />Sample initial state and hidden truth; assign identity, permissions, tools, and budgets.</p>
			<p><strong>2 · Validate each action</strong><br />Apply tool schemas, side effects, stochastic transitions, and resource limits.</p>
			<p><strong>3 · Preserve the result</strong><br />Emit terminal state plus a reproducible event log of actions, observations, and side effects.</p>
			<p><strong>Output: an evidence-bearing trajectory</strong><br />Missing boundaries here make forbidden behavior possible or expected answers visible.</p>
		</section>
		<section class="visual-panel" aria-labelledby="grader-contract-lane-title">
			<h4 id="grader-contract-lane-title">GRADER CONTRACT · WHAT COUNTS</h4>
			<p><strong>4 · Read outcome and process</strong><br />Check terminal state against hidden truth and inspect the complete event log.</p>
			<p><strong>5 · Apply hard gates first</strong><br />A permission, policy, or integrity failure makes the trajectory ineligible; quality cannot offset it.</p>
			<p><strong>6 · Return component evidence</strong><br />Keep task success, grounding, efficiency, communication, and uncertainty separate before optimization.</p>
			<p><strong>Output: a structured training signal</strong><br />Weak or leaked checks teach the policy to exploit the score instead of the intended behavior.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> move left to right across one episode. The environment first constrains and records what the agent can do; only then can the grader compare outcome and process evidence with hidden truth. Reject hard violations before combining any quality evidence, and audit both contracts after policy updates because optimization searches for omissions in either one. Original synthesis informed by the <a href="https://incompleteideas.net/book/the-book-2nd.html">Sutton and Barto (2018)</a>, <a href="https://gymnasium.farama.org/api/env/">Gymnasium environment API</a>, and the <a href="https://github.com/SWE-bench/SWE-bench">SWE-bench evaluation harness</a>.</figcaption>
</figure>

## Environment contract

Define:

- initial-state distribution;
- observable and hidden state;
- tool schemas and side effects;
- identity, permissions, and resource budgets;
- stochastic transition behavior;
- action validation;
- success, failure, timeout, and reset;
- reproducibility and event log;
- versioned dependencies and external services.

A support agent and a coding agent need different truth. One may validate database state and permission checks; the other needs repository state, tests, sandbox behavior, and patch scope.

## Grader stack

### Deterministic graders

Use executable checks for exact state, tests, policy rules, budgets, or verifier outputs. They are reproducible but cover only formalized behavior.

### Model graders

Use models for semantic correctness, relevance, communication, or open-ended quality. They scale but have bias, variance, prompt sensitivity, and attack surfaces.

### Human review

Use humans for ambiguous, novel, or high-consequence cases and to calibrate automated graders. Human labels also vary and need a protocol.

A strong system combines them and returns component evidence rather than one unexplained reward.

## Outcome and process

Separate:

- final task result;
- policy compliance;
- trajectory efficiency;
- tool grounding;
- communication;
- uncertainty or abstention;
- side effects.

Some constraints are hard gates. A correct answer produced through unauthorized data access is not partially successful.

## Coverage

Build episodes across task difficulty, tool failures, ambiguous instructions, adversarial context, long horizon, sparse reward, permission changes, and unusual but valid solutions. Hold out behavioral families to test generalization.

Avoid contamination. If training sees exact evaluator tasks or grader artifacts, the benchmark measures recall and exploit learning rather than capability transfer.

## Grader evaluation

Before training, test:

- human agreement and calibration;
- false positives on unusual valid behavior;
- false negatives on polished unsafe behavior;
- invariance to irrelevant style and length;
- sensitivity to the intended behavior;
- adversarial optimization;
- stability across model and environment versions;
- evidence quality for debugging.

After training, repeat with on-policy trajectories. The policy changes the grader's input distribution.

## Common confusions

- **"An environment is a prompt and expected answer."** Agent environments include state, actions, transitions, side effects, and termination.
- **"One reward is simpler."** It hides which behavior improved and lets components compensate incorrectly.
- **"Unit tests prove coding-agent quality."** Tests can be incomplete, weakened, or overfit.
- **"A model grader understands intent."** It predicts from a prompt and can share target-model blind spots.
- **"More episodes mean better coverage."** Near-duplicate tasks inflate count without expanding behavior.
- **"Offline reward improvement means product improvement."** Real users, tools, and incentives can differ.

## In an interview

Start with the capability and forbidden behavior. Then define episode semantics, grader evidence, data distribution, adversarial cases, contamination controls, versioning, and the external decision the reward supports.

*Related: [preference data and reward models](/concepts/preference-data-and-reward-models/), [reward shaping](/concepts/reward-shaping/), and [post-training environment lab](/prep/labs/post-training-environment/).*
