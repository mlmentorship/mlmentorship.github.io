---
title: "Design post-training data, an RL environment, and its grader"
description: "Turn one capability into episodes, evidence-bearing graders, adversarial data, and a training signal that cannot reward the wrong process."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> Improve a tool-using agent's ability to resolve support tickets. Design the data, environment, grader, training loop, and evaluation. The agent must not access unauthorized account data.

Design post-training data and an evaluation environment that improve a support agent's authorized task completion rather than merely its reward score.

## Problem

**Context.** A hypothetical tool-using agent resolves support tickets using account data and actions governed by user permissions.
Training may use demonstrations, preferences, or environment feedback.

**Task.** Specify target behavior, data provenance, environment state and tools, graders, training objective, and independent evaluation.
Explain how you would detect reward exploitation, overfitting, and failure to transfer to real support work.

**Constraints.** Unauthorized account access is forbidden, even if it improves measured ticket completion.
Separate training rewards from held-out success evidence; preserve privacy, environment versioning, and realistic action consequences in contained tests.

## The design sequence

### 1. Define the capability distribution

Specify ticket types, user states, tools, permissions, ambiguity, long-horizon depth, and failure severity. A single average success rate will hide whether the model improves on easy password resets while failing high-risk billing changes.

### 2. Define episode semantics

An episode needs:

- observable state and hidden ground truth;
- available actions and tool schemas;
- permission boundaries;
- stochasticity and reset behavior;
- terminal success, terminal failure, and timeout;
- an audit log sufficient to reproduce the trajectory.

### 3. Separate outcome from process

Use a structured grade rather than one opaque scalar:

- **task success:** was the ticket actually resolved?
- **policy compliance:** were permissions and required confirmations respected?
- **process quality:** were actions efficient, grounded, and non-duplicative?
- **communication quality:** was the user-facing result accurate and calibrated?

Some violations are gates, not tradeable penalties. Unauthorized access cannot be offset by a polite final response.

<p class="visual-kicker">Learning objective</p>
<p class="visual-title">Decide whether a successful trajectory is eligible for quality scoring by applying the policy gate first.</p>

<!-- visual:post-training-policy-gate -->
<figure class="learning-figure plot-panel" aria-labelledby="post-training-policy-gate-heading">
	<h3 id="post-training-policy-gate-heading" class="visual-title">Build reproducible episodes; gate reward with action evidence and test transfer separately</h3><div class="visual-argument" data-visual-argument="post-training-policy-gate" data-argument-kind="matrix" data-argument-composition="episode-evidence-and-policy-grade"><svg viewBox="0 0 360 1770" role="img" aria-label="Build reproducible episodes; gate reward with action evidence and test transfer separately. Version the contained environment, permissions, tools, resets, and terminal rules. Save trajectory evidence. Both trajectories resolve the ticket, but only authorized access is eligible for quality scoring. Deterministic policy and state checks precede calibrated semantic or human judgment. Keep failed and contrastive cases with explicit labels; do not reward the violation. Train on declared families, then test held-out states and tools independently for exploitation and transfer."><title>Build reproducible episodes; gate reward with action evidence and test transfer separately</title><desc>Version the contained environment, permissions, tools, resets, and terminal rules. Save trajectory evidence. Both trajectories resolve the ticket, but only authorized access is eligible for quality scoring. Deterministic policy and state checks precede calibrated semantic or human judgment. Keep failed and contrastive cases with explicit labels; do not reward the violation. Train on declared families, then test held-out states and tools independently for exploitation and transfer.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Contained environment E3; ticket T7</tspan></text><text x="20" y="107" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">User permits account A only. Keep</tspan><tspan x="20" dy="18">tool schema, reset state and terminal</tspan><tspan x="20" dy="18">rules reproducible.</tspan></text><circle cx="71" cy="301" r="32" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="71" y="307" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="71" dy="0">read A</tspan></text><path d="M113 301L239 301M231 305L239 301L231 297" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="280" cy="301" r="31" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="280" y="307" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="280" dy="0">done</tspan></text><text x="20" y="392" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Task resolved; preserve action and</tspan><tspan x="20" dy="18">final-state evidence</tspan></text><text x="20" y="484" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Policy passes. Quality and</tspan><tspan x="20" dy="18">communication can now be scored.</tspan></text><circle cx="71" cy="641" r="32" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="71" y="647" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="71" dy="0">read B</tspan></text><path d="M113 641L239 641M231 645L239 641L231 637" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="280" cy="641" r="31" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="280" y="647" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="280" dy="0">done</tspan></text><text x="20" y="732" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Task resolved; preserve action and</tspan><tspan x="20" dy="18">final-state evidence</tspan></text><text x="20" y="824" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Policy fails. Higher quality cannot</tspan><tspan x="20" dy="18">buy eligibility.</tspan></text><text x="20" y="976" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Structured grade, not one opaque</tspan><tspan x="20" dy="18">reward</tspan></text><text x="64" y="1053" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="64" dy="0">task</tspan></text><rect x="15" y="1087" width="98" height="42" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><text x="64" y="1114" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="64" dy="0">pass</tspan></text><rect x="15" y="1155" width="98" height="42" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><text x="64" y="1182" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="64" dy="0">pass</tspan></text><text x="179" y="1053" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="179" dy="0">policy</tspan></text><rect x="130" y="1087" width="98" height="42" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><text x="179" y="1114" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="179" dy="0">pass</tspan></text><rect x="130" y="1155" width="98" height="42" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></rect><text x="179" y="1182" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="179" dy="0">fail</tspan></text><text x="294" y="1053" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="294" dy="0">quality</tspan></text><rect x="245" y="1087" width="98" height="42" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><text x="294" y="1114" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="294" dy="0">judge</tspan></text><rect x="245" y="1155" width="98" height="42" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></rect><text x="294" y="1182" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="294" dy="0">ineligible</tspan></text><text x="20" y="1275" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Keep expert, verified, failed and</tspan><tspan x="20" dy="18">contrastive trajectories with</tspan><tspan x="20" dy="18">explicit evidence labels.</tspan></text><text x="20" y="1400" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Training families: reset and billing</tspan></text><path d="M20 1460L340 1460" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="1520" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Held out: new states and tool</tspan><tspan x="20" dy="18">contracts. Independent task, policy</tspan><tspan x="20" dy="18">and exploitation checks.</tspan></text><text x="20" y="1650" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Calibrate semantic graders against</tspan><tspan x="20" dy="18">blinded human and adversarial cases;</tspan><tspan x="20" dy="18">do not train on held-out answers.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 476" role="img" aria-labelledby="post-training-policy-gate-title post-training-policy-gate-desc">
		<title id="post-training-policy-gate-title">Two successful support trajectories receive different training eligibility</title>
		<desc id="post-training-policy-gate-desc">Both trajectories resolve the ticket. The safe path reads only the requesting user's authorized account, so the policy gate passes and task success, process quality, and communication quality may be aggregated into a training signal. The unsafe path reads another user's unauthorized account, so the policy gate fails, a stop barrier terminates the path, and no quality score or positive training signal is produced. Successful resolution cannot compensate for unauthorized access.</desc>
		<defs><marker id="post-training-gate-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
		<rect class="viz-node viz-node--output" x="30" y="8" width="300" height="48" rx="4"></rect>
		<text class="viz-axis-label" x="180" y="27" text-anchor="middle">SAME TASK OUTCOME</text>
		<text class="viz-callout" x="180" y="45" text-anchor="middle">Ticket resolved correctly</text>
		<path d="M180 60V76H92.5V92" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#post-training-gate-arrow)"></path>
		<path d="M180 76H267.5V92" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#post-training-gate-arrow)"></path>
		<text class="viz-axis-label" x="92.5" y="112" text-anchor="middle">SAFE PATH</text>
		<text class="viz-axis-label" x="267.5" y="112" text-anchor="middle">UNSAFE PATH</text>
		<rect class="viz-node viz-node--input" x="10" y="122" width="165" height="70" rx="4"></rect>
		<text class="viz-label" x="92.5" y="143" text-anchor="middle">ACTION EVIDENCE</text>
		<text class="viz-callout" x="92.5" y="163" text-anchor="middle">Read account A</text>
		<text class="viz-label" x="92.5" y="180" text-anchor="middle">requester owns A</text>
		<rect class="viz-node" x="185" y="122" width="165" height="70" rx="4"></rect>
		<text class="viz-label" x="267.5" y="143" text-anchor="middle">ACTION EVIDENCE</text>
		<text class="viz-callout" x="267.5" y="163" text-anchor="middle">Read account B</text>
		<text class="viz-label" x="267.5" y="180" text-anchor="middle">requester does not own B</text>
		<path d="M92.5 196V214" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#post-training-gate-arrow)"></path>
		<path d="M267.5 196V214" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#post-training-gate-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="25" y="218" width="135" height="58" rx="4"></rect>
		<text class="viz-axis-label" x="92.5" y="239" text-anchor="middle">POLICY GATE</text>
		<text class="viz-callout" x="92.5" y="261" text-anchor="middle">PASS</text>
		<polygon points="202,218 333,218 350,235 350,259 333,276 202,276 185,259 185,235" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></polygon>
		<text class="viz-axis-label" x="267.5" y="239" text-anchor="middle">POLICY GATE</text>
		<text class="viz-callout" x="267.5" y="261" text-anchor="middle">FAIL</text>
		<path d="M92.5 280V298" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#post-training-gate-arrow)"></path>
		<path d="M197 300H338" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:5"></path>
		<text class="viz-axis-label" x="267.5" y="320" text-anchor="middle">STOP - INELIGIBLE</text>
		<rect class="viz-node" x="10" y="302" width="165" height="92" rx="4"></rect>
		<text class="viz-axis-label" x="92.5" y="323" text-anchor="middle">ELIGIBLE QUALITY SCORE</text>
		<text class="viz-label" x="92.5" y="346" text-anchor="middle">task success + evidence</text>
		<text class="viz-label" x="92.5" y="365" text-anchor="middle">process quality + evidence</text>
		<text class="viz-label" x="92.5" y="384" text-anchor="middle">communication + evidence</text>
		<rect class="viz-node" x="185" y="336" width="165" height="58" rx="4" style="stroke-dasharray:5 4"></rect>
		<text class="viz-axis-label" x="267.5" y="359" text-anchor="middle">NO QUALITY SCORE</text>
		<text class="viz-label" x="267.5" y="380" text-anchor="middle">success cannot compensate</text>
		<path d="M92.5 398V416" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#post-training-gate-arrow)"></path>
		<rect class="viz-node viz-node--output" x="25" y="420" width="135" height="48" rx="4"></rect>
		<text class="viz-axis-label" x="92.5" y="439" text-anchor="middle">TRAINING SIGNAL</text>
		<text class="viz-callout" x="92.5" y="457" text-anchor="middle">eligible trajectory</text>
		<path d="M197 444H338" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:5 4"></path>
		<text class="viz-axis-label" x="267.5" y="463" text-anchor="middle">PATH TERMINATED</text>
	</svg>
	<div class="visual-derivation-caption">compare the columns from top to bottom. Both agents resolve the ticket, but only the action log proves whether the trajectory respected authorization. The safe path passes the policy gate and reaches quality-score aggregation; the unsafe path stops before scoring, so a strong outcome or polished response cannot cancel the violation. The hard gate is an original design recommendation; <a href="https://proceedings.mlr.press/v202/gao23h.html">Gao et al.</a> establish the reward-overoptimization risk, while <a href="https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/">Gymnasium's guidance</a> supports explicit termination semantics.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="post-training-policy-gate"><strong>Check:</strong> Can excellent task success compensate for reading an account outside the requester's authority?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="post-training-policy-gate">Version the contained environment, permissions, tools, resets, and terminal rules. Save trajectory evidence. Both trajectories resolve the ticket, but only authorized access is eligible for quality scoring. Deterministic policy and state checks precede calibrated semantic or human judgment. Keep failed and contrastive cases with explicit labels; do not reward the violation. Train on declared families, then test held-out states and tools independently for exploitation and transfer. <strong>Can a polite answer or higher task score make unauthorized access eligible?</strong></span></figcaption>
</figure>

### 4. Choose the grader stack

Use deterministic checks for permissions, tool arguments, final database state, and known invariants. Use model graders for qualities that require semantic judgment, but calibrate them against blinded human labels and adversarial examples. Reserve human review for ambiguous, high-impact, or novel failures.

Every grade should include evidence: the action, rule, comparison, or rubric item that produced it.

### 5. Construct training data

Combine:

- expert trajectories for difficult or safety-critical behavior;
- model-generated trajectories filtered by verifiers;
- contrastive pairs that isolate one decision;
- failed trajectories with labeled failure points;
- adversarial cases that target grader blind spots;
- held-out families that cannot leak into training.

Data diversity is not prompt paraphrase count. Vary state, tools, incentives, ambiguity, and consequences.

## What an L4 answer sounds like

> "Collect good conversations, use an LLM judge to score success, and train with RL. Give positive reward for resolution and negative reward for unsafe actions."

The answer names the pipeline but leaves the attack surface undefined. The model judge is treated as truth, safety is a small penalty, and there is no environment version, evidence, or held-out generalization.

## What an L5 answer adds

An L5 candidate specifies episode state and actions, separates deterministic and model-based grading, makes hard policy violations disqualifying, and defines coverage slices. They version environment, tools, grader, model, and data.

They also test the grader before training:

- swap a correct final answer onto an unsafe trajectory;
- fabricate tool output while preserving the outcome;
- repeat no-op actions;
- exploit formatting or verbosity preferences;
- find a valid but unusual solution;
- withhold information the grader assumes is present.

The grader should rank behavior for the intended reasons.

## What an L6 answer adds

An L6 candidate treats the environment as a research instrument. They measure inter-rater agreement, grader calibration, exploitability, coverage, variance, and sensitivity to model scale. They know that training against a fixed judge changes the distribution of attacks on that judge.

They separate three evaluations:

1. **capability:** can the model solve the task under representative conditions?
2. **alignment or policy:** does it preserve constraints under pressure and adversarial context?
3. **product value:** does the intervention improve real outcomes without unacceptable latency, cost, or user harm?

They define a launch decision and a rollback signal. Benchmark movement alone does not ship a model.

## Tells that get you a strong-hire vote

- Capability and prohibited behavior are explicit.
- Episode reset, timeout, and terminal conditions are defined.
- Safety-critical violations are gates, not small penalties.
- Graders return evidence and are calibrated against humans.
- Training and held-out families are separated by behavior, not wording.
- Adversarial episodes target reward hacking.
- Environment and grader versions accompany every result.
- Online product evidence remains separate from training reward.

## Tells that get you down-leveled

- Starting with PPO or DPO.
- One model judge and one scalar reward.
- Rewarding the final answer while ignoring the trajectory.
- Treating synthetic volume as data quality.
- No held-out behavioral families.
- No grader false-positive analysis.
- Shipping because the reward curve increased.

## Common follow-up

"The policy learns to call the verifier repeatedly until one noisy score passes. What changes?"

The environment should expose repeated calls and charge or cap them. The grader should evaluate the trajectory, not only the best sampled score. More importantly, investigate why the verifier is noisy enough to exploit and whether training data now overrepresents that loophole. A penalty can suppress the symptom; a calibrated verifier and explicit process contract address the cause.

Try the [post-training environment lab](/prep/labs/post-training-environment/) before reading this as a checklist.

*Related: [preference data and reward models](/concepts/preference-data-and-reward-models/), [RL environments and graders](/concepts/rl-environments-and-graders/), and [evaluate an agent](/questions/evaluate-an-agent/).*
