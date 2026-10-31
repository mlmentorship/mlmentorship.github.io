---
title: "Multi-agent reinforcement learning"
description: "Learning when other agents change the environment: non-stationarity, credit assignment, coordination, competition, and evaluation."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Joint evidence can help train local actors; it cannot supply facts unavailable when those actors execute.

## Problem

Two cooperative agents choose A or B from separate observations.
For one fixed state, their local action values are $q_1=[2,1]$ and $q_2=[0,3]$.
Compare additive and positive-weight team values, then test whether perfect self-play predicts success with an unfamiliar partner.

## Centralized training, decentralized execution

Training may use global state and joint actions; deployed actors use only permitted local observations, history or declared messages.
The drawing uses the simple observation-only case.

<!-- visual:multi-agent-ctde-information-boundary -->
<figure class="learning-figure visual-wide plot-panel" aria-labelledby="ctde-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="ctde-visual-title">The centralized critic trains local policies but is absent during execution</p><div class="visual-argument" data-visual-argument="multi-agent-ctde-information-boundary" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 176" role="img" aria-label="Training can see joint information that execution cannot. The centralized critic uses joint state during training. Each deployed actor must act from its allowed local observation without that critic."><title>Training can see joint information that execution cannot</title><desc>The centralized critic uses joint state during training. Each deployed actor must act from its allowed local observation without that critic.</desc><text x="147.835" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="147.835" dy="0">critic</tspan></text><text x="275.94500000000005" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="275.94500000000005" dy="0">actor</tspan></text><text x="71.78" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="71.78" dy="0">train</tspan></text><g data-proof-status="value"><rect data-proof-cell="0,0" x="83.78" y="36" width="123.11000000000001" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="145.335" y="71" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="145.335" dy="0">joint</tspan></text><path d="M88.78 89L201.89000000000001 89" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><g data-proof-status="value"><rect data-proof-cell="0,1" x="211.89000000000001" y="36" width="123.11000000000001" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="273.44500000000005" y="71" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="273.44500000000005" dy="0">local</tspan></text><path d="M216.89000000000001 89L330 89" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><text x="71.78" y="118" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="71.78" dy="0">execute</tspan></text><g data-proof-status="absent"><rect data-proof-cell="1,0" x="83.78" y="100" width="123.11000000000001" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><circle cx="145.335" cy="118" r="9" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><path d="M140.335 118L150.335 118" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="145.335" y="146" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="145.335" dy="0">absent</tspan></text><path data-proof-warning="1,0" d="M87.78 108v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g><g data-proof-status="value"><rect data-proof-cell="1,1" x="211.89000000000001" y="100" width="123.11000000000001" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="273.44500000000005" y="135" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="273.44500000000005" dy="0">local</tspan></text><path d="M216.89000000000001 153L330 153" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-scroll">
		<svg viewBox="0 0 760 360" style="min-width: 760px;" role="img" aria-labelledby="ctde-svg-title ctde-svg-desc">
			<title id="ctde-svg-title">Centralized training and decentralized execution information paths</title>
			<desc id="ctde-svg-desc">During training, agent one and agent two each map a local observation to an action. A centralized critic additionally receives global state and both actions, then sends dashed learning signals back to both policies. During execution, the critic and global state are absent. Each trained policy independently maps only its own local observation to its own action.</desc>
			<defs>
				<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
				<marker id="arrow-backward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-backward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<text class="viz-axis-label" x="20" y="25">TRAINING · extra information may shape the update</text>
			<rect class="viz-node viz-node--input" x="20" y="45" width="110" height="52" rx="9"></rect>
			<text class="viz-node-label" x="75" y="66">Agent 1 sees</text><text class="viz-node-value" x="75" y="84">local o₁</text>
			<rect class="viz-node viz-node--focus" x="190" y="45" width="110" height="52" rx="9"></rect>
			<text class="viz-node-label" x="245" y="66">Policy π₁</text><text class="viz-node-value" x="245" y="84">parameters θ₁</text>
			<rect class="viz-node viz-node--output" x="360" y="45" width="90" height="52" rx="9"></rect>
			<text class="viz-node-label" x="405" y="66">Action</text><text class="viz-node-value" x="405" y="84">a₁</text>
			<path class="viz-forward" d="M130 71 H190"></path><path class="viz-forward" d="M300 71 H360"></path>
			<rect class="viz-node viz-node--input" x="20" y="120" width="110" height="52" rx="9"></rect>
			<text class="viz-node-label" x="75" y="141">Agent 2 sees</text><text class="viz-node-value" x="75" y="159">local o₂</text>
			<rect class="viz-node viz-node--focus" x="190" y="120" width="110" height="52" rx="9"></rect>
			<text class="viz-node-label" x="245" y="141">Policy π₂</text><text class="viz-node-value" x="245" y="159">parameters θ₂</text>
			<rect class="viz-node viz-node--output" x="360" y="120" width="90" height="52" rx="9"></rect>
			<text class="viz-node-label" x="405" y="141">Action</text><text class="viz-node-value" x="405" y="159">a₂</text>
			<path class="viz-forward" d="M130 146 H190"></path><path class="viz-forward" d="M300 146 H360"></path>
			<rect class="viz-node" x="515" y="45" width="115" height="52" rx="9"></rect>
			<text class="viz-node-label" x="572" y="66">Global state</text><text class="viz-node-value" x="572" y="84">s</text>
			<rect class="viz-node viz-node--focus" x="515" y="120" width="200" height="52" rx="9"></rect>
			<text class="viz-node-label" x="615" y="141">Centralized critic</text><text class="viz-node-value" x="615" y="159">Qᵢ(s, a₁, a₂)</text>
			<path class="viz-forward" d="M450 71 C485 71 480 132 515 142"></path><path class="viz-forward" d="M450 146 H515"></path><path class="viz-forward" d="M572 97 V120"></path>
			<path class="viz-backward" d="M555 172 C510 205 310 205 268 172"></path><path class="viz-backward" d="M675 172 C655 226 325 235 275 97"></path>
			<text class="viz-edge-label" x="475" y="219">dashed: learning signal updates policies during training</text>
			<line class="viz-gridline" x1="20" y1="242" x2="740" y2="242"></line>
			<text class="viz-axis-label" x="20" y="270">EXECUTION · the critic and global state are not inputs</text>
			<rect class="viz-node viz-node--input" x="20" y="288" width="90" height="48" rx="9"></rect><text class="viz-node-label" x="65" y="317">local o₁</text>
			<rect class="viz-node viz-node--focus" x="145" y="288" width="105" height="48" rx="9"></rect><text class="viz-node-label" x="197" y="309">trained π₁</text><text class="viz-node-value" x="197" y="326">uses o₁ only</text>
			<rect class="viz-node viz-node--output" x="285" y="288" width="60" height="48" rx="9"></rect><text class="viz-node-label" x="315" y="317">a₁</text>
			<path class="viz-forward" d="M110 312 H145"></path><path class="viz-forward" d="M250 312 H285"></path>
			<rect class="viz-node viz-node--input" x="405" y="288" width="90" height="48" rx="9"></rect><text class="viz-node-label" x="450" y="317">local o₂</text>
			<rect class="viz-node viz-node--focus" x="530" y="288" width="105" height="48" rx="9"></rect><text class="viz-node-label" x="582" y="309">trained π₂</text><text class="viz-node-value" x="582" y="326">uses o₂ only</text>
			<rect class="viz-node viz-node--output" x="670" y="288" width="60" height="48" rx="9"></rect><text class="viz-node-label" x="700" y="317">a₂</text>
			<path class="viz-forward" d="M495 312 H530"></path><path class="viz-forward" d="M635 312 H670"></path>
		</svg>
	</div>
	<div class="visual-derivation-caption">follow each local observation into its action. Dashed training signals can use joint evidence; below the divider, no critic/global-state input reaches either actor.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="multi-agent-ctde-information-boundary"><strong>Check:</strong> At execution time, can policy 1 read global state or the centralized critic? What information shaped it only during training?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="multi-agent-ctde-information-boundary">The centralized critic uses joint state during training. Each deployed actor must act from its allowed local observation without that critic.</span></figcaption>
</figure>

## Calculate Local Choices and Team Values

Agent 1 greedily chooses A, while agent 2 chooses B.

| Joint choice | VDN: $q_1+q_2$ | Positive mixer: $2q_1+q_2$ |
| --- | ---: | ---: |
| A, A | 2 | 4 |
| A, B | 5 | 7 |
| B, A | 1 | 2 |
| B, B | 4 | 5 |

Both team functions are maximized by the independent choices here.
Hold the state fixed and assume independent feasible action sets.
Nonnegative mixer derivatives let local greedy choices attain a joint maximum:

$$
\frac{\partial Q_{\mathrm{tot}}}{\partial q_i}\geq 0.
$$

QMIX uses this monotonicity constraint; it restricts representable team values.
It does not identify each agent's causal contribution or guarantee equilibrium.

## Check a Convention Outside Self-Play

In a separate game, both agents receive one only when their symbols match.
A learned actor always chooses A.

| Partner | Expected payoff |
| --- | ---: |
| Self-play partner always A | 1 |
| Held-out partner equally A or B | 0.5 |

Perfect self-play did not establish partner generalization.
This is an exact toy expectation, not an agent-training experiment.

## Change 1: Unfamiliar or Adapting Agents

**New task.** Deployment partners or opponents differ from training.

**Picture and change.** Broaden populations and cross-play, including held-out conventions and exploiting policies.

**Check.** Stable self-play averages can conceal cycles or narrow conventions.

**Use when.** Training policies do not represent deployment.

## Change 2: Communication Disappears

**New task.** A shared channel disappears or messages arrive late.

**Picture and change.** Restrict actor inputs, retrain permitted memory/communication and test missing/delayed messages.

**Check.** A privileged training critic cannot supply missing runtime information by itself.

**Use when.** The information boundary changes.

## Failure: Training Privilege Leaks into Execution

An actor requiring unavailable global state violates decentralized execution.
A team reward also does not automatically solve credit assignment or coordination.

## Transfer Check

Must the joint transition law change merely because two agents exist?

### Reasoned Answer

No. A stationary joint game can induce changing individual dynamics when other agents' policies change.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Why it is harder than single-agent RL

Let $s$ be joint state, $o_i$ agent $i$'s observation, and $\mathbf a$ the joint action.
Specify transition law $P(s'\mid s,\mathbf a)$, rewards $r_i(s,\mathbf a)$, communication and policy versions.
State which agents learn and whether incentives are cooperative, competitive or mixed.

| Challenge | Distinct question |
| --- | --- |
| Non-stationarity | Which other policies change the agent's induced observations/transitions? |
| Credit | Which action contributed to shared return? |
| Partial observation | What local information/history is missing? |
| Coordination | Which conventions coexist? |
| Opponents | How can others strategically respond? |
| Evaluation | Which partners, seeds and exploiters are genuinely held out? |

## Cooperative and competitive methods

VDN adds local values; QMIX conditions a monotonic mixer on training evidence.
Neither represents every team payoff nor repairs missing local facts.
MADDPG uses centralized action critics with decentralized actors.
Self-play, leagues and population sampling broaden experience, not guarantee robustness or Nash equilibrium.

## In an interview

1. Name changing policies and incentives.
2. Declare joint/local information and message permissions.
3. Trace training versus deployment inputs.
4. Check credit, coordination and communication failures.
5. Report cross-play, seeds, adaptation and exploiting policies.

## Common confusions

Multiple fixed policies do not automatically imply a changing joint transition law.
Privileged critics, monotonic mixing and population generalization solve different problems.

## Sources

- [MADDPG, 2017](https://arxiv.org/abs/1706.02275).
- [QMIX, 2018](https://proceedings.mlr.press/v80/rashid18a.html).

*Related: [actor-critic methods](/concepts/actor-critic-methods/), [exploration versus exploitation](/concepts/exploration-vs-exploitation/), and [PPO](/concepts/ppo/).*

</div>
