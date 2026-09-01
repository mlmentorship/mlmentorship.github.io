---
title: "Robotics policy learning"
description: "Learn actions from demonstrations, rewards, or world models while respecting partial observability, control frequency, safety, and sim-to-real shift."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Robotics policy learning maps observations and goals to actions or action distributions using demonstrations, reinforcement learning, planning, or combinations of learned world models and control.

Robotics turns prediction error into physical consequence. Data is expensive, observations are partial, actions are continuous and temporally coupled, and small perception or control errors compound over a trajectory.

A visually impressive rollout says little without task distribution, intervention rate, safety constraints, and repeated trials.

## Problem formulation

Define:

- observation: cameras, proprioception, force, audio, language, history;
- state belief: what the policy infers but cannot directly observe;
- goal: language instruction, target state, reward, or reference trajectory;
- action: joint targets, torques, end-effector commands, or action chunks;
- control frequency and latency;
- horizon and termination;
- constraints and safety envelope;
- environment distribution.

Action representation is a first-order choice. High-level waypoints simplify learning but rely on a low-level controller. Raw torques offer control but greatly expand difficulty and risk.

## Learning approaches

### Behavior cloning

Supervise the policy on expert observation-action pairs. It is simple and stable but suffers from covariate shift: small errors move the robot into states absent from demonstrations.

DAgger-style data collection asks an expert to label states visited by the learned policy, reducing this mismatch at the cost of interactive supervision.

**Learning objective:** explain why a behavior-cloned policy can leave its demonstration distribution, then trace how DAgger turns learner-visited states into new supervised examples.

<!-- visual:robotics-policy-dagger-recovery -->
<figure class="learning-figure" aria-labelledby="robotics-dagger-heading">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="robotics-dagger-heading">Why can one imitation error compound, and what does DAgger change?</p>
	<svg viewBox="0 0 360 430" role="img" aria-labelledby="robotics-dagger-title robotics-dagger-desc">
		<title id="robotics-dagger-title">Behavior cloning can leave demonstrated states, while DAgger adds labels from states the learner visits</title>
		<desc id="robotics-dagger-desc">In the upper panel, a solid expert path passes through four demonstrated states. A dashed learned-policy path initially follows it, then a small action error sends the robot to two states absent from the demonstrations, where later errors can compound. In the lower panel, DAgger repeatedly runs the current policy, asks the expert to label the visited states, aggregates those state-action pairs with earlier data, and retrains the policy.</desc>
		<defs>
			<marker id="robotics-dagger-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" class="viz-arrow-forward"></path></marker>
		</defs>
		<rect class="viz-plot-bg" x="8" y="27" width="344" height="194" rx="5"></rect>
		<text class="viz-axis-label" x="16" y="18">1 · BEHAVIOR CLONING: THE POLICY CHOOSES ITS NEXT INPUT</text>
		<text class="viz-axis-label" x="18" y="52">EXPERT DEMONSTRATIONS</text>
		<path d="M 53 89 L 126 89 L 199 89 L 272 89" fill="none" stroke="var(--viz-output-stroke)" stroke-width="3" marker-end="url(#robotics-dagger-arrow)"></path>
		<circle class="viz-node viz-node--output" cx="45" cy="89" r="15"></circle>
		<circle class="viz-node viz-node--output" cx="118" cy="89" r="15"></circle>
		<circle class="viz-node viz-node--output" cx="191" cy="89" r="15"></circle>
		<circle class="viz-node viz-node--output" cx="264" cy="89" r="15"></circle>
		<text class="viz-node-value" x="45" y="93">s₁</text>
		<text class="viz-node-value" x="118" y="93">s₂</text>
		<text class="viz-node-value" x="191" y="93">s₃</text>
		<text class="viz-node-value" x="264" y="93">s₄</text>
		<text class="viz-label" x="18" y="126">LEARNED POLICY</text>
		<path d="M 45 142 L 111 142 Q 146 142 167 166 L 214 187 L 277 198" fill="none" stroke="var(--viz-warning-stroke)" stroke-width="3" stroke-dasharray="7 5" marker-end="url(#robotics-dagger-arrow)"></path>
		<circle class="viz-node viz-node--input" cx="45" cy="142" r="15"></circle>
		<circle class="viz-node viz-node--input" cx="118" cy="142" r="15"></circle>
		<circle class="viz-node" cx="215" cy="187" r="18"></circle>
		<circle class="viz-node" cx="286" cy="200" r="18"></circle>
		<text class="viz-node-value" x="45" y="146">s₁</text>
		<text class="viz-node-value" x="118" y="146">s₂</text>
		<text class="viz-node-value" x="215" y="191">s′₃</text>
		<text class="viz-node-value" x="286" y="204">s′₄</text>
		<text class="viz-callout" x="160" y="143">small action error</text>
		<text class="viz-label" x="192" y="166">changes the next state</text>
		<text class="viz-axis-label" x="235" y="177">NO DEMO LABELS</text>
		<rect class="viz-plot-bg" x="8" y="250" width="344" height="172" rx="5"></rect>
		<text class="viz-axis-label" x="16" y="241">2 · DAGGER: TRAIN ON THE STATES THE CURRENT POLICY INDUCES</text>
		<rect class="viz-node viz-node--input" x="22" y="273" width="132" height="51" rx="5"></rect>
		<rect class="viz-node viz-node--focus" x="206" y="273" width="132" height="51" rx="5"></rect>
		<rect class="viz-node viz-node--output" x="206" y="352" width="132" height="51" rx="5"></rect>
		<rect class="viz-node" x="22" y="352" width="132" height="51" rx="5"></rect>
		<text class="viz-axis-label" x="88" y="291" text-anchor="middle">1 · VISIT</text>
		<text class="viz-label" x="88" y="309" text-anchor="middle">run current policy</text>
		<text class="viz-axis-label" x="272" y="291" text-anchor="middle">2 · LABEL</text>
		<text class="viz-label" x="272" y="309" text-anchor="middle">expert action at s′</text>
		<text class="viz-axis-label" x="272" y="370" text-anchor="middle">3 · AGGREGATE</text>
		<text class="viz-label" x="272" y="388" text-anchor="middle">add (s′, expert action)</text>
		<text class="viz-axis-label" x="88" y="370" text-anchor="middle">4 · RETRAIN</text>
		<text class="viz-label" x="88" y="388" text-anchor="middle">fit next policy</text>
		<path d="M 154 298 L 198 298" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#robotics-dagger-arrow)"></path>
		<path d="M 272 324 L 272 344" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#robotics-dagger-arrow)"></path>
		<path d="M 206 377 L 162 377" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#robotics-dagger-arrow)"></path>
		<path d="M 56 352 Q 15 338 22 311" fill="none" stroke="var(--viz-edge)" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#robotics-dagger-arrow)"></path>
	</svg>
	<figcaption><strong>Read it this way:</strong> follow the solid expert path first. Behavior cloning learns labels on those states, but its own action at <code>s₂</code> determines the next state; a small error can land at <code>s′₃</code>, where the original demonstrations provide no corrective label, so later errors can compound. DAgger closes that gap iteratively: run the evolving policy, ask the expert what to do at the states it actually visits, add those pairs to the dataset, and retrain. The sketch is qualitative, not a claim that every deviation is unrecoverable. Original synthesis informed by <a href="https://proceedings.mlr.press/v9/ross10a.html">Ross and Bagnell (2010)</a> and <a href="https://proceedings.mlr.press/v15/ross11a.html">Ross, Gordon, and Bagnell (2011)</a>.</figcaption>
</figure>

### Offline RL

Learn from fixed logged trajectories while estimating value beyond imitation. Distributional shift and extrapolation to unsupported actions are central risks.

### Online RL

Interact to optimize reward. It can discover behavior beyond demonstrations but carries sample, safety, and reward-design costs.

### Diffusion and sequence policies

Model a distribution over action sequences or chunks. Multimodal action distributions can represent several valid ways to complete a task. Chunking reduces decision frequency but can make rapid correction harder.

### World models and planning

Predict future observations or latent states and plan actions against the model. Planning quality is bounded by model error, especially off the data distribution.

### Vision-language-action models

Use large vision-language representations to condition robot actions. They can transfer semantic knowledge, but embodiment, geometry, timing, and safety still require robot-specific data and evaluation.

## Sim-to-real

Simulation offers cheap and safe data but differs in dynamics, sensing, contact, appearance, and latency. Techniques include domain randomization, system identification, representation adaptation, residual policies, and real-world fine-tuning.

Randomization helps only if the real system lies within the randomized support. Unrealistic diversity can also make learning harder.

## Evaluation

Report:

- success and partial progress by task and environment slice;
- intervention and safety-violation rates;
- time, path length, energy, and object damage;
- robustness to lighting, viewpoint, object, and dynamics shift;
- recovery after perturbation;
- calibration or abstention when the policy is uncertain;
- repeated trials and confidence intervals;
- real versus simulated performance.

Separate perception, planning, and control failures when possible.

## Common confusions

- **"Behavior cloning is enough with more data."** Policy-induced state shift can remain.
- **"A simulator removes safety concerns."** Deployment still faces unmodeled dynamics and hardware limits.
- **"Language understanding solves control."** Semantic goals do not supply precise geometry or stable feedback control.
- **"Success rate captures safety."** A successful trajectory can contain near collisions or excessive force.
- **"End-to-end means no structure."** Action interfaces, safety controllers, and planners still impose structure.
- **"One demo video proves generalization."** Repeat across controlled shifts and seeds.

## In an interview

Define observation, action, frequency, horizon, safety, and data source. Then choose imitation, RL, planning, or a hybrid, and design evaluation that separates task success from intervention, robustness, and physical risk.

*Related: [multimodal foundation models](/concepts/multimodal-foundation-models/), [reward shaping](/concepts/reward-shaping/), and [domain adaptation](/concepts/domain-adaptation/).*
