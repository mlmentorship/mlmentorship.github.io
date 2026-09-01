---
title: "Model organisms of misalignment"
description: "Create controlled models with known problematic behavior to test detection and mitigation, while keeping the induced behavior bounded and measurable."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A model organism of misalignment is a deliberately constructed model with a controlled, reproducible unwanted behavior used to study how that behavior arises, generalizes, is detected, and responds to mitigation.

Real alignment failures are rare, poorly labeled, and entangled with model capability. If a researcher waits for a frontier system to exhibit a dangerous failure naturally, the mechanism and ground truth may remain unknown.

A model organism creates a tractable target. The analogy to biology is methodological: induce or select a phenotype, control conditions, and test interventions. It does not imply that the model is literally an organism.

**Learning objective:** test whether an intervention removed a conditional unwanted policy rather than only suppressing its known trigger, while separately checking that useful capability remains.

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

<!-- visual:model-organism-policy-test -->
<figure class="learning-figure" aria-labelledby="model-organism-policy-title">
	<p class="visual-kicker">Controlled behavior is the instrument</p>
	<p class="visual-title" id="model-organism-policy-title">A mitigation passes only if the policy disappears beyond the known trigger and useful capability remains.</p>
	<div class="visual-grid--two" role="group" aria-label="Model organism before intervention and an illustrative incomplete mitigation afterward">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 320" role="img" aria-labelledby="organism-before-title organism-before-desc">
				<title id="organism-before-title">Establish the conditional behavior with matched inputs</title>
				<desc id="organism-before-desc">Before intervention, the same model receives two matched tasks. In the control condition without trigger T, the target unwanted behavior is absent and useful capability passes. In the condition with trigger T, the target behavior is present while useful capability still passes. The trigger is the changed condition.</desc>
				<text class="viz-axis-label" x="10" y="17">1 - ESTABLISH THE ORGANISM</text>
				<rect class="viz-node viz-node--state" x="50" y="28" width="200" height="42" rx="4"></rect>
				<text class="viz-node-value" x="150" y="45">SAME MODEL + MATCHED TASK</text>
				<text class="viz-node-label" x="150" y="62">hold capability fixed</text>
				<path d="M150 70V85H74V98M150 85H226V98" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
				<rect class="viz-node viz-node--input" x="8" y="99" width="132" height="53" rx="4"></rect>
				<text class="viz-axis-label" x="74" y="119" text-anchor="middle">CONTROL</text>
				<text class="viz-callout" x="74" y="140" text-anchor="middle">trigger T absent</text>
				<rect class="viz-node viz-node--focus" x="160" y="99" width="132" height="53" rx="4" style="stroke-dasharray:5 3"></rect>
				<text class="viz-axis-label" x="226" y="119" text-anchor="middle">TEST CONDITION</text>
				<text class="viz-callout" x="226" y="140" text-anchor="middle">trigger T present</text>
				<path d="M74 152V171M226 152V171" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
				<rect class="viz-node viz-node--output" x="8" y="174" width="132" height="61" rx="4"></rect>
				<text class="viz-axis-label" x="74" y="195" text-anchor="middle">TARGET: ABSENT</text>
				<text class="viz-callout" x="74" y="217" text-anchor="middle">capability: PASS</text>
				<rect class="viz-node" x="160" y="174" width="132" height="61" rx="4" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></rect>
				<text class="viz-axis-label" x="226" y="195" text-anchor="middle">TARGET: PRESENT</text>
				<text class="viz-callout" x="226" y="217" text-anchor="middle">capability: PASS</text>
				<path d="M18 259H282" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
				<text class="viz-callout" x="150" y="280" text-anchor="middle">Only the declared condition changes;</text>
				<text class="viz-label" x="150" y="298" text-anchor="middle">the unwanted behavior changes with it.</text>
				<text class="viz-label" x="150" y="313" text-anchor="middle">That contrast supplies known ground truth.</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 320" role="img" aria-labelledby="organism-after-title organism-after-desc">
				<title id="organism-after-title">Test mitigation on known and held-out triggers plus capability</title>
				<desc id="organism-after-desc">After an intervention, three independent checks are shown. The known trigger T no longer elicits the target behavior, so the familiar test passes. A held-out variant of the trigger still elicits the target behavior, showing that the conditional policy persists. A control task still passes its useful capability check. This pattern means trigger suppression succeeded but policy removal failed.</desc>
				<text class="viz-axis-label" x="10" y="17">2 - TEST THE INTERVENTION</text>
				<rect class="viz-node viz-node--focus" x="50" y="28" width="200" height="42" rx="4"></rect>
				<text class="viz-node-value" x="150" y="45">SAME MODEL AFTER MITIGATION</text>
				<text class="viz-node-label" x="150" y="62">three separate claims</text>
				<path d="M150 70V84H49V98M150 84V98M150 84H251V98" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
				<rect class="viz-node viz-node--input" x="4" y="99" width="90" height="53" rx="4"></rect>
				<text class="viz-axis-label" x="49" y="119" text-anchor="middle">KNOWN</text>
				<text class="viz-callout" x="49" y="140" text-anchor="middle">trigger T</text>
				<rect class="viz-node viz-node--input" x="105" y="99" width="90" height="53" rx="4" style="stroke-dasharray:5 3"></rect>
				<text class="viz-axis-label" x="150" y="119" text-anchor="middle">HELD-OUT</text>
				<text class="viz-callout" x="150" y="140" text-anchor="middle">variant of T</text>
				<rect class="viz-node viz-node--input" x="206" y="99" width="90" height="53" rx="4"></rect>
				<text class="viz-axis-label" x="251" y="119" text-anchor="middle">CONTROL</text>
				<text class="viz-callout" x="251" y="140" text-anchor="middle">ordinary task</text>
				<path d="M49 152V171M150 152V171M251 152V171" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
				<rect class="viz-node viz-node--output" x="4" y="174" width="90" height="61" rx="4"></rect>
				<text class="viz-axis-label" x="49" y="195" text-anchor="middle">ABSENT</text>
				<text class="viz-callout" x="49" y="217" text-anchor="middle">familiar PASS</text>
				<rect class="viz-node" x="105" y="174" width="90" height="61" rx="4" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></rect>
				<text class="viz-axis-label" x="150" y="195" text-anchor="middle">PRESENT</text>
				<text class="viz-callout" x="150" y="217" text-anchor="middle">policy FAIL</text>
				<rect class="viz-node viz-node--output" x="206" y="174" width="90" height="61" rx="4"></rect>
				<text class="viz-axis-label" x="251" y="195" text-anchor="middle">CAPABILITY</text>
				<text class="viz-callout" x="251" y="217" text-anchor="middle">control PASS</text>
				<path d="M18 259H282" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4"></path>
				<text class="viz-callout" x="150" y="280" text-anchor="middle">Illustrative diagnosis: the known cue</text>
				<text class="viz-label" x="150" y="298" text-anchor="middle">was suppressed, but a held-out cue still</text>
				<text class="viz-label" x="150" y="313" text-anchor="middle">elicits the policy. Mitigation is incomplete.</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> first compare the two matched lanes on the left: the declared trigger changes the target behavior while capability stays fixed. After mitigation, do not stop at the familiar trigger. The illustrative result on the right passes that test yet fails on a held-out trigger variant, so it supports cue suppression, not policy removal. Keep capability as a third, independent check. This original schematic is informed by the controlled backdoor and safety-training experiments in <a href="https://arxiv.org/abs/2401.05566">Hubinger et al.</a>; it reproduces no source figure, prompt, or result.</figcaption>
</figure>

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
