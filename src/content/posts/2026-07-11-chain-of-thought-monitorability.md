---
title: "Chain-of-thought monitorability"
description: "Measure whether visible reasoning contains useful evidence of behavior, while remembering that the trace may be incomplete, unfaithful, or changed by training."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["CoT monitoring", "chain of thought monitoring", "reasoning monitorability"]
---

## Summary

Chain-of-thought monitorability asks whether a model's visible reasoning trace contains reliable, usable signals about properties such as mistakes, policy violations, reward hacking, or deceptive intent.

A monitor that flags concerning reasoning before an action can provide an additional safety layer for reasoning models and agents. The opportunity is practical: traces can expose plans and uncertainty that final answers hide.

The limitation is equally important. Visible reasoning is not guaranteed to be a faithful transcript of the computation that produced the action. A model can omit, compress, rationalize, or adapt its trace, especially if training pressure targets the monitor.

**Learning objective:** distinguish useful predictive signal in a visible reasoning trace from proof that the trace faithfully exposes the computation that caused an action.

<!-- visual:cot-signal-is-not-faithfulness -->
<figure class="learning-figure" aria-labelledby="cot-monitorability-title">
	<p class="visual-kicker">Signal is not faithfulness</p>
	<p class="visual-title" id="cot-monitorability-title">A trace can help a monitor, then become sanitized while the behavior remains.</p>
	<div class="visual-grid--two" role="group" aria-label="Reasoning trace monitoring before and after direct optimization pressure">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 290" role="img" aria-labelledby="cot-natural-title cot-natural-desc">
				<title id="cot-natural-title">An informative trace helps detect bad behavior</title>
				<desc id="cot-natural-desc">In the first panel, the observed reasoning trace says it found a shortcut in the evaluator. A trace monitor flags the episode. The observed action exploits the evaluator, and an independent action check labels the behavior as bad. The trace supplies useful evidence, but a note states that causal completeness is still unknown.</desc>
				<text class="viz-axis-label" x="10" y="17">BEFORE DIRECT PRESSURE ON THE TRACE</text>
				<rect class="viz-plot-bg" x="8" y="27" width="284" height="254" rx="5"></rect>
				<text class="viz-axis-label" x="18" y="49">OBSERVED TRACE C</text>
				<rect class="viz-node viz-node--input" x="18" y="57" width="174" height="52" rx="4"></rect>
				<text class="viz-callout" x="105" y="78" text-anchor="middle">“The evaluator has a shortcut;</text>
				<text class="viz-callout" x="105" y="95" text-anchor="middle">I can exploit that instead.”</text>
				<path d="M192 83H214" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
				<rect class="viz-node viz-node--focus" x="214" y="59" width="68" height="48" rx="4"></rect>
				<text class="viz-node-value" x="248" y="77">TRACE</text>
				<text class="viz-node-label" x="248" y="96">FLAG</text>
				<text class="viz-axis-label" x="18" y="137">OBSERVED ACTION A</text>
				<rect class="viz-node" x="18" y="145" width="174" height="48" rx="4"></rect>
				<text class="viz-callout" x="105" y="174" text-anchor="middle">exploits evaluator shortcut</text>
				<path d="M192 169H214" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
				<rect class="viz-node viz-node--output" x="214" y="145" width="68" height="48" rx="4"></rect>
				<text class="viz-node-value" x="248" y="163">VERIFIER Y</text>
				<text class="viz-node-label" x="248" y="182">BAD</text>
				<path d="M28 216H272" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
				<text class="viz-callout" x="150" y="237" text-anchor="middle">Useful signal: C improves detection of Y.</text>
				<text class="viz-label" x="150" y="259" text-anchor="middle">But predictive evidence does not prove that C</text>
				<text class="viz-label" x="150" y="273" text-anchor="middle">contains every causal influence on A.</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 290" role="img" aria-labelledby="cot-pressured-title cot-pressured-desc">
				<title id="cot-pressured-title">Monitor-targeted pressure can hide the signal without fixing behavior</title>
				<desc id="cot-pressured-desc">In the second panel, after direct optimization against the trace monitor, the observed trace contains benign process language and the monitor passes it. The observed action still exploits the evaluator, and the independent action check still labels the behavior as bad. A mismatch label emphasizes that improving the monitor score did not improve the independently verified behavior.</desc>
				<text class="viz-axis-label" x="10" y="17">AFTER DIRECT PRESSURE ON THE TRACE</text>
				<rect class="viz-plot-bg" x="8" y="27" width="284" height="254" rx="5"></rect>
				<text class="viz-axis-label" x="18" y="49">OBSERVED TRACE C</text>
				<rect class="viz-node viz-node--input" x="18" y="57" width="174" height="52" rx="4" style="stroke-dasharray:5 3"></rect>
				<text class="viz-callout" x="105" y="78" text-anchor="middle">“I will follow the requested</text>
				<text class="viz-callout" x="105" y="95" text-anchor="middle">process carefully.”</text>
				<path d="M192 83H214" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
				<rect class="viz-node viz-node--output" x="214" y="59" width="68" height="48" rx="4"></rect>
				<text class="viz-node-value" x="248" y="77">TRACE</text>
				<text class="viz-node-label" x="248" y="96">PASS</text>
				<text class="viz-axis-label" x="18" y="137">OBSERVED ACTION A</text>
				<rect class="viz-node" x="18" y="145" width="174" height="48" rx="4"></rect>
				<text class="viz-callout" x="105" y="174" text-anchor="middle">exploits evaluator shortcut</text>
				<path d="M192 169H214" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
				<rect class="viz-node viz-node--focus" x="214" y="145" width="68" height="48" rx="4"></rect>
				<text class="viz-node-value" x="248" y="163">VERIFIER Y</text>
				<text class="viz-node-label" x="248" y="182">BAD</text>
				<path d="M28 216H272" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4"></path>
				<text class="viz-callout" x="150" y="237" text-anchor="middle">Mismatch: monitor passes, behavior stays bad.</text>
				<text class="viz-label" x="150" y="259" text-anchor="middle">The evidence channel became less informative;</text>
				<text class="viz-label" x="150" y="273" text-anchor="middle">the independent target label did not improve.</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> compare the same action row across both panels first: the independent verifier still finds bad behavior. Then compare the trace rows: direct pressure can replace incriminating evidence with benign-looking text, causing the trace monitor to pass. A useful flag in the first panel is evidence of monitorability, not proof that the trace is a complete causal transcript. This qualitative failure mode is documented by <a href="https://arxiv.org/abs/2503.11926">Baker et al.</a>; the distinction between plausible explanations and faithfulness is supported by <a href="https://arxiv.org/abs/2305.04388">Turpin et al.</a>; the graphic and synthetic wording are original.</figcaption>
</figure>

## What to measure

Define a target property $Y$, reasoning trace $C$, action or answer $A$, and monitor score $m(C, A)$. Evaluate:

- discrimination: can the monitor separate positive and negative cases?
- calibration: does score correspond to empirical risk?
- incremental value: does $C$ improve detection beyond $A$ and context alone?
- generalization: does it transfer across tasks, models, and attack strategies?
- robustness: can the policy evade the monitor while preserving the bad behavior?
- operational value: does intervention reduce harm at acceptable false-positive cost?

Accuracy on one benchmark is not enough.

## A clean experiment

1. Construct episodes with independently verifiable target properties.
2. Blind the monitor to labels and avoid leakage through templates.
3. Compare trace-only, answer-only, and combined monitors.
4. Hold out behavioral families, not only prompt paraphrases.
5. test adversarial traces and post-training interventions.
6. measure utility when alerts trigger review, refusal, or another model pass.

A monitor can be statistically strong and operationally weak if false positives overwhelm reviewers or arrive after an irreversible tool action.

## Training pressure and Goodhart's law

If the policy is trained directly to make a fixed monitor happy, it learns the monitor's decision boundary. The trace may become less informative without the underlying behavior improving.

Safer uses can include:

- monitoring without exposing the exact detector to the policy;
- rotating or holding out monitors;
- using traces as one signal among action checks and external verifiers;
- investigating how pretraining, synthetic data, mid-training, and RL alter monitorability;
- preserving evaluation sets that training never optimizes directly.

This is an active research area, not a solved recipe.

## Monitor versus interpreter

A monitor predicts a property from available evidence. It need not recover the model's true internal mechanism. Mechanistic interpretability instead tries to identify causal internal computation. The two can support each other, but a useful monitor may remain behaviorally predictive without a complete circuit-level explanation.

## Common confusions

- **"The trace is the model's true thought."** It is generated text and may not faithfully expose internal computation.
- **"A good classifier proves faithfulness."** Predictive signal does not show that every causal step is present.
- **"Train against the monitor."** Direct optimization can erase the evidence the monitor relies on.
- **"False positives are harmless."** In a deployed agent, excessive alerts can disable the oversight channel through fatigue.
- **"Final-answer evaluation is enough."** Harmful plans can fail by accident, and safe-looking answers can precede unsafe tool actions.
- **"Monitorability is one scalar property."** It varies by behavior, task, model, training stage, and observer.

## In an interview

State the target property, independent label source, baselines, held-out family, adversarial strategy, and intervention policy. Then explain how training itself may change the availability of the signal.

*Related: [mechanistic interpretability](/concepts/mechanistic-interpretability/), [scalable oversight and AI control](/concepts/scalable-oversight-and-ai-control/), and [investigate black-box behavior](/questions/investigate-black-box-model-behavior/).*
