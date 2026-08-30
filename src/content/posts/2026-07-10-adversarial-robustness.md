---
title: "Adversarial robustness"
description: "Small worst-case perturbations, threat models, adversarial training, robust evaluation, and the difference between security and ordinary distribution shift."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A model can hit 99% test accuracy and still fail on inputs a human cannot tell apart from clean ones, because average-case accuracy says nothing about the worst case an attacker can search for. That gap matters wherever an adversary is present: fraud, content moderation, malware, authentication, autonomous perception. An adversarial example is an input deliberately perturbed to cause failure while staying inside a defined threat model, and robustness only means something relative to that threat model: the attacker's knowledge, capabilities, norm or semantic budget, and the system being defended.

## First-order attacks

For loss $L(\theta, x, y)$, FGSM takes a single signed-gradient step:

$$x' = x + \epsilon\,\text{sign}(\nabla_x L).$$

Projected gradient descent iterates gradient steps and projects back into the allowed set. A meaningful evaluation uses multiple restarts and attacks that adapt to the defense, not one weak attack.

<!-- visual:adversarial-threat-set-boundary -->
<figure class="learning-figure plot-panel" aria-labelledby="adversarial-visual-title">
	<p class="visual-kicker">Threat-model geometry</p>
	<p class="visual-title" id="adversarial-visual-title">A small allowed move can contain a misclassified point.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 300" role="img" aria-labelledby="adversarial-svg-title adversarial-svg-desc">
			<title id="adversarial-svg-title">An adversarial perturbation crossing a decision boundary inside an L-infinity threat set</title>
			<desc id="adversarial-svg-desc">An illustrative two-feature plane has a dashed diagonal decision boundary separating class A at lower left from class B at upper right. A circle labeled clean x lies in class A at coordinates 125, 180. A dashed square centered on it represents all perturbations whose two feature changes are each at most epsilon. A solid arrow in the signed input-gradient direction ends at a diamond labeled adversarial x prime at coordinates 180, 125. The diamond remains inside the square but lies across the decision boundary in class B.</desc>
			<defs>
				<marker id="adversarial-arrowhead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
					<path class="viz-arrow-backward" d="M0 0L10 5L0 10Z"></path>
				</marker>
			</defs>
			<rect class="viz-plot-bg" x="38" y="28" width="286" height="222" rx="3"></rect>
			<path class="viz-gridline" d="M38 102H324 M38 176H324 M109.5 28V250 M181 28V250 M252.5 28V250"></path>
			<path class="viz-axis" d="M38 28V250H324"></path>
			<path class="viz-baseline" d="M105 28L245 250"></path>
			<text class="viz-axis-label" x="185" y="68" transform="rotate(58 185 68)">decision boundary</text>
			<text class="viz-callout" x="62" y="226">CLASS A</text>
			<text class="viz-callout" x="264" y="55">CLASS B</text>
			<rect class="viz-operating-guide" x="70" y="125" width="110" height="110" rx="2"></rect>
			<text class="viz-callout" x="76" y="144">allowed L∞ set</text>
			<text class="viz-label" x="76" y="159">each |δᵢ| ≤ ε</text>
			<path class="viz-pr-curve" marker-end="url(#adversarial-arrowhead)" d="M125 180L180 125"></path>
			<text class="viz-callout" x="137" y="170">sign(∇ₓL)</text>
			<circle class="viz-operating-point" cx="125" cy="180" r="6"></circle>
			<text class="viz-callout" x="108" y="200">clean x</text>
			<polygon class="viz-node--output" points="180,117 188,125 180,133 172,125"></polygon>
			<text class="viz-callout" x="192" y="121">adversarial x′</text>
			<text class="viz-label" x="192" y="136">still within budget</text>
			<text class="viz-axis-label" x="244" y="276">illustrative input feature 1</text>
			<text class="viz-axis-label" x="16" y="205" transform="rotate(-90 16 205)">input feature 2</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> start at the clean circle, then follow the signed input-gradient arrow. The diamond crosses the dashed decision boundary but stays inside the allowed square: ordinary accuracy gets the circle right, while worst-case robustness asks whether every point in that threat set keeps the same prediction.</figcaption>
</figure>

## Defenses

Adversarial training optimizes against generated worst-case perturbations and remains the strongest general baseline for norm-bounded attacks. It is expensive and usually trades clean accuracy for robust accuracy. Certified defenses prove robustness within a limited region but may not scale to realistic semantic threats.

## Why evaluation fails

- Gradient masking makes weak attacks fail, which creates false confidence in the defense.
- The attack does not adapt to preprocessing or randomness.
- The threat model is irrelevant to the real attacker.
- Robustness is measured on average while the rare failures are the catastrophic ones.
- The deployed pipeline has non-model attack surfaces the eval ignores.

## In an interview

1. Specify the threat model and the failure you care about.
2. Establish clean and attacked baselines.
3. Use adaptive, sufficiently strong attacks and independent tools.
4. Discuss adversarial training and its cost/accuracy trade-off.
5. Widen from model robustness to detection, rate limiting, human review, and incident response.

## Common confusions

- **"Noise augmentation gives adversarial robustness."** Random noise does not reliably approximate worst-case optimization.
- **"High PGD accuracy means secure."** Only within the tested threat model and attack implementation.
- **"Adversarial and natural robustness are the same."** They can interact but test different failure processes.

*Related: [regularization](/concepts/regularization/), [model interpretability](/concepts/model-interpretability/), and [epistemic uncertainty](/concepts/epistemic-vs-aleatoric-uncertainty/).*
