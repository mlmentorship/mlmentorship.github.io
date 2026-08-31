---
title: "L1 vs L2 regularization, beyond the formula"
description: "The math is identical to most candidates: penalty terms in the loss. The senior signal is the Bayesian interpretation, the optimization geometry, and when each is the right choice."
date: "2025-12-04"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth at every level.*

The L4 answer states the formulas. The L6 answer explains *why* L1 induces sparsity geometrically, names the Bayesian priors each corresponds to, and gives a clear practitioner rule for when to pick which.

## What an L4 answer sounds like

> "L1 adds the absolute value of weights to the loss, L2 adds the squared values. L1 leads to sparse solutions, L2 leads to small weights."

Correct, no depth. You've heard the rule, not the reason.

## What an L5 answer sounds like

> "Both add a penalty term to the loss to constrain the weights:
>
> - L1: `lambda * sum(|w_i|)`
> - L2: `lambda * sum(w_i^2)`
>
> The Bayesian view: L1 corresponds to a Laplace prior on the weights centered at zero; L2 corresponds to a Gaussian prior. Both pull weights toward zero, but with different shapes.
>
> Why L1 induces sparsity (the geometric view): the L1 ball has corners at the axes. The optimum of the loss + L1 penalty often lands at one of those corners, which means many weights are exactly zero. The L2 ball is round, so the optimum lands somewhere off the axes, with all weights small but rarely zero.
>
> Practical use: L2 (weight decay) is the default for deep networks; pick L1 when you want feature selection (sparse models, interpretability, downstream sparse computation)."

This is L5. You've named the prior interpretation, given the geometric intuition, and made a practitioner recommendation.

<!-- visual:l1-l2-first-contact-geometry -->
<figure class="learning-figure" aria-labelledby="regularization-contact-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="regularization-contact-title">Where does the first loss contour touch the penalty boundary?</p>
	<div class="visual-grid--two" role="group" aria-label="Matched coefficient-space comparison of L1 and L2 constrained optima">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 270" role="img" aria-labelledby="l1-contact-panel-title l1-contact-panel-desc">
				<title id="l1-contact-panel-title">L1 contact at an axis corner</title>
				<desc id="l1-contact-panel-desc">A diamond-shaped equal-L1 boundary is centered at the origin of a w1 and w2 coefficient plane. A circular loss contour centered on the shared unconstrained minimum first touches the diamond at its right corner on the w1 axis. The marked constrained optimum therefore has positive w1 and w2 exactly zero.</desc>
				<rect class="viz-plot-bg" x="8" y="27" width="284" height="231" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="17">L1 · CORNER CAN ZERO A COEFFICIENT</text>
				<path class="viz-axis" d="M30 145H276M130 238V43"></path>
				<text class="viz-label" x="274" y="139" text-anchor="end">w1</text>
				<text class="viz-label" x="137" y="51">w2</text>
				<path d="M130 80L195 145L130 210L65 145Z" style="fill:var(--viz-focus-bg);fill-opacity:.45;stroke:var(--viz-focus-stroke);stroke-width:2.5"></path>
				<text class="viz-callout" x="64" y="226">|w1| + |w2| ≤ t</text>
				<circle cx="240" cy="115" r="54.1" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-dasharray:5 3"></circle>
				<text class="viz-label" x="205" y="70">loss contour</text>
				<path d="M240 106L243 112L250 113L245 118L246 125L240 122L234 125L235 118L230 113L237 112Z" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.8"></path>
				<text class="viz-label" x="229" y="101" text-anchor="end">no penalty</text>
				<circle class="viz-operating-point" cx="195" cy="145" r="6"></circle>
				<path class="viz-operating-guide" d="M195 145V212"></path>
				<text class="viz-callout" x="202" y="191">first contact</text>
				<text class="viz-axis-label" x="202" y="207">w2 = 0</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 270" role="img" aria-labelledby="l2-contact-panel-title l2-contact-panel-desc">
				<title id="l2-contact-panel-title">L2 contact away from both axes</title>
				<desc id="l2-contact-panel-desc">A circular equal-L2 boundary is centered at the origin of the same w1 and w2 coefficient plane. A loss contour centered on the same unconstrained minimum first touches the smooth boundary between the axes. The marked constrained optimum therefore keeps both w1 and w2 nonzero.</desc>
				<rect class="viz-plot-bg" x="8" y="27" width="284" height="231" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="17">L2 · SMOOTH CONTACT IS USUALLY DENSE</text>
				<path class="viz-axis" d="M30 145H276M130 238V43"></path>
				<text class="viz-label" x="274" y="139" text-anchor="end">w1</text>
				<text class="viz-label" x="137" y="51">w2</text>
				<circle cx="130" cy="145" r="65" style="fill:var(--viz-focus-bg);fill-opacity:.45;stroke:var(--viz-focus-stroke);stroke-width:2.5"></circle>
				<text class="viz-callout" x="64" y="226">w1² + w2² ≤ t</text>
				<circle cx="240" cy="115" r="49" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-dasharray:5 3"></circle>
				<text class="viz-label" x="205" y="70">loss contour</text>
				<path d="M240 106L243 112L250 113L245 118L246 125L240 122L234 125L235 118L230 113L237 112Z" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.8"></path>
				<text class="viz-label" x="229" y="101" text-anchor="end">no penalty</text>
				<circle class="viz-operating-point" cx="193" cy="128" r="6"></circle>
				<path class="viz-operating-guide" d="M193 128V212"></path>
				<text class="viz-callout" x="200" y="184">first contact</text>
				<text class="viz-axis-label" x="200" y="200">w1, w2 ≠ 0</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> imagine a loss contour expanding from the star until it first reaches the allowed coefficient region. With the L1 diamond, an axis-aligned corner can be the first contact, making one coefficient exactly zero. With the smooth L2 circle, contact is usually between the axes, so both coefficients shrink but remain nonzero. The geometry explains a tendency, not a guarantee that every L1 solution is sparse. This is an original schematic checked against <a href="https://doi.org/10.1111/j.2517-6161.1996.tb02080.x">Tibshirani's lasso paper</a> and <a href="https://hastie.su.domains/ElemStatLearn/"><cite>The Elements of Statistical Learning</cite></a>.</figcaption>
</figure>

## What an L6 answer sounds like

> "...a few more things worth saying:
>
> **L1 is non-differentiable at zero**, which means standard gradient descent doesn't drive weights exactly to zero; you need proximal methods (ISTA, FISTA) or specialized solvers. In practice, frameworks use a sub-gradient (typically zero at zero) and rely on small numerical noise for actual sparsity.
>
> **AdamW vs Adam with weight decay**: 'L2 regularization' as a `+ lambda * w^2` term in the loss interacts badly with adaptive optimizers (the regularization gets scaled by the adaptive learning rate). AdamW decouples weight decay by applying it directly to the weights, not the gradient. This is what most modern transformers use.
>
> **Elastic Net** combines both. Useful when you want some sparsity but L1 alone is unstable in the presence of correlated features.
>
> **Implicit regularization** matters more than people think. SGD's noise, early stopping, and architecture (e.g., dropout, batch norm) often dominate explicit penalty regularization in deep learning. For very large models, weight decay is more about training stability than overfitting prevention."

## Tells that get you a strong-hire vote

- You name the **Bayesian priors** (Laplace for L1, Gaussian for L2).
- You give the **geometric** sparsity argument, not just the empirical claim.
- You distinguish **AdamW vs Adam with weight decay**.
- You acknowledge **implicit regularization** matters more for deep nets.

## Tells that get you down-leveled

- "L1 is sparse, L2 is small" with no further explanation.
- Confusion about the AdamW vs Adam weight-decay distinction.
- Suggesting L1 for deep networks as a default (rare; usually L2 or weight decay).
- No awareness that L1 needs special optimizers for true sparsity.

## Common follow-up

"You said weight decay is the default. When would you turn it off?"

The L6 answer:

> "Frontier-scale LLM pretraining uses very small weight decay (e.g., 0.01 or 0.1 in AdamW) because the model is underfitting at trillion-token scale, not overfitting. Some teams report no quality loss with weight decay = 0; others find a small WD helps with training stability. For typical supervised models on small data, default weight decay (1e-4 to 1e-2) is meaningful regularization."

---

*Related: [regularization](/concepts/regularization/) and [Adam and AdamW](/concepts/adam-and-adamw/).*
