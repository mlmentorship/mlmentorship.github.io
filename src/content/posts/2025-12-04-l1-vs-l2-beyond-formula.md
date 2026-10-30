---
title: "L1 versus L2: derive the zero coefficient"
description: "Solve a common quadratic under absolute and squared penalties, test the zero subgradient condition, and separate objective sparsity from solver behavior."
date: "2025-12-04"
draft: false
tags: ["questions"]
category: "questions"
---


Compare the exact fitted coefficients produced by absolute and squared penalties under one explicitly defined quadratic objective.

## Problem

**Context.** Fit $a=(2,0.3)$ using data loss $L_{data}(w)=\frac12\lVert w-a\rVert_2^2$ and coefficient $\lambda=0.5$.
Compare L1 penalty $\lambda\sum_i|w_i|$ with half-scaled L2 penalty $\frac\lambda2\sum_iw_i^2$.

**Task.** Derive the minimizing coefficients for each penalized objective and identify which coefficients become exactly zero.
Explain how changing the penalty convention or feature scaling changes the comparison.

**Constraints.** Coordinates are independent, feature units are comparable, and no intercept appears in this fixture.
Use subgradients at the absolute-value corner; distinguish objective sparsity from a solver merely returning a small number.

<span id="what-an-l4-answer-sounds-like"></span>
## Fix the objective

Use L1 penalty $\lambda\sum_i|w_i|$ and L2 penalty $\frac\lambda2\sum_iw_i^2$.
An unhalved L2 convention, $\lambda\sum_iw_i^2$, doubles its penalty derivative for the same numerical coefficient.
Changing summed loss to mean loss also changes relative penalty strength unless rescaled.
State intercept treatment and feature scaling in a real fit.

<span id="what-an-l5-answer-sounds-like"></span>
## Solve the corner and the smooth condition

For one L1 coordinate, minimize $\frac12(w-a)^2+\lambda|w|$.

- If $w>0$, stationarity gives $w-a+\lambda=0$, valid for $a>\lambda$.
- If $w<0$, it gives $w-a-\lambda=0$, valid for $a<-\lambda$.
- At zero, $0\in-a+\lambda[-1,1]$ holds exactly when $|a|\le\lambda$.

Together these give $w=\operatorname{sign}(a)\max(|a|-\lambda,0)$.
For L2, $w-a+\lambda w=0$ gives $w=a/(1+\lambda)$.

| Objective | First coordinate | Second coordinate |
| --- | --- | --- |
| Unpenalized | $2$ | $0.3$ |
| L1 | $2-0.5=1.5$ | $0$, since $0.3\le0.5$ |
| Half-scaled L2 | $2/1.5=4/3$ | $0.3/1.5=0.2$ |

<!-- visual:l1-l2-first-contact-geometry -->
<figure class="learning-figure" aria-labelledby="regularization-contact-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="regularization-contact-title">An L1 corner can touch the loss at an exactly zero coefficient</p><div class="visual-argument" data-visual-argument="l1-l2-first-contact-geometry" data-argument-kind="geometry"><p class="visual-argument-premise">A corner can put one coefficient exactly at zero</p><svg viewBox="0 0 360 443" role="img" aria-label="A corner can put one coefficient exactly at zero. For squared distance from (2,0.4), the unit L1 constraint selects (1,0). The unit L2 constraint selects a nonzero second coordinate."><title>A corner can put one coefficient exactly at zero</title><desc>For squared distance from (2,0.4), the unit L1 constraint selects (1,0). The unit L2 constraint selects a nonzero second coordinate.</desc><path d="M32 220L330 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M134 26L134 322" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="328" y="239" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="328" dy="0">x</tspan></text><text x="144" y="25" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="144" dy="0">y</tspan></text><polygon data-proof-polygon points="180,220 134,174 88,220 134,266" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.75"></polygon><circle cx="134" cy="220" r="46" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75;stroke-dasharray:5 3"></circle><path d="M134 220L180 220" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M171.8959760788259 223.91468980700108L180 220" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M171.8959760788259 216.08531019299892L180 220" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><circle cx="180" cy="220" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M134 220L179.10671108178232 210.97865778364354" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M171.92779566455593 216.40665681020107L179.10671108178232 210.97865778364354" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M170.39232799412815 208.72931845806212L179.10671108178232 210.97865778364354" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><circle cx="179.10671108178232" cy="210.97865778364354" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><path d="M20 349L45 349" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="57" y="353" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="57" dy="0">L1-constrained optimum: (1, 0)</tspan></text><path d="M20 381L45 381" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><text x="57" y="385" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="57" dy="0">L2-constrained optimum: (0.9806,</tspan><tspan x="57" dy="18">0.1961)</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption">expand a loss contour from the star until it touches the allowed region. A diamond corner can zero a coefficient; a smooth-circle contact can retain both. This schematic illustrates a tendency, not the numerical coordinates of our separate analytic fixture. Equivalent constrained budgets depend on the fitted optimum; equal numeric budgets across norms are not equivalent penalties. Sources: <a href="https://doi.org/10.1111/j.2517-6161.1996.tb02080.x">Tibshirani's lasso paper</a> and <a href="https://hastie.su.domains/ElemStatLearn/"><cite>The Elements of Statistical Learning</cite></a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="l1-l2-first-contact-geometry"><strong>Check:</strong> What feature of the L1 boundary makes a zero coefficient possible at first contact?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="l1-l2-first-contact-geometry">For squared distance from (2,0.4), the unit L1 constraint selects (1,0). The unit L2 constraint selects a nonzero second coordinate.</span></figcaption>
</figure>

<span id="what-an-l6-answer-sounds-like"></span>
## Change the requirements

**New task:** reduce L1 strength to $0.2$ while keeping the fitting surface fixed.
Move the threshold inward.
**Check:** $(2-0.2,0.3-0.2)=(1.8,0.1)$; the removed coefficient returns.
**Use when:** validation or a feature-count requirement favors less sparsity.

**New task:** combine L1 strength $\lambda_1=0.5$ with L2 strength $\lambda_2=1$.
Threshold, then shrink: $w=\operatorname{soft}(a,\lambda_1)/(1+\lambda_2)$, giving $(0.75,0)$.
**Use when:** sparse selection and correlated-feature stability both matter.
**Boundary:** this reference has independent coordinates; correlated designs require solving the coupled objective.

**Failure:** for $\frac12w^2+|w|$, step size one sends $w=0.25$ to $-1$, then alternates between $1$ and $-1$.
The optimum is zero; fixed-step subgradient updates need not reach it.
Numerical jitter is not an exact-sparsity method.

**Reconstruct:** multiply a feature by $100$ while preserving its prediction contribution. Is its coefficient penalty unchanged?

<details><summary>Answer</summary>

No. Its coefficient becomes one hundredth as large.
The same contribution receives one hundredth the L1 penalty and one ten-thousandth the L2 penalty.
Standardize features or justify scale-specific penalties.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

<span id="tells-that-get-you-a-strong-hire-vote"></span>
## Priors and solver choices

Independent Laplace and Gaussian priors yield L1 and L2 MAP penalties under matching likelihood units and scale.
MAP is a point estimate, not the whole posterior.
Intercept exclusions and rescaled features alter that interpretation.

Proximal gradient methods apply soft thresholding after the smooth-loss step; their threshold includes step size.
ISTA, FISTA, coordinate descent, and specialized solvers handle nonsmoothness explicitly.
The separable closed form does not extend unchanged to an arbitrary correlated design.
See <a href="https://web.stanford.edu/~boyd/papers/prox_algs.html">Parikh and Boyd's Proximal Algorithms</a>.

<span id="tells-that-get-you-down-leveled"></span>
## Keep optimizer effects separate

For frozen coordinate preconditioners, a coupled L2 gradient is preconditioned; AdamW's shrinkage is separate.
[The decay trace](/concepts/weight-decay-vs-l2/) computes that difference.
Neither procedure is universally preferable for every statistical objective.

Optimizer noise, architecture, and early stopping can bias fitted solutions without an explicit penalty.
Their generalization effects and relative importance depend on workload and training protocol.

## Common follow-up

When should decay be disabled?
Compare matched validation outcomes, parameter groups, schedules, and deployment constraints, including a zero-decay control.
Model size alone does not prescribe a universal coefficient.

---

*Related: [regularization](/concepts/regularization/) and [Adam and AdamW](/concepts/adam-and-adamw/).*

</div>
