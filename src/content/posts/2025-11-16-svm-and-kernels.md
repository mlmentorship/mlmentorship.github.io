---
title: "SVM and the kernel trick"
description: "Solve a margin classifier, price violations, and change the feature space without confusing a score with a probability."
date: "2025-11-16"
draft: false
tags: ["concepts"]
category: "concepts"
---

An SVM chooses a decision boundary by balancing its norm against small signed margins.

## Summary

Let negative examples be $x=-2,-1$ and positive examples be $x=1,2$.
Which linear score $f(x)=wx+b$ separates them with the widest normalized margin?
Use labels $y\in\{-1,1\}$ and predict the sign of the score, with an explicit tie policy at zero.
The figure illustrates a soft-margin fit; its three highlighted points are not this four-point dataset.

<!-- visual:svm-support-vectors-set-boundary -->
<figure class="learning-figure plot-panel" aria-labelledby="svm-support-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="svm-support-title">Only nonzero dual weights contribute to the SVM decision sum</p><div class="visual-argument" data-visual-argument="svm-support-vectors-set-boundary" data-argument-kind="ledger" data-argument-composition="compare"><svg viewBox="0 0 360 268" role="img" aria-label="Zero dual weight means no contribution to the decision sum. Only support vectors with nonzero alpha enter the weighted kernel decision. Other training examples still exist but add zero here."><title>Zero dual weight means no contribution to the decision sum</title><desc>Only support vectors with nonzero alpha enter the weighted kernel decision. Other training examples still exist but add zero here.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Training-example dual weights</tspan></text><g data-proof-token=".7" data-proof-row="0" data-proof-position="0" data-proof-box="20,60,74,48"><rect x="20" y="60" width="74" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="57" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="57" dy="0">.7</tspan></text></g><g data-proof-token="0" data-proof-row="0" data-proof-position="1" data-proof-box="100,60,74,48"><rect x="100" y="60" width="74" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="137" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="137" dy="0">0</tspan></text></g><g data-proof-token=".4" data-proof-row="0" data-proof-position="2" data-proof-box="180,60,74,48"><rect x="180" y="60" width="74" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="217" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="217" dy="0">.4</tspan></text></g><g data-proof-token="0" data-proof-row="0" data-proof-position="3" data-proof-box="260,60,74,48"><rect x="260" y="60" width="74" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="297" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="297" dy="0">0</tspan></text></g><text x="20" y="144" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Contributions to decision sum</tspan></text><g data-proof-token=".7 y1 K1" data-proof-row="1" data-proof-position="0" data-proof-box="20,176,74,56"><rect x="20" y="176" width="74" height="56" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="57" y="198" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="57" dy="0">.7 y1</tspan><tspan x="57" dy="18">K1</tspan></text></g><g data-proof-token="0" data-proof-row="1" data-proof-position="1" data-proof-box="100,176,74,56"><rect x="100" y="176" width="74" height="56" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="137" y="207" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="137" dy="0">0</tspan></text></g><g data-proof-token=".4 y3 K3" data-proof-row="1" data-proof-position="2" data-proof-box="180,176,74,56"><rect x="180" y="176" width="74" height="56" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="217" y="198" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="217" dy="0">.4 y3</tspan><tspan x="217" dy="18">K3</tspan></text></g><g data-proof-token="0" data-proof-row="1" data-proof-position="3" data-proof-box="260,176,74,56"><rect x="260" y="176" width="74" height="56" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="297" y="207" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="297" dy="0">0</tspan></text></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 340" role="img" aria-labelledby="svm-support-svg-title svm-support-svg-desc">
		<title id="svm-support-svg-title">Support vectors on or inside two SVM margin rails</title>
		<desc id="svm-support-svg-desc">A vertical decision boundary lies between dashed negative-one and positive-one margin rails. Negative-class circles appear on the left and positive-class diamonds on the right. Three points on or inside the rails have prominent outer rings and labels saying alpha i is greater than zero, so they are support vectors. Four correctly classified points beyond the rails have no rings and labels saying alpha i equals zero. A note states that prediction sums only over the three ringed support vectors.</desc>
		<rect class="viz-plot-bg" x="12" y="30" width="336" height="240" rx="4"></rect>
		<text class="viz-axis-label" x="54" y="51">CLASS −1 · CIRCLES</text>
		<text class="viz-axis-label" x="306" y="51" text-anchor="end">CLASS +1 · DIAMONDS</text>
		<path class="viz-baseline" d="M120 58V246M240 58V246"></path>
		<path class="viz-axis" d="M180 58V246"></path>
		<text class="viz-label" x="116" y="66" text-anchor="end">f(x) = −1</text>
		<text class="viz-label" x="184" y="66">f(x) = 0</text>
		<text class="viz-label" x="244" y="66">f(x) = +1</text>
		<circle cx="58" cy="112" r="6" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></circle>
		<circle cx="74" cy="211" r="6" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></circle>
		<text class="viz-label" x="38" y="132">beyond margin</text>
		<text class="viz-callout" x="38" y="147">αᵢ = 0</text>
		<circle cx="120" cy="137" r="11" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5"></circle>
		<circle cx="120" cy="137" r="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></circle>
		<text class="viz-callout" x="108" y="160" text-anchor="end">on rail · αᵢ &gt; 0</text>
		<path d="M240 112L247 119L240 126L233 119Z" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5"></path>
		<path d="M240 114L245 119L240 124L235 119Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></path>
		<text class="viz-callout" x="252" y="142">on rail · αᵢ &gt; 0</text>
		<circle cx="207" cy="198" r="11" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5"></circle>
		<path d="M207 192L213 198L207 204L201 198Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></path>
		<text class="viz-callout" x="219" y="220">inside · αᵢ &gt; 0</text>
		<path d="M302 94L308 100L302 106L296 100Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></path>
		<path d="M292 190L298 196L292 202L286 196Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></path>
		<text class="viz-label" x="322" y="166" text-anchor="end">beyond margin</text>
		<text class="viz-callout" x="322" y="181" text-anchor="end">αᵢ = 0</text>
		<rect class="viz-node viz-node--focus" x="30" y="282" width="300" height="40" rx="4"></rect>
		<text class="viz-callout" x="180" y="299" text-anchor="middle">decision(x) = sign(Σᵢ∈SV αᵢ yᵢ K(xᵢ, x) + b)</text>
		<text class="viz-node-value" x="180" y="315">prediction sums over the 3 ringed points only</text>
	</svg>
	<div class="visual-derivation-caption">compare each point with its class-margin rail. The three rings identify positive dual coefficients in this schematic fit. Strictly outside the margin, coefficients are zero. A point on a rail can also have zero coefficient in a degenerate solution. Soft margin permits positive coefficients inside the rails. Sources: <a href="https://link.springer.com/article/10.1023/A:1009715923555">Burges's tutorial</a> and the <a href="https://scikit-learn.org/stable/modules/svm.html">SVM formulation</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="svm-support-vectors-set-boundary"><strong>Check:</strong> Which displayed points actually contribute to the decision sum? Why is lying on a margin alone not a universal proof of nonzero alpha?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="svm-support-vectors-set-boundary">Only support vectors with nonzero alpha enter the weighted kernel decision. Other training examples still exist but add zero here.</span></figcaption>
</figure>

## The hard-margin SVM (separable case)

Normalize the nearest signed scores to at least one:

$$
\min_{w,b}\frac12\|w\|^2
\quad\text{subject to}\quad y_i(w^\top x_i+b)\geq1.
$$

The examples at $-1$ and $1$ require $w-b\geq1$ and $w+b\geq1$.
Adding gives $w\geq1$.
Thus $w=1,b=0$ achieves the minimum objective $1/2$.
Signed margins are $2,1,1,2$; the rails are at $-1$ and $1$, with width $2/\|w\|=2$.
Without score normalization, multiplying both parameters changes no boundary and makes raw score gaps meaningless.

## The soft-margin SVM (non-separable)

**Change the requirement:** price violations instead of forbidding them.
For $C>0$, minimize

$$
\frac12\|w\|^2+C\sum_i\max(0,1-y_i f(x_i)).
$$

The separating fit costs $1/2$; the constant-zero score costs $4C$.
At $C=0.1$, zero beats that separator: $0.4<0.5$.
Neither comparison proves that zero is optimal.
The optimum is $w=0.5,b=0$, with objective $0.125+0.1=0.225$.
Coefficients $(0.075,0.1,0.1,0.075)$ satisfy the dual constraints and achieve the same objective.
All four points now support the fit; retaining the hard-margin coefficients would violate their new upper bound.
Slack is $\xi_i=\max(0,1-y_i f(x_i))$, not necessarily a misclassification.
A correctly classified score $y_i f(x_i)=0.4$ still pays $0.6$.

## The dual formulation and the kernel trick

The soft-margin dual is

$$
\max_\alpha\sum_i\alpha_i-\frac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_jK(x_i,x_j),
\qquad 0\leq\alpha_i\leq C,\quad\sum_i\alpha_i y_i=0.
$$

For hard margin, remove the upper bound on $\alpha_i$.
In the four-point example, coefficients $(0,1/2,1/2,0)$ reconstruct $w=\sum_i\alpha_i y_ix_i=1$.
Their dual objective is $1-1/2=1/2$, matching the primal and certifying optimality.

**Change the requirement:** use an RBF kernel on only $x=-1,1$, with labels $-1,1$ and $C=3$.
Set $K(x,z)=\exp(-\gamma(x-z)^2)$ and $\gamma=\log(2)/4$.
The cross-similarity is $1/2$; diagonal entries are one.
The equality constraint makes both coefficients $a$.
Their dual is $2a-a^2/2$, maximized at $a=2$ within $[0,3]$.
With $b=0$, both signed scores equal $2(1-1/2)=1$.
Doubling $\gamma$ makes cross-similarity $1/4$ and changes the optimum to $a=4/3$.
Do not reuse the old coefficients after changing kernel geometry.
Use symmetric positive-semidefinite kernels; an arbitrary similarity need not yield a convex dual.

## Support vectors

Prediction uses $f(x)=\sum_{i:\alpha_i>0}\alpha_i y_iK(x_i,x)+b$.
Zero coefficients remove terms from this fitted score, not a guarantee about arbitrary retraining or preprocessing changes.
For a soft-margin optimum, $0<\alpha_i<C$ implies a point lies on its rail.
Points with $\alpha_i=C$ can lie on or inside it.

**Reconstruct:** does a large positive score mean a calibrated probability?
No: margin optimization supplies no probability normalization or calibration guarantee.

<!-- learning:core-end -->
<div data-learning-extension>

## When to use SVMs in 2026

Compare a scaled linear SVM, an RBF SVM and task-appropriate alternatives under the same validation protocol.
A dense Gram matrix stores $n^2$ entries; solver cost depends on the implementation and data.
Linear solvers need not materialize that matrix.
Kernel prediction evaluates one term per retained support vector, so measure deployment cost too.

## Common pitfalls

Scale features using training-only statistics and tune interacting parameters inside validation folds.
For $K(x,z)=\exp(-\gamma\|x-z\|^2)$, changing units changes similarities unless $\gamma$ changes appropriately.
Hinge loss is not log loss, and sigmoid similarities are not PSD for arbitrary parameters.
The primal is strictly convex in $w$, not jointly strictly convex in the unregularized intercept.
For example, opposite labels at the same zero input admit every $b\in[-1,1]$ when $w=0$.

## Related

- [Kernel methods](/concepts/kernel-methods-and-the-kernel-trick/) owns feature-inner-product equivalence.
- [Calibration](/concepts/calibration/) separates scores from estimated probabilities.
- [Linear regression](/concepts/linear-regression/) provides a squared-error contrast.

</div>
