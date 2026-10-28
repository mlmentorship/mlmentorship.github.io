---
title: "Logistic regression"
description: "Compute a Bernoulli likelihood and gradient, separate probability fitting from decision costs, and test interactions, separation, and odds interpretation."
date: "2026-04-17"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Logistic regression models binary outcomes with affine log-odds, not an affine probability.
Fit probabilities using a Bernoulli likelihood; choose actions separately using deployment costs.

**Task:** independent observations have $x=[0,1]$, $y=[0,1]$.
Evaluate candidate intercept $b=0$ and slope $w=\log2$ using summed negative log-likelihood.
Find both probabilities, the likelihood, and the intercept/slope gradient.
These coefficients are a checked candidate, not an asserted finite optimum for separable data.

The [maximum-likelihood lesson](/concepts/maximum-likelihood-estimation/) supplies the fitting principle; here the conditional observation model is Bernoulli.

## The model

For binary $y \in \{0, 1\}$:

$$
p(y = 1 \mid x; w, b) = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}.
$$

The log-odds (logit) is linear in $x$:

$$
\log \frac{p(y=1 \mid x)}{p(y=0 \mid x)} = w^\top x + b.
$$

Odds are $p/(1-p)=e^z$, with $z=w^\top x+b$.
Equal score increments multiply odds equally, but do not add equal probability increments.

<!-- visual:logit-odds-probability-map -->
<figure class="learning-figure" aria-labelledby="logit-map-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="logit-map-title">See how a linear score becomes multiplicative odds and a saturating probability.</p>
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 290" role="img" aria-labelledby="logit-map-svg-title logit-map-svg-desc">
			<title id="logit-map-svg-title">Linear logits mapped to odds and sigmoid probabilities</title>
			<desc id="logit-map-svg-desc">A sigmoid curve maps the linear score z to probability. Five equally spaced scores, minus two through two, have probabilities 0.12, 0.27, 0.50, 0.73, and 0.88. Unit score increments multiply odds by e. The dashed zero-score line is the probability-one-half decision boundary, a special case rather than a universal cost-sensitive threshold.</desc>
			<rect class="viz-plot-bg" x="34" y="26" width="310" height="204" rx="4"></rect>
			<path class="viz-gridline" d="M34 128H344M34 77H344M34 179H344"></path>
			<path class="viz-axis" d="M34 26V230H344"></path>
			<text class="viz-label" x="16" y="231">0</text>
			<text class="viz-label" x="8" y="132">0.5</text>
			<text class="viz-label" x="16" y="31">1</text>
			<text class="viz-axis-label" x="34" y="16">probability p</text>
			<path d="M34 214.5C45 212.5 55 209.5 65 205.7S111 186 127 175.1S174 140 189 128S235 91.8 251 80.9S298 54.4 313 50.3S333 43.5 344 41.5" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-linecap:round"></path>
			<path class="viz-operating-guide" d="M189 26V230"></path>
			<circle class="viz-operating-point" cx="65" cy="205.7" r="5"></circle>
			<circle class="viz-operating-point" cx="127" cy="175.1" r="5"></circle>
			<path d="M189 121L196 128L189 135L182 128Z" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2.5"></path>
			<circle class="viz-operating-point" cx="251" cy="80.9" r="5"></circle>
			<circle class="viz-operating-point" cx="313" cy="50.3" r="5"></circle>
			<text class="viz-callout" x="70" y="197">p=.12</text>
			<text class="viz-callout" x="132" y="167">p=.27</text>
			<text class="viz-callout" x="197" y="122">p=.50</text>
			<text class="viz-callout" x="244" y="72" text-anchor="end">p=.73</text>
			<text class="viz-callout" x="306" y="42" text-anchor="end">p=.88</text>
			<text class="viz-axis-label" x="197" y="145">p=.5 boundary</text>
			<path d="M65 238V244M127 238V244M189 238V244M251 238V244M313 238V244M65 241H313" style="fill:none;stroke:var(--c-text-soft);stroke-width:1.4"></path>
			<text class="viz-axis-label" x="65" y="258" text-anchor="middle">z=-2</text>
			<text class="viz-axis-label" x="127" y="258" text-anchor="middle">-1</text>
			<text class="viz-axis-label" x="189" y="258" text-anchor="middle">0</text>
			<text class="viz-axis-label" x="251" y="258" text-anchor="middle">1</text>
			<text class="viz-axis-label" x="313" y="258" text-anchor="middle">2</text>
			<text class="viz-label" x="65" y="276" text-anchor="middle">odds .14</text>
			<text class="viz-label" x="127" y="276" text-anchor="middle">.37</text>
			<text class="viz-label" x="189" y="276" text-anchor="middle">1</text>
			<text class="viz-label" x="251" y="276" text-anchor="middle">2.72</text>
			<text class="viz-label" x="313" y="276" text-anchor="middle">7.39</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> each unit score step multiplies odds by e. The five probability markers compress toward zero and one. The zero-score line marks p=0.5, appropriate for equal error costs under the stated probability model. Unequal costs can move the action threshold without refitting this sigmoid. Sources: <a href="https://www.statlearning.com/"><cite>An Introduction to Statistical Learning</cite></a> and <a href="https://cs229.stanford.edu/notes2022fall/main_notes.pdf?forcedefault=true">Stanford CS229</a>. <span class="visual-reader-check" data-reader-check="logit-odds-probability-map"><strong>Check:</strong> Why does z increasing from 0 to 1 multiply odds by 2.72 rather than add fixed probability?</span></figcaption>
</figure>

## Training

Negative log-likelihood (binary cross-entropy):

$$
L(w, b) = -\sum_{i=1}^{n} \big[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \big].
$$

Our scores $[0,\log2]$ give probabilities $[1/2,2/3]$.
The observed-label likelihood is $(1-1/2)(2/3)=1/3$, so $L=\log3\approx1.098612$.
Residuals $p-y=[1/2,-1/3]$ give

$$
\frac{\partial L}{\partial b}=\frac12-\frac13=\frac16,
\qquad \frac{\partial L}{\partial w}=0\cdot\frac12+1\cdot\left(-\frac13\right)=-\frac13.
$$

The negative gradient locally decreases the intercept and increases the slope.
This is Bernoulli cross-entropy, not a Gaussian squared-error objective.

With augmented design matrix $D$ containing the intercept column, $\nabla L=D^\top(p-y)$.
Its Hessian is $D^\top\operatorname{diag}(p_i(1-p_i))D$, positive semidefinite for fixed features.
Convexity does not ensure a finite, identifiable minimizer.
Evaluate each term stably as $\operatorname{softplus}(z)-yz$, with $\operatorname{softplus}(z)=\max(z,0)+\log(1+e^{-|z|})$.

### Change decision costs

**New task:** false positives cost $1$; false negatives cost $4$; correct actions cost zero.
Keep the probabilities and compare positive-action cost $1-p$ with negative-action cost $4p$.
**Check:** choose positive for $p>1/5$; at $p=0.3$, costs are $0.7$ versus $1.2$.
**Use when:** these probabilities describe the deployment population and errors have unequal costs.

### Change the feature basis

**New task:** represent XOR at $(0,0),(0,1),(1,0),(1,1)$.
Add interaction $x_1x_2$; candidate score $-1+2x_1+2x_2-4x_1x_2$ gives $[-1,1,1,-1]$.
**Check:** threshold $0.5$ recovers all four classes.
**Use when:** additive log-odds in original features misses a known interaction.
The model remains linear in expanded coefficients; these are fixture coefficients, not an MLE claim.

## Common pitfalls

**Failure:** for $x=[-1,1]$, $y=[0,1]$, and $b=0$, increasing $w$ drives loss toward zero.
No finite $w$ attains it: complete separation defeats the implied finite optimum.
A sigmoid does not establish calibration under finite samples, misspecification, regularization, or deployment shift.

**Reconstruct:** a coefficient $\log2$ doubles odds. Starting at $p=0.2$, what follows one feature unit, holding others fixed?

<details><summary>Answer</summary>

Odds move from $1/4$ to $1/2$, giving $p=(1/2)/(1+1/2)=1/3$, not $0.4$.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

## Multinomial / softmax regression

For mutually exclusive classes, normalize affine class scores with softmax and fit categorical cross-entropy.
Multilabel outcomes instead need an appropriate joint or separate-output model; not every classifier uses softmax.

## Properties

For an additive coefficient, $w_j$ is a conditional log-odds change holding other features fixed; $e^{w_j}$ is its odds ratio.
Interactions alter that interpretation, and it is not automatically a causal effect.
At probability threshold $t$, the boundary is $w^\top x+b=\log(t/(1-t))$.

## When to use vs. alternatives

Compare held-out probability quality, operating costs, latency, feature construction, and maintainability against relevant alternatives.
Validate calibration separately; a model family is not a calibration certificate.

L-BFGS uses quasi-Newton updates; SGD/Adam use gradient updates; Newton-Raphson/IRLS uses curvature through weighted least squares.
Solver choice depends on sparsity, dimensions, penalties, and resource limits.

An added $\lambda\|w\|^2/2$ is one declared L2 convention, not an automatic implementation default.
State whether the intercept is penalized and scale features using training-only statistics.
Control feature engineering and leakage when comparing trees or nonlinear models.
Use [the separate derivation](/questions/derive-logistic-regression/) for the Bernoulli gradient cancellation.

</div>
