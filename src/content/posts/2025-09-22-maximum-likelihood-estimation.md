---
title: "Maximum likelihood estimation"
description: "Compare parameter values using fixed observed data, derive a Bernoulli estimate, and distinguish likelihood, priors, and asymptotic guarantees."
date: "2025-09-22"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Maximum likelihood chooses a parameter that maximizes the probability or density assigned to the observed data.
The data stay fixed while candidate parameters vary.
For independent observations with a common model $p(x\mid\theta)$,

$$
\hat\theta_\text{MLE} = \arg\max_\theta \prod_{i=1}^{n} p(x_i \mid \theta) = \arg\max_\theta \sum_{i=1}^{n} \log p(x_i \mid \theta).
$$

The logarithm preserves the maximizer while turning products into sums.
For continuous observations, the factors are densities, not probabilities of exact values.
Dependent data require their joint or conditional likelihood rather than an unjustified independent product.

For seven successes and three failures under an independent Bernoulli model,

$$
L(p)=p^7(1-p)^3,\qquad
\ell'(p)=\frac7p-\frac3{1-p}.
$$

Setting the derivative to zero gives $7(1-p)=3p$, hence $\widehat p=0.7$.
The second derivative is negative for $0<p<1$, so this is the unique maximum for these mixed outcomes.
The diagram rescales likelihood by its peak only to make the shape readable.

<!-- visual:mle-bernoulli-parameter-sweep -->
<figure class="learning-figure plot-panel" aria-labelledby="mle-sweep-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="mle-sweep-title">Hold the observations fixed while comparing candidate probabilities</p><div class="visual-argument" data-visual-argument="mle-bernoulli-parameter-sweep" data-argument-kind="curve" data-argument-composition="landmark"><svg viewBox="0 0 360 377" role="img" aria-label="Keep the data fixed; vary the candidate probability. For three successes in five independent trials, likelihood is p^3(1-p)^2. Its maximum occurs at p=3/5, not at a changing data sample."><title>Keep the data fixed; vary the candidate probability</title><desc>For three successes in five independent trials, likelihood is p^3(1-p)^2. Its maximum occurs at p=3/5, not at a changing data sample.</desc><g opacity=".35"><path d="M48 177.5L330 177.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 135L330 135" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 92.5L330 92.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">0</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">1</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0.04</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0</tspan></text><path data-proof-landmark="0.6" d="M217.2 40V220" style="stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:2 4"></path><text x="217.2" y="245" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="217.2" dy="0">0.6</tspan></text><path data-proof-series="Three successes; two failures" d="M48 220L52.40625 219.98429021798074L56.8125 219.87827986478806L61.21875 219.60233946330845L65.625 219.08804893493652L70.03125 218.27772262506187L74.4375 217.12393432855606L78.84375 215.58904231525958L83.25 213.64471435546875L87.65625 211.27145274542272L92.0625 208.45811933279037L96.46875 205.2014605421573L100.875 201.5056324005127L105.28125 197.3817255627364L109.6875 192.84729033708572L114.09375 187.9258617106825L118.5 182.646484375L122.90625 177.0432377513498L127.3125 171.15476101636887L131.71875 165.02377812750638L136.125 158.69662284851074L140.53125 152.22276377491653L144.9375 145.6543293595314L149.34375 139.04563293792307L153.75 132.45269775390625L158.15625 125.93278198502958L162.5625 119.54390376806259L166.96875 113.34436622448266L171.375 107.39228248596191L175.78125 101.74510071985424L180.1875 96.45912915468216L184.59375 91.58906110562384L189 87.1875L193.40625 83.30448440276086L197.8125 79.98701304197311L202.21875 77.27856983430684L206.625 75.21864891052246L211.03125 73.84227964095771L215.4375 73.17955166101456L217.2 73.12L219.84375 73.25513989664614L224.25 74.08782958984375L228.65625 75.69004132412374L233.0625 78.0673560500145L237.46875 81.21804011054337L241.875 85.13257026672363L246.28125 89.79315872304142L250.6875 95.17327815294266L255.09375 101.23718672432005L259.5 107.939453125L263.90625 115.22448158822954L268.3125 123.0260369181633L272.71875 131.26676951535046L277.125 139.85774040222168L281.53125 148.69794624857605L285.9375 157.67384439706802L290.34375 166.6588778886944L294.75 175.51300048828125L299.15625 184.08220170997083L303.5625 192.1980318427086L307.96875 199.67712697573006L312.375 206.32073402404785L316.78125 211.91423575393856L321.1875 216.22667580842972L325.59375 219.01028373278677L330 220" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3"></path><g data-proof-observation="0" data-proof-x="0.6" data-proof-y="0.03456"><circle cx="217.2" cy="73.12" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">candidate Bernoulli probability</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">likelihood of this ordered sample</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Three successes; two failures</tspan></text><text x="60" y="333" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="60" dy="0">0.03456 at 0.6</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 330" role="img" aria-labelledby="mle-sweep-svg-title mle-sweep-svg-desc">
		<title id="mle-sweep-svg-title">Bernoulli likelihood across candidate success probabilities</title>
		<desc id="mle-sweep-svg-desc">An original diagram starts with ten fixed Bernoulli observations: seven ones and three zeros. For each candidate success probability p, the likelihood is p to the seventh power times one minus p to the third power. A solid relative-likelihood curve rises from zero, reaches its unique maximum at p equals seven tenths, the observed fraction of successes, then returns to zero. A vertical guide and labelled point identify the maximum likelihood estimate. The curve is a score as p varies and is not a probability distribution over p.</desc>
		<rect class="viz-plot-bg" x="4" y="4" width="352" height="322" rx="6"></rect>
		<text class="viz-axis-label" x="16" y="25">fixed observations D</text>
		<g aria-label="Seven successes followed by three failures">
			<rect class="viz-node viz-node--focus" x="16" y="36" width="25" height="25" rx="3"></rect><text class="viz-node-label" x="28.5" y="53">1</text>
			<rect class="viz-node viz-node--focus" x="48" y="36" width="25" height="25" rx="3"></rect><text class="viz-node-label" x="60.5" y="53">1</text>
			<rect class="viz-node viz-node--focus" x="80" y="36" width="25" height="25" rx="3"></rect><text class="viz-node-label" x="92.5" y="53">1</text>
			<rect class="viz-node viz-node--focus" x="112" y="36" width="25" height="25" rx="3"></rect><text class="viz-node-label" x="124.5" y="53">1</text>
			<rect class="viz-node viz-node--focus" x="144" y="36" width="25" height="25" rx="3"></rect><text class="viz-node-label" x="156.5" y="53">1</text>
			<rect class="viz-node viz-node--focus" x="176" y="36" width="25" height="25" rx="3"></rect><text class="viz-node-label" x="188.5" y="53">1</text>
			<rect class="viz-node viz-node--focus" x="208" y="36" width="25" height="25" rx="3"></rect><text class="viz-node-label" x="220.5" y="53">1</text>
			<rect class="viz-node" x="240" y="36" width="25" height="25" rx="3" style="stroke-dasharray:4 3"></rect><text class="viz-node-label" x="252.5" y="53">0</text>
			<rect class="viz-node" x="272" y="36" width="25" height="25" rx="3" style="stroke-dasharray:4 3"></rect><text class="viz-node-label" x="284.5" y="53">0</text>
			<rect class="viz-node" x="304" y="36" width="25" height="25" rx="3" style="stroke-dasharray:4 3"></rect><text class="viz-node-label" x="316.5" y="53">0</text>
		</g>
		<text class="viz-callout" x="16" y="82">candidate p → L(p; D) = p⁷(1 − p)³</text>
		<path class="viz-gridline" d="M45 170H340M45 222.5H340M45 275H340M45 170V275M192.5 170V275M340 170V275"></path>
		<path class="viz-axis" d="M45 170V275H340"></path>
		<path class="viz-roc-curve" d="M45 275 L59.8 275 L74.5 275 L89.3 275 L104 274.7 L118.8 273.8 L133.5 271.5 L148.3 266.7 L163 258.3 L177.8 245.6 L192.5 228.9 L207.2 209.5 L222 190.4 L236.8 175.7 L251.5 170 L266.3 176.5 L281 195.8 L295.8 223.9 L310.5 252.4 L325.3 270.9 L340 275"></path>
		<path class="viz-operating-guide" d="M251.5 170V275" style="stroke-dasharray:5 4"></path>
		<circle class="viz-operating-point" cx="251.5" cy="170" r="5"></circle>
		<text class="viz-callout" x="251.5" y="112" text-anchor="middle">maximum at p̂ = 7 / 10 = 0.7</text>
		<text class="viz-label" x="37" y="174" text-anchor="end">1</text>
		<text class="viz-label" x="37" y="226.5" text-anchor="end">0.5</text>
		<text class="viz-label" x="37" y="279" text-anchor="end">0</text>
		<text class="viz-label" x="45" y="294" text-anchor="middle">0</text>
		<text class="viz-label" x="192.5" y="294" text-anchor="middle">0.5</text>
		<text class="viz-label" x="251.5" y="294" text-anchor="middle">0.7</text>
		<text class="viz-label" x="340" y="294" text-anchor="middle">1</text>
		<text class="viz-axis-label" x="45" y="154">relative likelihood</text>
		<text class="viz-axis-label" x="192.5" y="316" text-anchor="middle">candidate success probability p</text>
	</svg>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> keep the observations fixed while varying <code>p</code>. Seven successes contribute <code>p⁷</code>; three failures contribute <code>(1 − p)³</code>. Their product peaks at <code>0.7</code>. Relative likelihood is a parameter score, not a normalized parameter distribution.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="mle-bernoulli-parameter-sweep"><strong>Check:</strong> What changes as the marker moves along the curve? Why is the curve not by itself a posterior distribution over p?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="mle-bernoulli-parameter-sweep">For three successes in five independent trials, likelihood is p^3(1-p)^2. Its maximum occurs at p=3/5, not at a changing data sample.</span></figcaption>
</figure>

## Properties

Under appropriate identification, sampling, regularity, and existence conditions, MLE can be consistent and asymptotically normal:

$$
\sqrt n(\widehat\theta-\theta^*)\Rightarrow\mathcal N(0,I(\theta^*)^{-1}).
$$

Here $I$ is Fisher information per observation and must be nonsingular for this expression.
Efficiency statements compare regular estimators under their stated asymptotic framework; they do not rule out every pointwise superefficient estimator.
Small samples, boundary parameters, growing dimension, or misspecification require separate analysis.

## Common cases

| Model | MLE solution |
|-------|-------------|
| Gaussian mean (known $\sigma^2$) | $\hat\mu = \bar x$ |
| Gaussian variance | $\hat\sigma^2 = \frac{1}{n} \sum (x_i - \bar x)^2$ (biased; sample-variance uses $n-1$) |
| Bernoulli ($p$) | $\hat p = \text{fraction of successes}$ |
| Categorical | empirical class frequencies |
| Linear regression with independent equal-variance Gaussian noise | Least squares; the displayed inverse formula needs full column rank |
| Logistic regression | Iterative optimization; complete separation can prevent a finite optimum |

## Connection to cross-entropy and KL

For categorical $p_\theta(x)$ and empirical distribution $\hat p_\text{data}$, the negative log-likelihood divided by $n$ is

$$
-\tfrac{1}{n} \sum_i \log p_\theta(x_i) = H(\hat p_\text{data}, p_\theta) = H(\hat p_\text{data}) + \mathrm{KL}(\hat p_\text{data} \| p_\theta).
$$

The empirical entropy is constant as model parameters vary, so both objectives select the same optimum.
Gaussian regression produces squared-error minimization when its noise-variance and independence assumptions are appropriate.

## MLE vs MAP

### Add Prior Evidence

**New task:** Incorporate a declared prior rather than reporting only the likelihood maximizer.

**Picture and change:** A $\operatorname{Beta}(2,2)$ prior multiplies the Bernoulli likelihood by a factor proportional to $p(1-p)$.
The posterior is $\operatorname{Beta}(9,5)$.

**Check:** Its interior mode is $8/12=2/3$, not the MLE $0.7$; its mean is $9/14$.

**Use when:** Prior assumptions belong in the model.
MAP maximizes posterior density in the chosen parameterization; it is not generally invariant under reparameterization.
Gaussian and Laplace parameter priors produce quadratic and absolute-value penalties under their specified scales.

### Change the Observation Unit

**New task:** Fit a binomial count of seven successes rather than one particular ordered sequence.

**Picture and change:** Multiply by $\binom{10}{7}$ to account for compatible sequences.

**Check:** The factor equals $120$ and does not depend on $p$, so the maximizer remains $0.7$.

**Use when:** Aggregation changes the recorded observation but preserves the model's sufficient counts.
If dependence changes instead, multiplying independent factors need not be valid.

<!-- learning:core-end -->
<div data-learning-extension>

## Common pitfalls

- **Wrong variance denominator.** With an unknown Gaussian mean, the variance MLE divides residual squares by $n$.
	The unbiased sample-variance estimator uses $n-1$ under its assumptions.
- **Missing identification.** Parameters producing the same observation distribution need not have unique estimates.
- **Missing existence.** All-success Bernoulli data maximize at $p=1$ if the boundary is allowed.
	Restricting the parameter to $(0,1)$ leaves only a supremum.
- **Unqualified misspecification claims.** Suitable convergence conditions may select a KL-best approximation, not the true data-generating mechanism.

## Reconstruct

<details>
<summary>Is the plotted likelihood a posterior over the parameter?</summary>

No. It scores fixed data across parameter values and has not included a prior or normalized over parameters.
A constant rescaling preserves the likelihood maximizer; a nonconstant prior can change the posterior mode.

</details>

</div>
