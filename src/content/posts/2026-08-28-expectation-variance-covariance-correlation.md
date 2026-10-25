---
title: "Expectation, variance, covariance, and correlation"
description: "Use moments to describe location, uncertainty, and dependence. Know what covariance measures, how transformations change it, and why correlation does not establish causation."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["expected value", "variance covariance", "covariance matrix", "pearson correlation"]
roles: ["Applied Scientist", "Research Scientist", "Research Engineer", "Machine Learning Engineer"]
rounds: ["Math", "Statistics", "ML breadth"]
difficulty: "Foundation"
priority: "Core"
prerequisites: []
---

## Summary

Expectation averages values under a probability law.
Variance averages squared deviations from a mean. Covariance averages products of two variables' deviations.

Suppose a hypothetical population gives probability $1/4$ to each pair $(-1,-1)$, $(-1,1)$, $(1,1)$ and $(1,3)$.
How does their co-movement affect the spread of $X+Y$?
These are population probabilities, not four observations used to estimate an unknown population.

## Covariance

Center both coordinates: the mean of $X$ is zero and the mean of $Y$ is one.
Multiply each point's horizontal and vertical deviations, then average those products.

<!-- visual:covariance-centered-product-signs -->
<figure class="learning-figure plot-panel" aria-labelledby="covariance-signs-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="covariance-signs-title">Multiply deviations from both means, then average the four products.</p>
	<svg viewBox="0 0 480 330" role="img" aria-labelledby="covariance-signs-svg-title covariance-signs-svg-desc">
		<title id="covariance-signs-svg-title">Signs of centered-product contributions to covariance</title>
		<desc id="covariance-signs-svg-desc">Four equally likely outcomes have X mean zero and Y mean one. Points (-1, -1) and (1, 3) each contribute a centered product of two. Points (-1, 1) and (1, 1) lie on the Y-mean line and contribute zero. Their average product is one. Empty quadrants show where opposite-sign deviations would contribute negatively.</desc>
		<rect x="55" y="30" width="370" height="240" rx="8" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect>
		<path class="viz-axis" d="M55 150H425M240 30V270"></path>
		<path d="M235 38L240 30L245 38M417 145L425 150L417 155" style="fill:none;stroke:var(--c-text);stroke-width:1.8"></path>
		<text class="viz-axis-label" x="438" y="155" style="font-size:15px">X</text>
		<text class="viz-axis-label" x="246" y="20" style="font-size:15px">Y</text>
		<text class="viz-axis-label" x="240" y="291" text-anchor="middle" style="font-size:15px">mean of X = 0</text>
		<text class="viz-axis-label" x="62" y="143" style="font-size:15px">mean of Y = 1</text>
		<text class="viz-callout" x="332" y="54" text-anchor="middle" style="font-size:15px">same signs: (+)(+) = +</text>
		<text class="viz-callout" x="148" y="54" text-anchor="middle" style="font-size:15px">opposite signs: (-)(+) = -</text>
		<text class="viz-callout" x="148" y="254" text-anchor="middle" style="font-size:15px">same signs: (-)(-) = +</text>
		<text class="viz-callout" x="332" y="254" text-anchor="middle" style="font-size:15px">opposite signs: (+)(-) = -</text>
		<circle cx="305" cy="80" r="12" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
		<text class="viz-axis-label" x="305" y="85" text-anchor="middle" style="font-size:15px">2</text>
		<text class="viz-axis-label" x="323" y="85" style="font-size:15px">(1, 3)</text>
		<circle cx="175" cy="220" r="12" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
		<text class="viz-axis-label" x="175" y="225" text-anchor="middle" style="font-size:15px">2</text>
		<text class="viz-axis-label" x="91" y="225" style="font-size:15px">(-1, -1)</text>
		<path d="M175 138L187 150L175 162L163 150Z" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:2"></path>
		<text class="viz-axis-label" x="175" y="155" text-anchor="middle" style="font-size:15px">0</text>
		<text class="viz-axis-label" x="130" y="181" style="font-size:15px">(-1, 1)</text>
		<path d="M305 138L317 150L305 162L293 150Z" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:2"></path>
		<text class="viz-axis-label" x="305" y="155" text-anchor="middle" style="font-size:15px">0</text>
		<text class="viz-axis-label" x="283" y="181" style="font-size:15px">(1, 1)</text>
		<text class="viz-callout" x="240" y="317" text-anchor="middle" style="font-size:15px">Cov(X, Y) averages every signed centered product.</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> each point has probability one quarter. Two centered products equal two; two equal zero. Their average is one. Opposite-sign deviations would contribute negatively.</figcaption>
</figure>

## Expectation

For probability mass $p(x)$, $\mathbb E[X]=\sum_x xp(x)$.
For a density $p(x)$, replace the sum by $\int xp(x)\,dx$.
Assume finite expectations. For constants $a,b$, linearity gives

$$
\mathbb E[aX+bY]=a\mathbb E[X]+b\mathbb E[Y].
$$

No independence is needed. An expectation need not be a possible outcome: a fair die has expectation $3.5$.

## Variance

Let $\mu_X=\mathbb E[X]$ and $\mu_Y=\mathbb E[Y]$. Assume finite second moments.
Variance squares one deviation; covariance multiplies two:

$$
\operatorname{Var}(X)=\mathbb E[(X-\mu_X)^2],\qquad
\operatorname{Cov}(X,Y)=\mathbb E[(X-\mu_X)(Y-\mu_Y)].
$$

Expanding these expressions gives $\operatorname{Var}(X)=\mathbb E[X^2]-\mu_X^2$ and $\operatorname{Cov}(X,Y)=\mathbb E[XY]-\mu_X\mu_Y$.

## Worked example

| Outcome | Probability | $X-0$ | $Y-1$ | Centered product |
| --- | --- | --- | --- | --- |
| $(-1,-1)$ | $1/4$ | $-1$ | $-2$ | $2$ |
| $(-1,1)$ | $1/4$ | $-1$ | $0$ | $0$ |
| $(1,1)$ | $1/4$ | $1$ | $0$ | $0$ |
| $(1,3)$ | $1/4$ | $1$ | $2$ | $2$ |

The means are $0$ and $1$. The second moments are $1$ and $3$.
Thus $\operatorname{Var}(X)=1$, $\operatorname{Var}(Y)=3-1^2=2$ and $\operatorname{Cov}(X,Y)=1$.

## Variance of a sum

Square $(X-\mu_X)+(Y-\mu_Y)$, then average. The two cross-products produce the covariance term:

$$
\operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X,Y).
$$

Here the variance is $1+2+2(1)=5$.
Direct check: $X+Y$ takes $-2,0,2,4$, with mean $1$.
Its squared deviations are $9,1,1,9$, averaging $5$.
Removing dependence while retaining the marginals would instead give sum variance $3$.

## Correlation

Pearson correlation divides covariance by positive standard deviations $\sigma_X$ and $\sigma_Y$:

$$
\rho_{X,Y} = \frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}.
$$

Here $\rho=1/\sqrt2\approx0.707107$. It is undefined if either variance is zero.
Correlation lies between $-1$ and $1$ and does not establish an intervention effect.

## Change the measurement

### 1. Convert the units of Y

**New task:** replace $Y$ by $100Y$ without changing the joint observations.

**Picture and change:** each vertical deviation grows by $100$.
Covariance scales once; $Y$ variance scales twice.

**Check:** covariance becomes $100$ and variance becomes $20{,}000$.
Correlation stays $1/\sqrt2$. A negative rescaling would reverse its sign.

**Use when:** units change but the underlying association does not.

### 2. Average correlated measurements

**New task:** average four variables, each with variance $4$ and pairwise correlation $0.5$.

**Picture and change:** retain all six pairwise covariances, each equal to $2$.
Divide the full sum variance by $4^2$.

**Check:** average variance is $(16+2\times12)/16=2.5$, not the independent value $1$.

**Use when:** measurements share a user, environment or disturbance.

## Zero covariance can hide dependence

Let $X$ be equally likely to be $-1,0,1$, and let $Y=X^2$.
Both $\mathbb E[X]$ and $\mathbb E[XY]$ are zero, so covariance is zero.
Yet $Y$ is determined by $X$. Independence implies zero covariance under these moment conditions; the reverse implication fails.

## Try averaging two independent quantities

Independent variables have variances $9$ and $16$. Find the variance and standard deviation of their average.

<details>
<summary>Check your answer</summary>

The covariance term is zero. Variance is $(9+16)/4=6.25$, and standard deviation is $2.5$.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Scaling and repeated measurements

For constants $a,b$, $\operatorname{Var}(aX+b)=a^2\operatorname{Var}(X)$.
Translation changes the mean but not spread. Standard deviation scales by $|a|$.
More generally,

$$
\operatorname{Var}(aX+bY)=a^2\operatorname{Var}(X)+b^2\operatorname{Var}(Y)+2ab\operatorname{Cov}(X,Y).
$$

For independent measurements with common variance $\sigma^2$, their mean has variance $\sigma^2/n$.
Its standard error falls as $1/\sqrt n$, not $1/n$.
As another check, variances $4,4$ and covariance $3$ give sum variance $4+4+2(3)=14$.

## Covariance matrices

For a random vector $x \in \mathbb{R}^d$ with mean $\mu$, the covariance matrix is

$$
\Sigma = \mathbb{E}[(x-\mu)(x-\mu)^\top].
$$

The diagonal contains feature variances. Entry $(i,j)$ contains the covariance between features $i$ and $j$.

A covariance matrix is symmetric and positive semidefinite. For any vector $v$,

$$
v^\top\Sigma v = \operatorname{Var}(v^\top x) \ge 0.
$$

This supports PCA, Gaussian models, whitening and uncertainty ellipsoids.
Use a scatter plot alongside Pearson correlation: outliers can change it sharply.
Spearman correlation instead measures association between ranks; it answers a different question about monotonic relationships.

## In an interview

Reconstruct the sum-variance formula by centering and squaring.
State the observation law, moment conditions and independence assumptions before simplifying it.

## Common mistakes

- Saying zero correlation means independence.
- Treating correlation as evidence of intervention effects.
- Forgetting the covariance term in the variance of a sum.
- Saying variance scales linearly with the units of $X$.
- Comparing covariances across variables with very different units.
- Using Pearson correlation without checking for nonlinearity or outliers.

## Practice next

Use these quantities in [bias and variance of estimators](/concepts/bias-variance-of-estimators/), [SVD and PCA](/concepts/svd-and-pca/), and the [ML math oral](/prep/labs/math-oral/).

</div>
