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

Expectation describes the average value of a random variable under its probability distribution. Variance describes spread around that average. Covariance describes how two variables move together, and correlation rescales covariance to a unit-free value between -1 and 1.

These quantities support loss analysis, uncertainty estimates, feature analysis, Monte Carlo methods, and multivariate models. They describe association under a distribution. They do not prove that changing one variable causes another to change.

## Expectation

For a discrete random variable $X$ with probability mass function $p(x)$,

$$
\mathbb{E}[X] = \sum_x x p(x).
$$

For a continuous variable with density $p(x)$,

$$
\mathbb{E}[X] = \int x p(x)\,dx.
$$

Expectation is linear. For constants $a$ and $b$,

$$
\mathbb{E}[aX+bY] = a\mathbb{E}[X] + b\mathbb{E}[Y].
$$

This property does not require $X$ and $Y$ to be independent.

A useful warning: an expected value need not be a likely outcome. The expected value of one fair die roll is 3.5, although 3.5 can never appear.

## Variance

Variance is the expected squared distance from the mean:

$$
\operatorname{Var}(X) = \mathbb{E}[(X-\mathbb{E}[X])^2].
$$

The computational form is

$$
\operatorname{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2.
$$

For constants $a$ and $b$,

$$
\operatorname{Var}(aX+b) = a^2\operatorname{Var}(X).
$$

Adding a constant changes the mean but not the variance. Scaling by $a$ scales standard deviation by $|a|$ and variance by $a^2$.

## Covariance

Covariance measures linear co-movement:

$$
\operatorname{Cov}(X,Y) = \mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])].
$$

It also has a computational form:

$$
\operatorname{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y].
$$

<!-- visual:covariance-centered-product-signs -->
<figure class="learning-figure plot-panel" aria-labelledby="covariance-signs-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="covariance-signs-title">Determine each observation's covariance sign from its position relative to both means.</p>
	<svg viewBox="0 0 480 330" role="img" aria-labelledby="covariance-signs-svg-title covariance-signs-svg-desc">
		<title id="covariance-signs-svg-title">Signs of centered-product contributions to covariance</title>
		<desc id="covariance-signs-svg-desc">A scatter plot is divided by the mean of X and the mean of Y. Circular points in the upper-right and lower-left regions have same-sign deviations and positive centered products. Diamond points in the upper-left and lower-right regions have opposite-sign deviations and negative centered products. Covariance averages all of these signed products.</desc>
		<rect x="55" y="30" width="370" height="240" rx="8" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect>
		<path class="viz-axis" d="M55 150H425M240 30V270"></path>
		<path d="M235 38L240 30L245 38M417 145L425 150L417 155" style="fill:none;stroke:var(--c-text);stroke-width:1.8"></path>
		<text class="viz-axis-label" x="438" y="155" style="font-size:15px">X</text>
		<text class="viz-axis-label" x="246" y="20" style="font-size:15px">Y</text>
		<text class="viz-axis-label" x="240" y="291" text-anchor="middle" style="font-size:15px">mean of X</text>
		<text class="viz-axis-label" x="62" y="143" style="font-size:15px">E[Y]</text>
		<text class="viz-callout" x="332" y="54" text-anchor="middle" style="font-size:15px">same signs: (+)(+) = +</text>
		<text class="viz-callout" x="148" y="54" text-anchor="middle" style="font-size:15px">opposite signs: (-)(+) = -</text>
		<text class="viz-callout" x="148" y="254" text-anchor="middle" style="font-size:15px">same signs: (-)(-) = +</text>
		<text class="viz-callout" x="332" y="254" text-anchor="middle" style="font-size:15px">opposite signs: (+)(-) = -</text>
		<circle cx="315" cy="90" r="12" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
		<text class="viz-axis-label" x="315" y="95" text-anchor="middle" style="font-size:15px">+</text>
		<circle cx="365" cy="118" r="12" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
		<text class="viz-axis-label" x="365" y="123" text-anchor="middle" style="font-size:15px">+</text>
		<circle cx="120" cy="205" r="12" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
		<text class="viz-axis-label" x="120" y="210" text-anchor="middle" style="font-size:15px">+</text>
		<circle cx="185" cy="230" r="12" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
		<text class="viz-axis-label" x="185" y="235" text-anchor="middle" style="font-size:15px">+</text>
		<path d="M135 83L147 95L135 107L123 95Z" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:2"></path>
		<text class="viz-axis-label" x="135" y="100" text-anchor="middle" style="font-size:15px">-</text>
		<path d="M190 105L202 117L190 129L178 117Z" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:2"></path>
		<text class="viz-axis-label" x="190" y="122" text-anchor="middle" style="font-size:15px">-</text>
		<path d="M305 190L317 202L305 214L293 202Z" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:2"></path>
		<text class="viz-axis-label" x="305" y="207" text-anchor="middle" style="font-size:15px">-</text>
		<path d="M375 207L387 219L375 231L363 219Z" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:2"></path>
		<text class="viz-axis-label" x="375" y="224" text-anchor="middle" style="font-size:15px">-</text>
		<text class="viz-callout" x="240" y="317" text-anchor="middle" style="font-size:15px">Cov(X, Y) averages every signed centered product.</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> Center both variables at their means. A point on the same side of both means multiplies two deviations with the same sign and contributes positively; a point on opposite sides contributes negatively. Covariance is the average of these signed products, so the contributions can cancel.</figcaption>
</figure>

Positive covariance means large values of one variable tend to occur with large values of the other. Negative covariance means they tend to move in opposite directions. Zero covariance means there is no linear association.

Independence implies zero covariance when the expectations exist. Zero covariance does not imply independence. For example, let $X$ be symmetric around zero and let $Y=X^2$. Their covariance is zero, but $Y$ is fully determined by $X$.

## Variance of a sum

For two random variables,

$$
\operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X,Y).
$$

For independent variables, the covariance term is zero. This is why averaging independent measurements reduces variance.

If $X_1,\ldots,X_n$ are independent with variance $\sigma^2$, then the sample mean has variance

$$
\operatorname{Var}(\bar{X}) = \frac{\sigma^2}{n}.
$$

The standard error therefore falls as $1/\sqrt{n}$, not $1/n$.

## Correlation

Pearson correlation standardizes covariance:

$$
\rho_{X,Y} = \frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}.
$$

Correlation is unit-free and invariant to positive rescaling. It measures linear association. A value near zero can hide a strong nonlinear relationship, and an extreme outlier can change the estimate sharply.

Use a scatter plot with the coefficient. For monotonic but nonlinear association, Spearman rank correlation may be more informative.

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

This links covariance to principal component analysis, Gaussian models, whitening, and uncertainty ellipsoids.

## Worked example

Suppose $X$ and $Y$ both have variance 4 and covariance 3.

$$
\operatorname{Var}(X+Y)=4+4+2(3)=14.
$$

If they were independent, the variance would be 8. Positive covariance makes their sum less stable because both variables tend to move in the same direction.

## In an interview

Use this order:

1. Define expectation, variance, and covariance.
2. State linearity of expectation.
3. Derive the variance of a sum.
4. Explain independence versus zero covariance.
5. Describe correlation as standardized linear association.
6. Connect the covariance matrix to variance along a direction.

A common follow-up asks why the standard error falls as $1/\sqrt{n}$. Start from the variance of an average of independent variables.

## Common mistakes

- Saying zero correlation means independence.
- Treating correlation as evidence of intervention effects.
- Forgetting the covariance term in the variance of a sum.
- Saying variance scales linearly with the units of $X$.
- Comparing covariances across variables with very different units.
- Using Pearson correlation without checking for nonlinearity or outliers.

## Practice next

Use these quantities in [bias and variance of estimators](/concepts/bias-variance-of-estimators/), [SVD and PCA](/concepts/svd-and-pca/), and the [ML math oral](/prep/labs/math-oral/).
