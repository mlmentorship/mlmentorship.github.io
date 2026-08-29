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
