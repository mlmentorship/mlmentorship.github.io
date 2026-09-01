---
title: "Probability distributions used in ML"
description: "Choose a distribution from the data type and generation process. Connect its support, parameters, mean, variance, likelihood, and common ML use."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["common probability distributions", "Bernoulli Gaussian Poisson", "Beta Dirichlet", "distribution choice"]
roles: ["Applied Scientist", "Research Scientist", "Research Engineer"]
rounds: ["Math", "Statistics", "ML breadth"]
difficulty: "Foundation"
priority: "Core"
prerequisites: ["expectation-variance-covariance-correlation"]
---

## Summary

Choose a probability distribution from the values that can occur and the process that generates them. The support rules out impossible values. The parameters describe location, spread, rate, or class probabilities. The likelihood then gives a training objective.

Binary outcomes suggest Bernoulli models. Counts often suggest binomial or Poisson models. Real-valued residuals may support Gaussian or Laplace assumptions. Probabilities and class mixtures often use Beta or Dirichlet distributions.

## A practical selection rule

Ask five questions:

1. Is the value binary, categorical, a count, positive, bounded, or real-valued?
2. Is one observation a single trial, a total over trials, or a waiting time?
3. Are events independent at the chosen unit?
4. Does variance grow with the mean?
5. Do the tails and zero frequency match the proposed model?

A familiar distribution with wrong support is already wrong. A Gaussian model can assign negative probability to a quantity that must be positive. A Poisson model cannot represent count data whose variance is far above its mean without an extension.

<!-- visual:distribution-observation-unit -->
<figure class="learning-figure" aria-labelledby="distribution-unit-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="distribution-unit-title">Choose among Bernoulli, binomial, Poisson, and exponential models by defining what one observation records.</p>
	<div class="visual-panel">
		<svg viewBox="0 0 360 590" role="img" aria-labelledby="distribution-unit-svg-title distribution-unit-svg-desc">
			<title id="distribution-unit-svg-title">Observation units select different distributions for trial and time processes</title>
			<desc id="distribution-unit-svg-desc">The first panel starts with five independent visits that share conversion probability p and have outcomes one, zero, one, zero, zero. Recording one visit gives a binary value with support zero or one and a Bernoulli p model. Recording the total from all five visits gives a count with support zero through five and a binomial five comma p model. The second panel starts with a constant-rate independent event process marked by three events on a timeline. Recording the number in a ten-minute window gives a nonnegative integer and a Poisson ten lambda model. Recording the time until the next event gives a nonnegative real value and an exponential lambda model. Numbered choices, explicit supports, process assumptions, distinct box outlines, and arrows communicate the selection without relying on color.</desc>
			<defs>
				<marker id="distribution-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0 0L7 3.5L0 7Z"></path></marker>
			</defs>
			<rect class="viz-plot-bg" x="12" y="12" width="336" height="270" rx="4"></rect>
			<text class="viz-callout" x="24" y="36">A · Trial process: independent visits, same p</text>
			<text class="viz-label" x="24" y="57">five visit outcomes</text>
			<rect class="viz-node" x="32" y="70" width="42" height="34" rx="4"></rect>
			<rect class="viz-node" x="94" y="70" width="42" height="34" rx="4"></rect>
			<rect class="viz-node" x="156" y="70" width="42" height="34" rx="4"></rect>
			<rect class="viz-node" x="218" y="70" width="42" height="34" rx="4"></rect>
			<rect class="viz-node" x="280" y="70" width="42" height="34" rx="4"></rect>
			<text class="viz-node-label" x="53" y="93">1</text>
			<text class="viz-node-label" x="115" y="93">0</text>
			<text class="viz-node-label" x="177" y="93">1</text>
			<text class="viz-node-label" x="239" y="93">0</text>
			<text class="viz-node-label" x="301" y="93">0</text>
			<path class="viz-baseline" d="M177 104V122H94V140" marker-end="url(#distribution-arrow)"></path>
			<path class="viz-baseline" d="M177 122H266V140" marker-end="url(#distribution-arrow)"></path>
			<rect class="viz-node viz-node--focus" x="24" y="142" width="140" height="112" rx="4"></rect>
			<text class="viz-callout" x="94" y="164" text-anchor="middle">1 · Record one visit</text>
			<text class="viz-node-value" x="94" y="187">X ∈ {0, 1}</text>
			<text class="viz-node-label" x="94" y="214">Bernoulli(p)</text>
			<text class="viz-node-value" x="94" y="237">one binary outcome</text>
			<rect class="viz-node viz-node--focus" x="196" y="142" width="140" height="112" rx="4"></rect>
			<text class="viz-callout" x="266" y="164" text-anchor="middle">2 · Total five visits</text>
			<text class="viz-node-value" x="266" y="187">S ∈ {0, 1, …, 5}</text>
			<text class="viz-node-label" x="266" y="214">Binomial(5, p)</text>
			<text class="viz-node-value" x="266" y="237">success count</text>
			<rect class="viz-plot-bg" x="12" y="298" width="336" height="280" rx="4"></rect>
			<text class="viz-callout" x="24" y="322">B · Events: independent, constant rate λ</text>
			<path class="viz-axis" d="M38 372H322"></path>
			<path class="viz-operating-guide" d="M54 354V390 M306 354V390"></path>
			<circle class="viz-operating-point" cx="103" cy="372" r="6"></circle>
			<circle class="viz-operating-point" cx="177" cy="372" r="6"></circle>
			<circle class="viz-operating-point" cx="268" cy="372" r="6"></circle>
			<text class="viz-label" x="54" y="407" text-anchor="middle">0 min</text>
			<text class="viz-label" x="306" y="407" text-anchor="middle">10 min</text>
			<text class="viz-label" x="177" y="348" text-anchor="middle">three event times</text>
			<path class="viz-baseline" d="M177 408V426H94V444" marker-end="url(#distribution-arrow)"></path>
			<path class="viz-baseline" d="M177 426H266V444" marker-end="url(#distribution-arrow)"></path>
			<rect class="viz-node viz-node--focus" x="24" y="446" width="140" height="112" rx="4"></rect>
			<text class="viz-callout" x="94" y="468" text-anchor="middle">3 · Count in window</text>
			<text class="viz-node-value" x="94" y="491">K ∈ {0, 1, 2, …}</text>
			<text class="viz-node-label" x="94" y="518">Poisson(10λ)</text>
			<text class="viz-node-value" x="94" y="541">events per 10 min</text>
			<rect class="viz-node viz-node--focus" x="196" y="446" width="140" height="112" rx="4"></rect>
			<text class="viz-callout" x="266" y="468" text-anchor="middle">4 · Wait to next event</text>
			<text class="viz-node-value" x="266" y="491">T ∈ [0, ∞)</text>
			<text class="viz-node-label" x="266" y="518">Exponential(λ)</text>
			<text class="viz-node-value" x="266" y="541">continuous duration</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> choose the process first, then draw a box around exactly what one training row records. For repeated visits with the same independent success probability, one outcome is Bernoulli while a fixed-size total is binomial. For independent constant-rate events, a fixed-time count is Poisson while the wait to the next event is exponential. The distribution follows the observation unit and assumptions, not the noun “conversion” or “arrival.”</figcaption>
</figure>
<p class="diagram-source">Original process-and-observation construction checked against the open <a href="https://openstax.org/books/introductory-statistics-2e/pages/4-3-binomial-distribution">OpenStax binomial</a>, <a href="https://openstax.org/books/introductory-statistics-2e/pages/4-6-poisson-distribution">Poisson</a>, and <a href="https://openstax.org/books/introductory-statistics-2e/pages/5-3-the-exponential-distribution">exponential</a> textbook sections and the <a href="https://www.itl.nist.gov/div898/handbook/eda/section3/eda366j.htm">NIST Poisson reference</a>.</p>

## Bernoulli and binomial

A Bernoulli variable $X \in \{0,1\}$ has

$$
P(X=1)=p, \qquad P(X=0)=1-p.
$$

Its mean is $p$ and variance is $p(1-p)$. Binary cross-entropy is the negative Bernoulli log-likelihood.

A binomial variable counts successes in $n$ independent Bernoulli trials with the same probability $p$:

$$
X \sim \operatorname{Binomial}(n,p).
$$

Use Bernoulli for one conversion and binomial for the number of conversions among a fixed number of comparable opportunities. Dependence, changing probabilities, or varying exposure can break the binomial assumptions.

## Categorical and multinomial

A categorical variable selects one of $K$ classes with probabilities $p_1,\ldots,p_K$. Softmax cross-entropy is the negative categorical log-likelihood.

A multinomial variable counts how often each class appears across $n$ categorical trials. The counts sum to $n$.

Use categorical models for one label and multinomial models for a vector of class counts. A multi-label problem is different because several labels can be true at once.

## Gaussian and Laplace

A Gaussian variable has support over the real line:

$$
X \sim \mathcal{N}(\mu,\sigma^2).
$$

It is a common model for accumulated noise and averages. Maximizing a Gaussian likelihood with fixed variance gives mean squared error.

A Laplace distribution has sharper center mass and heavier tails than a Gaussian. Maximizing a Laplace likelihood with fixed scale gives mean absolute error.

This connection turns loss selection into a noise-model choice:

| Residual model | Training loss | Main behavior |
| --- | --- | --- |
| Gaussian | squared error | large errors receive strong weight |
| Laplace | absolute error | less sensitive to large errors |

Inspect residuals instead of assuming either model from habit.

## Poisson and negative binomial

A Poisson variable models a count in a fixed exposure interval:

$$
P(X=k)=\frac{\lambda^k e^{-\lambda}}{k!}, \qquad k=0,1,2,\ldots
$$

Its mean and variance both equal $\lambda$. A log link is common in regression because it keeps the predicted rate positive.

Real counts often have variance above the mean. This is overdispersion. A negative-binomial model adds dispersion and is often a better fit for clicks, incidents, or purchases with heterogeneous rates.

Always include exposure. Ten failures in one hour and ten failures in one month do not imply the same rate.

## Exponential and Gamma

The exponential distribution models a positive waiting time under a constant event rate. It is memoryless:

$$
P(T>s+t \mid T>s)=P(T>t).
$$

Use it when a constant hazard is defensible. It fails when risk changes with elapsed time.

The Gamma distribution models positive values and includes the exponential as a special case. It can represent waiting time until several events or positive right-skewed quantities.

## Beta and Dirichlet

A Beta distribution has support on $[0,1]$ and two positive shape parameters. It is a convenient prior for a Bernoulli probability:

$$
p \sim \operatorname{Beta}(\alpha,\beta).
$$

After observing $s$ successes and $f$ failures, the posterior is

$$
p \mid D \sim \operatorname{Beta}(\alpha+s,\beta+f).
$$

The Dirichlet distribution generalizes this idea to a vector of class probabilities. It is conjugate to the categorical and multinomial likelihoods.

Conjugacy means the posterior stays in the same distribution family. It gives simple updates, but it is a computational convenience rather than proof that the prior fits the problem.

## Worked example

A product records 20 conversions among 500 independent visits. A binomial likelihood fits the fixed-trial count. With a $\operatorname{Beta}(1,1)$ prior, the posterior conversion probability is

$$
p \mid D \sim \operatorname{Beta}(21,481).
$$

If visits come from repeated users with correlated behavior, the independent-trial assumption fails. The uncertainty must account for user groups.

## In an interview

For any proposed distribution, state:

1. Its support.
2. What one observation represents.
3. Its parameters and mean-variance relation.
4. The likelihood or loss it implies.
5. The assumption most likely to fail.
6. A diagnostic or alternative model.

A strong answer chooses a distribution after defining the data unit. It does not list distribution names without a generative story.

## Common mistakes

- Using Bernoulli and binomial interchangeably.
- Using Poisson without checking overdispersion or exposure.
- Treating softmax as a multi-label model.
- Using a Gaussian for strongly positive, right-skewed data without checking residuals.
- Calling conjugacy evidence that a model is true.
- Ignoring dependence among observations.

## Practice next

Connect distribution choice to [maximum likelihood estimation](/concepts/maximum-likelihood-estimation/), [the exponential family](/concepts/exponential-family/), and [loss-function selection](/questions/how-to-choose-loss-function/).
