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

A probability distribution assigns probabilities to values one observation can record.
Choose the observation unit and generating process before the family name.

Five hypothetical independent visits share conversion probability $p=0.2$.
Their outcomes are $1,0,1,0,0$.
What changes when one record contains one visit rather than the total from five visits?

## A practical selection rule

First specify the **support**: the possible recorded values.
Then ask whether the record is a trial, trial total, count during an exposure or waiting time.
Check dependence, changing rates, tail behavior and the frequency of zeros.
Correct support alone does not establish the model's assumptions.

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
	<figcaption><strong>Read it this way:</strong> decide what one row records. One visit is Bernoulli; independent visits with the same probability give a binomial total. For independent constant-rate events, fixed-window counts are Poisson and next-event waits are exponential. <span class="visual-reader-check" data-reader-check="distribution-observation-unit"><strong>Check:</strong> Why use Poisson for event counts but Exponential for waiting until the next event?</span></figcaption>
</figure>
<p class="diagram-source">Original process-and-observation construction checked against the open <a href="https://openstax.org/books/introductory-statistics-2e/pages/4-3-binomial-distribution">OpenStax binomial</a>, <a href="https://openstax.org/books/introductory-statistics-2e/pages/4-6-poisson-distribution">Poisson</a>, and <a href="https://openstax.org/books/introductory-statistics-2e/pages/5-3-the-exponential-distribution">exponential</a> textbook sections and the <a href="https://www.itl.nist.gov/div898/handbook/eda/section3/eda366j.htm">NIST Poisson reference</a>.</p>

## Bernoulli and binomial

A single-visit outcome $X\in\{0,1\}$ has a Bernoulli distribution:

$$
P(X=1)=p, \qquad P(X=0)=1-p.
$$

Its mean is $p=0.2$ and variance is $p(1-p)=0.16$.
Binary cross-entropy is its negative log-likelihood.

A total $S$ over $n$ independent visits with the same probability has a binomial distribution:

$$
S \sim \operatorname{Binomial}(n,p),\qquad
P(S=s)=\binom ns p^s(1-p)^{n-s}.
$$

Here $n=5$, so the support is $0,1,\ldots,5$.
Its mean is $np=1$ and variance is $np(1-p)=0.8$.
The observed count of two need not equal the expected count of one.

## Count the possible orders

The particular ordered sequence has probability $0.2^2\times0.8^3=0.02048$.
But ten different orders contain two successes:

$$
P(S=2)=\binom52(0.2)^2(0.8)^3=10\times0.02048=0.2048.
$$

The binomial coefficient counts those orders. Summing the six probabilities for $s=0$ through $5$ gives one.
The likelihood must match what the observation records.

## Change what a record measures

### 1. Double the exposure window

**New task:** count independent constant-rate events for twenty minutes instead of ten.

**Picture and change:** keep rate $\lambda=0.3$ events per minute.
For exposure $t$ minutes, $K_t\sim\operatorname{Poisson}(\lambda t)$.
The count mean changes from $3$ to $6$. The next-event wait remains $\operatorname{Exponential}(0.3)$.

**Check:** the twenty-minute count has mean and variance $6$.
Its zero-event probability is $e^{-6}\approx0.002479$.
The mean next-event wait remains $1/0.3$ minutes.

**Use when:** exposure changes without changing the process rate.

### 2. Allow different underlying count rates

**New task:** describe a hypothetical count population with mean $\mu=3$ and variance $9$.

**Picture and change:** Poisson's equal mean and variance cannot match both moments.
Use a negative-binomial parameterization with variance $\mu+\mu^2/r$, where dispersion parameter $r>0$.

**Check:** $r=3^2/(9-3)=1.5$, giving variance $3+9/1.5=9$.
Matching two moments does not establish the whole distribution's fit.

**Use when:** overdispersion remains after accounting for exposure and relevant groups.

## Dependence can invalidate the count model

Suppose all five visits repeat one shared Bernoulli outcome instead of being independent.
Then $S$ is either zero or five, with mean one and variance $25\times0.16=4$.
Now $P(S=2)=0$.
The support bound and mean stayed compatible with binomial assumptions; independence did not.

## Try a different record

One record contains class counts $(2,1,2)$ from five exclusive three-class trials.
Is that record categorical, multinomial or multi-label?

<details>
<summary>Check your answer</summary>

It is multinomial: the counts sum to five.
One trial would be categorical. Multi-label outcomes allow several labels within one trial and need a different model.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Categorical and multinomial

A categorical variable selects one of $K$ classes with nonnegative probabilities $p_1,\ldots,p_K$ summing to one.
Softmax cross-entropy is the negative categorical log-likelihood.

A multinomial variable counts how often each class appears across $n$ categorical trials. The counts sum to $n$.

Use categorical models for one label and multinomial models for a vector of class counts. A multi-label problem is different because several labels can be true at once.

## Gaussian and Laplace

A Gaussian variable has real-valued support, location $\mu$ and variance $\sigma^2>0$:

$$
X \sim \mathcal{N}(\mu,\sigma^2).
$$

Gaussian residuals with fixed variance give squared-error loss, up to positive scaling and an additive constant.
The mean-variance parameters need not describe bounded or strongly skewed residuals well.

A Laplace residual has location $\mu$, scale $b>0$ and variance $2b^2$.
Its tails decay more slowly than Gaussian tails. Fixed-scale Laplace negative log-likelihood gives absolute-error loss.
If scale is learned, retain its likelihood terms, including the log-scale term.

This connection turns loss selection into a noise-model choice:

| Residual model | Training loss | Main behavior |
| --- | --- | --- |
| Gaussian | squared error | large errors receive strong weight |
| Laplace | absolute error | less sensitive to large errors |

Inspect residuals instead of assuming either model from habit.

## Poisson and negative binomial

A Poisson count with dimensionless expected count $\mu=\lambda t$ has

$$
P(K=k)=\frac{\mu^k e^{-\mu}}{k!}, \qquad k=0,1,2,\ldots
$$

Its mean and variance both equal $\mu$. A log link keeps a predicted rate positive.

Real counts often have variance above the mean. This is overdispersion. A negative-binomial model adds dispersion and is often a better fit for clicks, incidents, or purchases with heterogeneous rates.

Always include exposure. Ten failures in one hour and ten failures in one month do not imply the same rate.

## Exponential and Gamma

An exponential wait has nonnegative real support and rate $\lambda>0$ in events per unit time.
Its mean is $1/\lambda$ and variance is $1/\lambda^2$. It is memoryless:

$$
P(T>s+t \mid T>s)=P(T>t).
$$

Use it when a constant hazard is defensible. It fails when risk changes with elapsed time.

A Gamma variable with shape $a>0$ and rate $\lambda>0$ has mean $a/\lambda$ and variance $a/\lambda^2$.
Scale is the reciprocal of rate. Shape one gives an exponential distribution.
Integer shape $a$ gives the wait until event $a$ in a constant-rate Poisson process.
General shapes also model positive right-skewed values. Censoring needs its own likelihood contribution.

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

Conjugacy keeps the posterior in the prior's family. It simplifies updating; it does not validate the prior or observation model.
Beta and Dirichlet distributions describe unknown probability parameters, not individual observed class labels.

## Worked example

A product records 20 conversions among 500 independent visits sharing probability $p$.
With a $\operatorname{Beta}(1,1)$ prior, the posterior is

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

State the model's generating assumptions after defining its observation unit.

## Common mistakes

- Using Bernoulli and binomial interchangeably.
- Using Poisson without checking overdispersion or exposure.
- Treating softmax as a multi-label model.
- Using a Gaussian for strongly positive, right-skewed data without checking residuals.
- Calling conjugacy evidence that a model is true.
- Ignoring dependence among observations.

## Practice next

Connect distribution choice to [maximum likelihood estimation](/concepts/maximum-likelihood-estimation/), [the exponential family](/concepts/exponential-family/), and [loss-function selection](/questions/how-to-choose-loss-function/).

</div>
