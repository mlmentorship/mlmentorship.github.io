---
title: "Entropy, mutual information, and information gain"
description: "Entropy measures uncertainty, while mutual information measures shared dependence. Both require careful estimation and neither proves causation or calibrated confidence."
date: "2026-08-29"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["Shannon entropy", "conditional entropy", "mutual information", "information gain", "binary entropy"]
roles: ["Applied Scientist", "Research Scientist", "Research Engineer", "Machine Learning Engineer"]
rounds: ["Math", "Statistics", "ML breadth", "Research"]
difficulty: "Intermediate"
priority: "Core"
prerequisites: ["expectation-variance-covariance-correlation", "probability-distributions-in-ml", "kl-divergence"]
---

## Summary

Entropy is the expected surprisal of a random variable. Conditional entropy is the uncertainty left after observing another variable. Mutual information is the reduction in uncertainty from that observation. Information gain applies the same reduction to a candidate decision-tree split.

These are population properties of chosen random variables. Finite samples, sparse joint tables, binning, learned representations, and confounding can distort their estimates. Mutual information measures association, not causation. Low predictive entropy does not imply that a model is correct or calibrated.

## Discrete entropy and units

For a discrete random variable $X$ with probability mass function $p(x)$, entropy with logarithm base $b$ is

$$
H_b(X)=-\sum_{x}p(x)\log_b p(x).
$$

Use the convention $0\log_b 0=0$. The surprisal of outcome $x$ is $-\log_b p(x)$, so entropy is expected surprisal.

The logarithm base fixes the unit:

- Base 2 gives bits.
- Base $e$ gives nats.
- Base 10 gives hartleys, also called decimal digits of information.

Changing the base rescales the value:

$$
H_b(X)=\frac{H_e(X)}{\ln b}.
$$

Always state the base when reporting a number. An entropy value without a base has an unspecified unit.

A deterministic variable has entropy zero. If $X$ is uniform over $K$ outcomes, then

$$
H_b(X)=\log_b K,
$$

which is the largest entropy among distributions on that fixed support. Entropy depends on both probabilities and the variable definition. Relabeling outcomes preserves it, while merging outcomes can reduce it.

## Coding interpretation

Suppose a source draws independent symbols from a known distribution $p$. For binary prefix codes, the expected code length cannot be below $H_2(p)$. A Huffman code has expected length below $H_2(p)+1$ bit per symbol.

Arithmetic coding works on sequences and can approach the entropy rate more closely. The claim concerns average length over source draws. It does not promise that every individual message has length $H_2(p)$.

Dependence changes the relevant rate. If each symbol depends on its history, a coder can use conditional probabilities. The chain rule gives

$$
H(X_1,\ldots,X_T)=\sum_{t=1}^{T}H(X_t\mid X_1,\ldots,X_{t-1}).
$$

A code built for a wrong distribution $q$ has ideal expected length $H(p,q)$. Its excess over the entropy is the KL divergence from $p$ to $q$, measured in the same log-base units.

## Conditional entropy

Conditional entropy averages the remaining uncertainty in $X$ after observing $Y$:

$$
H_b(X\mid Y)
=\sum_y p(y)H_b(X\mid Y=y)
=-\sum_{x,y}p(x,y)\log_b p(x\mid y).
$$

The average over $Y$ is required. One rare value of $Y$ may determine $X$, while common values leave substantial uncertainty.

The chain rule decomposes joint uncertainty:

$$
H_b(X,Y)=H_b(Y)+H_b(X\mid Y)
=H_b(X)+H_b(Y\mid X).
$$

For discrete variables, conditioning cannot increase entropy on average:

$$
H_b(X\mid Y)\le H_b(X).
$$

Equality holds when $X$ and $Y$ are independent. Conditional entropy can be zero even when $X$ is random. This happens when $Y$ determines $X$ exactly.

## Mutual information in equivalent forms

Mutual information measures how much observing one variable reduces uncertainty about the other:

$$
I_b(X;Y)=H_b(X)-H_b(X\mid Y).
$$

The chain rule gives several equivalent forms:

$$
\begin{aligned}
I_b(X;Y)
&=H_b(Y)-H_b(Y\mid X)\\
&=H_b(X)+H_b(Y)-H_b(X,Y)\\
&=\sum_{x,y}p(x,y)\log_b\frac{p(x,y)}{p(x)p(y)}\\
&=D_{\mathrm{KL},b}\!\left(p_{X,Y}\,\middle\|\,p_Xp_Y\right).
\end{aligned}
$$

It is also an expected change from prior to posterior:

$$
I_b(X;Y)
=\sum_y p(y)D_{\mathrm{KL},b}\!\left(p(X\mid y)\,\middle\|\,p(X)\right).
$$

The KL form shows that mutual information is nonnegative. It is zero exactly when discrete $X$ and $Y$ are independent. Although one entropy-reduction form starts with $X$, mutual information is symmetric:

$$
I_b(X;Y)=I_b(Y;X).
$$

It has no causal direction. It is bounded by the uncertainty available in either variable:

$$
0\le I_b(X;Y)\le \min\{H_b(X),H_b(Y)\}.
$$

## Conditional mutual information

Conditional mutual information measures the additional association between $X$ and $Y$ after $Z$ is known:

$$
I(X;Y\mid Z)
=H(X\mid Z)-H(X\mid Y,Z)
=\sum_z p(z)I(X;Y\mid Z=z).
$$

It can vanish when $Z$ explains a marginal association. It can also be positive when marginal mutual information is zero, as in interactions that cancel across groups. Conditioning on a common effect can create association, so conditional mutual information is not automatically a causal test.

This quantity helps measure the incremental value of a feature after other features are selected. Its empirical estimation is harder because each conditioning context receives fewer observations.

## Three worked examples

### Fair bit

Let $X\sim\operatorname{Bernoulli}(1/2)$. Then

$$
H_2(X)
=-\frac12\log_2\frac12-\frac12\log_2\frac12
=1\text{ bit}.
$$

Before observing the bit, one binary question is needed on average. A deterministic bit would have zero entropy.

### Perfect copy

Let $X$ be a fair bit and let $Y=X$. Both marginal entropies are one bit, but $H_2(X\mid Y)=0$. Therefore,

$$
I_2(X;Y)=H_2(X)-H_2(X\mid Y)=1\text{ bit}.
$$

The joint entropy is one bit, not two, because the pair has only two possible values: $(0,0)$ and $(1,1)$.

### Binary noisy channel

Let $X$ be a fair bit, let $N\sim\operatorname{Bernoulli}(\varepsilon)$ be independent noise, and define $Y=X\mathbin{\oplus}N$. The output remains fair. The binary entropy function is

$$
h_2(\varepsilon)
=-\varepsilon\log_2\varepsilon
-(1-\varepsilon)\log_2(1-\varepsilon).
$$

Given $Y$, the remaining uncertainty about $X$ is $h_2(\varepsilon)$. Thus,

$$
I_2(X;Y)=1-h_2(\varepsilon).
$$

For $\varepsilon=0.1$, $h_2(0.1)\approx0.469$, so the channel carries about $0.531$ bits per input bit. At $\varepsilon=0.5$, the output is independent of the input and mutual information is zero.

<!-- visual:mutual-information-noisy-channel-budget -->
<figure class="learning-figure" aria-labelledby="mutual-information-budget-title">
	<p class="visual-kicker">Worked uncertainty budget</p>
	<p class="visual-title" id="mutual-information-budget-title">If both bits stay fair, where do 0.531 bits of mutual information come from?</p>
	<div class="visual-grid--two" role="group" aria-label="Binary channel with a fair input and ten percent flip probability, followed by its entropy decomposition">
		<section class="visual-panel" aria-labelledby="mutual-information-joint-title">
			<h4 id="mutual-information-joint-title">Joint distribution: outcomes agree 90% of the time</h4>
			<p>Each marginal stays 50/50, but probability concentrates on the two matching outcomes.</p>
			<table class="cm-grid" aria-label="Joint distribution of fair input X and output Y with ten percent crossover probability">
				<thead><tr><th scope="col">X \ Y</th><th scope="col">Y = 0</th><th scope="col">Y = 1</th><th scope="col">P(X)</th></tr></thead>
				<tbody>
					<tr><th scope="row">X = 0</th><td class="cm-selected"><strong>0.45</strong> match</td><td><strong>0.05</strong> flip</td><td><strong>0.50</strong></td></tr>
					<tr><th scope="row">X = 1</th><td><strong>0.05</strong> flip</td><td class="cm-selected"><strong>0.45</strong> match</td><td><strong>0.50</strong></td></tr>
					<tr><th scope="row">P(Y)</th><td><strong>0.50</strong></td><td><strong>0.50</strong></td><td><strong>1.00</strong></td></tr>
				</tbody>
			</table>
		</section>
		<section class="visual-panel" aria-labelledby="mutual-information-ledger-title">
			<h4 id="mutual-information-ledger-title">Observe Y: uncertainty about X shrinks</h4>
			<p>Given either output, X matches it with probability 0.9 and differs with probability 0.1.</p>
			<table class="cm-grid" aria-label="Entropy ledger showing prior uncertainty, remaining uncertainty, and mutual information in bits">
				<thead><tr><th scope="col">Quantity</th><th scope="col">Bits</th><th scope="col">Meaning</th></tr></thead>
				<tbody>
					<tr><th scope="row">H(X)</th><td><strong>1.000</strong></td><td>before Y</td></tr>
					<tr><th scope="row">H(X | Y)</th><td><strong>0.469</strong></td><td>left after Y</td></tr>
					<tr><th scope="row">I(X; Y)</th><td class="cm-selected"><strong>0.531</strong></td><td>removed by Y</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">I(X; Y) = H(X) - H(X | Y) = 1 - 0.469 = 0.531 bits</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> fair marginals mean X and Y each have one bit of uncertainty; they do not mean independence. The diagonal 0.45 cells show that Y usually predicts X, leaving 0.469 bits unresolved and removing 0.531 bits. That removed uncertainty is their mutual information.</figcaption>
</figure>

## Cross-entropy and KL divergence

For a true distribution $p$ and model distribution $q$, cross-entropy is

$$
H_b(p,q)=-\sum_x p(x)\log_b q(x).
$$

Adding and subtracting $\log_b p(x)$ gives

$$
H_b(p,q)=H_b(p)+D_{\mathrm{KL},b}(p\|q).
$$

The first term is fixed when fitting $q$ to a fixed data distribution. Minimizing cross-entropy therefore minimizes forward KL divergence. If $q(x)=0$ where $p(x)>0$, both cross-entropy and KL divergence are infinite.

This identity concerns expected log loss. A low empirical cross-entropy does not by itself prove good calibration under deployment shift or on poorly represented slices.

## Information gain in decision trees

At a classification-tree node, let $Y$ be the class label and let $S$ identify the child selected by a candidate split. Information gain is

$$
\operatorname{IG}(Y;S)
=H(Y)-\sum_s \frac{n_s}{n}H(Y\mid S=s)
=I(Y;S)
$$

under the node's empirical distribution. The child entropies must be weighted by their sample counts.

If balanced binary labels become perfectly separated, the gain is one bit. If every child keeps the parent class proportions, the gain is zero.

Greedy trees choose the split with the largest estimated gain. Features with many possible split points can gain by chance. Minimum leaf sizes, pruning, held-out evaluation, and adjusted criteria help control this bias. Regression trees usually use variance or squared-error reduction instead of Shannon entropy.

## Independence, symmetry, and processing

Mutual information detects any population dependence, including nonlinear dependence. Zero Pearson correlation does not imply zero mutual information. Zero mutual information does imply independence when the distributions are well defined.

For a Markov chain $X\rightarrow Y\rightarrow Z$, the data processing inequality states

$$
I(X;Z)\le I(X;Y).
$$

Processing $Y$ without new access to $X$ cannot create information about $X$. In particular, for a deterministic function $g$,

$$
I(X;g(Y))\le I(X;Y).
$$

Equality can hold when $g(Y)$ retains all information about $X$, as with an invertible transformation. A task-focused representation may discard nuisance information while retaining label information. The inequality does not require every useful representation to preserve every detail.

## Continuous variables need different care

For a continuous variable with density $p(x)$, differential entropy is

$$
h_b(X)=-\int p(x)\log_b p(x)\,dx.
$$

Differential entropy can be negative. It also changes with measurement scale. For nonzero scalar $a$,

$$
h_b(aX)=h_b(X)+\log_b|a|.
$$

A value measured in meters therefore has a different differential entropy after conversion to centimeters. Discrete entropy does not have this coordinate dependence.

Continuous mutual information remains a KL divergence between the joint density and product of marginals. It is nonnegative and invariant under suitable invertible reparameterizations. A perfect continuous copy often has infinite mutual information because its joint distribution lies on a lower-dimensional set.

Binning a continuous variable creates a discrete variable whose entropy depends on bin width. Finer bins can keep increasing discrete entropy. State whether a result uses differential entropy, quantization, or a continuous mutual-information estimator.

## Estimation from finite data

For discrete observations, the plug-in estimator replaces probabilities with frequencies:

$$
\widehat{H}_b(X)=-\sum_x \widehat{p}(x)\log_b\widehat{p}(x).
$$

It is usually biased downward because unobserved outcomes receive no mass. With $K_+$ observed categories, the Miller-Madow correction adds approximately

$$
\frac{K_+-1}{2n\ln b}.
$$

Plug-in mutual information is often biased upward under independence. A flexible joint table fits accidental dependence. Under regular large-sample conditions, the leading null bias is approximately

$$
\frac{(K_X-1)(K_Y-1)}{2n\ln b}.
$$

These approximations fail when counts are sparse. High-dimensional joint supports grow rapidly, so most configurations may never appear. More dimensions can make a direct frequency table unusable even with thousands of samples.

Continuous estimators use bins, kernels, nearest neighbors, density ratios, or learned critics. Each choice adds assumptions and tuning. Results can change with scaling, neighborhood size, architecture, and negative sampling.

Report the estimator, log base, sample size, preprocessing, and tuning choices. Use permutation tests to measure spurious dependence under a null. Use resampling or repeated datasets for uncertainty. A bootstrap measures sample variation, but it cannot restore unseen support or remove structural estimator bias.

## Feature selection does not establish causation

Mutual information can rank features that have nonlinear association with a target. Compute the ranking inside each training fold. Selecting features on the full dataset leaks target information into evaluation.

High marginal mutual information can come from a confounder, a target-derived field, or a logging policy. It does not show that intervening on the feature changes the target.

Marginal ranking also misses redundancy and synergy. Two copies of one feature can each score highly while adding little together. For $Y=X_1\mathbin{\oplus}X_2$ with independent fair bits, each input alone has zero mutual information with $Y$, while the pair determines $Y$.

Conditional mutual information can ask what a feature adds after a selected set. It is harder to estimate because the conditioning space is larger. Validate the final feature set on held-out data and inspect its provenance.

## InfoNCE and representation learning

Contrastive learning often trains a critic to identify one positive pair among $N$ candidates. Under the standard construction, the positive comes from the joint distribution and the other candidates come independently from the marginal distribution.

With natural logarithms, the expected InfoNCE loss has the form

$$
\mathcal{L}_{\mathrm{NCE}}
=-\mathbb{E}\left[
\log\frac{\exp f(x,y_1)}{\sum_{j=1}^{N}\exp f(x,y_j)}
\right].
$$

Under those sampling assumptions,

$$
I(X;Y)\ge \log N-\mathcal{L}_{\mathrm{NCE}}.
$$

This gives lower-bound intuition for contrastive objectives. The bound is capped by $\log N$ and can be loose. A lower loss shows that the critic distinguishes positives from the sampled alternatives. It does not prove that the representation preserves every downstream factor.

In-batch negatives may be correlated or contain semantically valid matches. False negatives and sampling shortcuts through source, position, or formatting can loosen the bound's practical interpretation. Treat InfoNCE as a training objective with a population interpretation under assumptions, rather than an exact mutual-information meter.

## Predictive entropy, calibration, and confidence

Predictive entropy describes the spread of a model's predicted distribution. A wrong model can assign 0.999 probability to one class and have very low entropy. A high-entropy prediction can reflect real ambiguity, missing features, or model failure.

Calibration asks whether events predicted with probability $p$ occur about fraction $p$ of the time. Expected cross-entropy is a proper scoring rule, but finite training, misspecification, regularization, selection, and distribution shift can leave a model miscalibrated.

Do not report entropy as a confidence interval. Attach sampling uncertainty to estimated information quantities. When using ensemble disagreement or parameter-label mutual information, state that the result depends on the model class and posterior approximation.

## Interview procedure

Use this order:

1. Name the variables and whether they are discrete or continuous.
2. Define entropy as expected surprisal and state the log base.
3. Define conditional entropy and give the chain rule.
4. Derive mutual information in entropy and KL forms.
5. State symmetry, independence, bounds, and data processing.
6. Check a fair bit, a perfect copy, and an independent channel.
7. Connect information gain to weighted tree splits.
8. Discuss estimator bias, dimensionality, causation, and calibration.

If the interviewer changes an assumption, recompute the marginals before reusing a memorized formula.

## Common mistakes

- Reporting entropy without a log base or unit.
- Forgetting to weight child entropy in a tree split.
- Calling mutual information directional because one formula uses conditional entropy.
- Claiming high mutual information proves a causal effect.
- Treating empirical zero correlation as independence.
- Applying discrete entropy intuition directly to differential entropy.
- Calling an InfoNCE value the exact mutual information.
- Ignoring upward bias in empirical mutual information.
- Reading low predictive entropy as calibrated confidence.
- Adding information from correlated observations as if they were independent.

## Changed-assumption practice

Start with the fair binary channel where $\varepsilon=0.1$.

1. **Set $\varepsilon=0.5$.** The output becomes independent of the input, so mutual information falls to zero.
2. **Set $\varepsilon=0.9$.** The mutual information returns to about $0.531$ bits. Flipping the output converts it into a channel with error $0.1$.
3. **Change the prior to $P(X=1)=0.9$.** Now $P(Y=1)=0.82$. The mutual information is $h_2(0.82)-h_2(0.1)\approx0.211$ bits, which is below the fair-input value.
4. **Replace $Y$ with a constant.** All information is discarded, so $I(X;g(Y))=0$. Replacing $Y$ with its inverse preserves mutual information.
5. **Observe two noisy copies.** Information increases but cannot exceed $H(X)=1$ bit. Do not add the two single-channel values because the outputs are dependent through $X$.
6. **Give a tree feature a unique value per row.** Training information gain can rise through memorization. Evaluate the split on fresh data or constrain leaf size.

For each change, identify which equality or assumption stopped applying before calculating.

## Related links

- [Probability distributions used in ML](/concepts/probability-distributions-in-ml/)
- [KL divergence](/concepts/kl-divergence/)
- [Decision trees](/concepts/decision-trees/)
- [Contrastive and self-supervised learning](/concepts/contrastive-self-supervised-learning/)
- [Calibration](/concepts/calibration/)
