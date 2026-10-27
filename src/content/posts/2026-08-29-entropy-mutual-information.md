---
title: "Entropy and mutual information: a noisy observation"
description: "Derive uncertainty removed from a joint table, then change the prior and the observation grouping without confusing information with causation."
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

Entropy measures expected surprisal, while mutual information measures the uncertainty removed by observing another variable.
Can two fair bits each have one bit of uncertainty while one reveals only part of the other?

## Summary

Use finite discrete variables and base-two logarithms unless stated otherwise.
Our teaching fixture sends a fair bit through independent noise that flips it with probability $0.1$.
We will compute the joint table before subtracting uncertainty.
Population information is not a causal effect or a guarantee about an empirical estimate.

## Discrete entropy and units

Outcome $x$ has surprisal $-\log_b p(x)$ for logarithm base $b>1$.
Entropy averages that quantity:

$$
H_b(X)=-\sum_xp(x)\log_b p(x),\qquad 0\log_b0=0.
$$

Base two gives bits, base $e$ gives nats, and base ten gives hartleys.
The conversion is $H_b=H_e/\ln b$.
A deterministic variable has zero entropy.
On a fixed support of $K$ outcomes, uniform mass maximizes entropy at $\log_b K$.
Relabeling preserves entropy; merging outcomes can reduce it.

## Three worked examples

### Fair bit

For $\Pr(X=1)=1/2$, each outcome has surprisal one bit.
Therefore $H_2(X)=-(1/2)\log_2(1/2)-(1/2)\log_2(1/2)=1$.

### Perfect copy

If $Y=X$, observing $Y$ determines $X$.
The remaining entropy is zero, so the removed uncertainty is one bit.
The pair has only two possible states, not four equally likely states.

### Binary noisy channel

Let $N\sim\operatorname{Bernoulli}(0.1)$ be independent of the fair input, and set $Y=X\mathbin{\oplus}N$.
Multiply input mass by the copy or flip probability:

| Joint mass | $Y=0$ | $Y=1$ | Row total |
| --- | --- | --- | --- |
| $X=0$ | $0.45$ | $0.05$ | $0.50$ |
| $X=1$ | $0.05$ | $0.45$ | $0.50$ |
| Column total | $0.50$ | $0.50$ | $1$ |

Both marginals are fair, but agreement has probability $0.9$.
Independent fair bits would instead give $0.25$ in every cell.

<!-- visual:mutual-information-noisy-channel-budget -->
<figure class="learning-figure" aria-labelledby="mutual-information-budget-title" aria-describedby="mutual-information-budget-description">
<p class="visual-kicker">Worked uncertainty budget</p>
<p class="visual-title" id="mutual-information-budget-title">Observing Y removes the shared part of X's uncertainty.</p>
<p id="mutual-information-budget-description">One bit splits into 0.531 bits removed by Y and 0.469 bits still uncertain.</p>
<div class="visual-scroll" tabindex="0" role="region" aria-label="Noisy-channel joint table and uncertainty budget">
<svg class="information-budget-visual" viewBox="0 0 900 420" role="img" aria-labelledby="mutual-information-svg-title mutual-information-svg-description" width="900" height="420">
<title id="mutual-information-svg-title">Mutual information is the uncertainty removed by observing another variable</title>
<desc id="mutual-information-svg-description">A one-bit horizontal bar for H of X is split into zero point five three one bits removed by Y and zero point four six nine bits still uncertain after Y. A second inset shows a two by two joint table where diagonal match probabilities are zero point four five and off-diagonal flip probabilities are zero point zero five, with both marginals equal to one half.</desc>
<text x="46" y="45" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="20" font-weight="700">One bit of X before observing Y</text>
<text x="46" y="70" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="13">binary noisy channel with flip probability 0.1</text>
<rect x="60" y="112" width="520" height="72" rx="8" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" />
<rect x="60" y="112" width="276" height="72" rx="8" fill="var(--viz-focus-bg)" stroke="var(--viz-focus-stroke)" stroke-width="2" />
<rect x="336" y="112" width="244" height="72" rx="8" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)" stroke-width="2" />
<text x="198" y="145" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="18" font-weight="700">0.531 bits</text>
<text x="198" y="166" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">removed by observing Y</text>
<text x="458" y="145" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="18" font-weight="700">0.469 bits</text>
<text x="458" y="166" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">still uncertain: H(X|Y)</text>
<text x="60" y="210" fill="var(--viz-focus-stroke)" font-family="Lato, sans-serif" font-size="14" font-weight="700">I(X;Y) = H(X) - H(X|Y) = 1.000 - 0.469</text>
<circle cx="230" cy="295" r="74" fill="var(--viz-input-bg)" fill-opacity="0.78" stroke="var(--viz-input-stroke)" stroke-width="2" />
<circle cx="318" cy="295" r="74" fill="var(--viz-output-bg)" fill-opacity="0.78" stroke="var(--viz-output-stroke)" stroke-width="2" />
<path d="M274 236 A74 74 0 0 1 274 354 A74 74 0 0 1 274 236" fill="var(--viz-focus-bg)" stroke="var(--viz-focus-stroke)" stroke-width="2" />
<text x="195" y="292" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="14" font-weight="700">H(X)</text>
<text x="351" y="292" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="14" font-weight="700">H(Y)</text>
<text x="274" y="292" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="14" font-weight="700">shared</text>
<text x="274" y="310" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13">0.531</text>
<text x="86" y="386" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="12">Overlap is a memory aid: shared information, not a causal direction.</text>
<text x="640" y="34" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="20" font-weight="700">Fair marginals</text>
<text x="640" y="58" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="20" font-weight="700">can still depend</text>
<text x="640" y="79" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="13">mass concentrates on agreement</text>
<rect x="635" y="108" width="86" height="58" rx="5" fill="var(--viz-focus-bg)" stroke="var(--viz-focus-stroke)" stroke-width="2" />
<rect x="725" y="108" width="86" height="58" rx="5" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" />
<rect x="635" y="170" width="86" height="58" rx="5" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" />
<rect x="725" y="170" width="86" height="58" rx="5" fill="var(--viz-focus-bg)" stroke="var(--viz-focus-stroke)" stroke-width="2" />
<text x="678" y="141" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="17" font-weight="700">0.45</text>
<text x="768" y="141" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="17" font-weight="700">0.05</text>
<text x="678" y="203" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="17" font-weight="700">0.05</text>
<text x="768" y="203" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="17" font-weight="700">0.45</text>
<text x="678" y="95" text-anchor="middle" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="12">Y=0</text>
<text x="768" y="95" text-anchor="middle" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="12">Y=1</text>
<text x="610" y="142" text-anchor="end" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="12">X=0</text>
<text x="610" y="204" text-anchor="end" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="12">X=1</text>
<text x="826" y="142" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">row = 0.50</text>
<text x="826" y="204" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">row = 0.50</text>
<text x="651" y="255" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">column totals also 0.50 / 0.50</text>
<path d="M620 292 H850" stroke="var(--c-rule)" stroke-width="1.5" />
<text x="640" y="328" fill="var(--viz-focus-stroke)" font-family="Lato, sans-serif" font-size="14" font-weight="700">Y is fair, but usually equal to X</text>
<text x="640" y="350" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="12">That diagonal concentration</text>
<text x="640" y="368" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="12">is the shared information.</text>
</svg>
</div>
<figcaption><strong>Read it this way:</strong> the full bar is the input's one bit of uncertainty. Observing Y removes 0.531 bits and leaves 0.469 bits. The diagonal table entries explain the dependence despite fair marginals. The circles are a memory aid, not a proof or causal diagram.</figcaption>
</figure>

## Conditional entropy

Given either output, the input matches it with probability $0.9$.
Write $h_2(t)=-t\log_2t-(1-t)\log_2(1-t)$.
Average the posterior entropies over the output probabilities:

$$
H_2(X\mid Y)=\sum_yp(y)H_2(X\mid Y=y)
=\tfrac12h_2(0.1)+\tfrac12h_2(0.1)
\approx0.468996.
$$

Conditioning reduces entropy on average, not necessarily for every particular observed value.
Using $p(x,y)=p(y)p(x\mid y)$ inside the logarithm proves the chain rule:

$$
H(X,Y)=H(Y)+H(X\mid Y)=1.468996\text{ bits}.
$$

## Mutual information in equivalent forms

Subtract remaining uncertainty:

$$
\begin{aligned}
I(X;Y)&=H(X)-H(X\mid Y)\\
&=H(X)+H(Y)-H(X,Y)\\
&=\sum_{x,y}p(x,y)\log_2\frac{p(x,y)}{p(x)p(y)}.
\end{aligned}
$$

The reference gives $1-0.468996=0.531004$ bits.
The table independently gives $0.9\log_2(1.8)+0.1\log_2(0.2)$, the same result.
The middle expression proves symmetry; it does not supply a causal direction.
Gibbs' inequality applied to the last expression gives nonnegativity, with equality exactly at independence.
Nonnegative conditional entropies also give $I(X;Y)\le\min(H(X),H(Y))$.

### Change the input prior

Set $\Pr(X=1)=0.9$, retaining independent flips with probability $0.1$.
The four joint cells become $0.09,0.01,0.09,0.81$ in row order.
Now $\Pr(Y=1)=0.82$, and

$$
I(X;Y)=H(Y)-H(Y\mid X)=h_2(0.82)-h_2(0.1)\approx0.211081.
$$

Recompute marginals when class priors change; the fair-input shortcut no longer applies.

## Information gain in decision trees

Let $X$ be a label and $S$ identify its tree child.
Split four balanced-label records into one pure record and three records with class counts one and two.

$$
\operatorname{IG}=H(X)-\sum_s\frac{n_s}{n}H(X\mid S=s)
=1-\tfrac14(0)-\tfrac34h_2(1/3)
\approx0.311278\text{ bits}.
$$

Use record weights, not an unweighted child average.
Perfect separation gives one bit; unchanged child proportions give zero.
This is mutual information under the node's empirical distribution, not evidence of generalization.

## Conditional mutual information

For independent fair $X_1,X_2$, the target $T=X_1\mathbin{\oplus}X_2$ is independent of either input alone.
Thus $I(T;X_1)=I(T;X_2)=0$, while $I(T;(X_1,X_2))=1$ bit.
Marginal ranking misses the interaction.
Conditional information measures the added association after context $Z$ is known:

$$
I(X;Y\mid Z)=H(X\mid Z)-H(X\mid Y,Z)
=\sum_zp(z)I(X;Y\mid Z=z).
$$

For XOR, $I(T;X_2\mid X_1)=1$ bit.
Conditioning can create association, including through a common effect; this is not automatically a causal test.

**Reconstruct:** what changes at flip probability $0.9$, or when the observation is replaced by a constant?

<details><summary>Answer</summary>

Inverting the $0.9$ channel recovers the $0.1$ channel, preserving $0.531004$ bits.
A constant observation leaves all input uncertainty, giving zero mutual information.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

## Coding interpretation

For known iid symbols, a binary prefix code's expected length is at least $H_2(p)$.
Huffman coding achieves expected length below $H_2(p)+1$.
Arithmetic coding over sequences can approach the entropy rate more closely.
These are average-length statements, not lengths promised for every message.
Dependent symbols instead permit conditional coding:

$$
H(X_1,\ldots,X_T)=\sum_{t=1}^TH(X_t\mid X_1,\ldots,X_{t-1}).
$$

## Cross-entropy and KL divergence

For source law $p$ and model law $q$, expected log loss decomposes as

$$
H_b(p,q)=-\sum_xp(x)\log_bq(x)
=H_b(p)+D_{\mathrm{KL},b}(p\|q).
$$

Add and subtract $\log_b p(x)$ to obtain the identity.
For fixed $p$, minimizing cross-entropy minimizes forward KL.
Missing $q$ support where $p>0$ makes both quantities infinite.
A mismatched ideal coder incurs this same expected excess length.

The joint-table expression above is $D_{\mathrm{KL}}(p_{X,Y}\|p_Xp_Y)$.
Equivalently, information averages posterior-to-prior divergence:

$$
I(X;Y)=\sum_yp(y)D_{\mathrm{KL}}(p(X\mid y)\|p(X)).
$$

## Independence, symmetry, and processing

Zero Pearson correlation does not imply independence.
For a Markov chain $X\rightarrow Y\rightarrow Z$, data processing gives

$$
I(X;Z)\le I(X;Y),\qquad I(X;g(Y))\le I(X;Y).
$$

Processing without new access to $X$ cannot create information about it.
Invertible transformations preserve information.
A useful representation can discard nuisance details while retaining task information.

## Continuous variables need different care

For a density, differential entropy is $h_b(X)=-\int p(x)\log_b p(x)\,dx$.
It can be negative and depends on scale:

$$
h_b(aX)=h_b(X)+\log_b|a|,\qquad a\ne0.
$$

A uniform variable on $[0,0.5]$ has differential entropy $-1$ bit.
Converting meters to centimeters adds $\log_b100$ without adding information.
Continuous mutual information remains nonnegative and invariant under suitable invertible reparameterizations.
For a non-atomic continuous variable, an exact copy has infinite mutual information.
The joint law is singular relative to the product law, so an ordinary joint-density formula is unavailable.
Binning creates a discrete variable whose entropy depends on bin width.
State whether the quantity uses densities, quantized values, or a continuous information estimator.

## Estimation from finite data

The plug-in entropy substitutes empirical frequencies:

$$
\widehat H_b=-\sum_x\widehat p(x)\log_b\widehat p(x).
$$

It is downward biased; unseen outcomes receive zero mass.
For well-sampled fixed discrete support, Miller-Madow adds approximately

$$
\frac{K_+-1}{2n\ln b}.
$$

Here $K_+$ counts observed categories and $n$ counts iid observations.
Under regular fixed-support independence, plug-in mutual information has approximate upward null bias

$$
\frac{(K_X-1)(K_Y-1)}{2n\ln b}.
$$

Sparse joint tables invalidate these expansions; more dimensions rapidly exhaust observed support.
Continuous estimators use bins, kernels, neighbors, density ratios, or learned critics, each with tuning assumptions.
Report the estimator, units, sample size, preprocessing, and tuning choices.
Use an appropriate permutation null for spurious dependence and repeated sampling for uncertainty.
Bootstrap variation cannot restore unseen support or eliminate structural estimator bias.

## Feature selection does not establish causation

Compute feature selection inside each training fold; full-data ranking leaks target information into evaluation.
Confounding, target-derived fields, and logging policies can all create association.
None establishes the effect of intervening on the feature.
Duplicate features can each score highly while adding nothing together; XOR shows the opposite interaction.
Conditional information asks about incremental association but requires estimating a larger conditioning space.
Validate selected features on held-out data and inspect provenance.
Tree split searches can exploit accidental gain; constrain leaf size, prune, and evaluate independently.
Regression trees usually optimize variance or squared-error reduction, not Shannon entropy.

## InfoNCE and representation learning

Under the standard construction, $(x,y_1)$ comes from the joint law.
The other $N-1$ candidates are independent marginal draws.
With natural logarithms and critic $f$, the expected loss satisfies

$$
\mathcal L_{\mathrm{NCE}}=-\mathbb E\log
\frac{e^{f(x,y_1)}}{\sum_{j=1}^Ne^{f(x,y_j)}},\qquad
I(X;Y)\ge\ln N-\mathcal L_{\mathrm{NCE}}.
$$

This lower bound is capped at $\ln N$ and can be loose.
Correlated negatives or sampling shortcuts can violate its construction or weaken its relevance.
Semantically matching negatives can also occur under valid marginal sampling; semantics alone do not invalidate the bound.
Lower loss demonstrates discrimination against sampled alternatives, not an exact information measurement or universal downstream preservation.

## Predictive entropy, calibration, and confidence

Predictive entropy measures spread within the model's stated distribution.
A wrong prediction with probability $0.999$ can have very low entropy.
High entropy can reflect ambiguity, missing inputs, or model failure.
Calibration compares predicted probabilities with frequencies in a stated population.
Proper population log loss does not make finite, misspecified, selected, or shifted predictions automatically calibrated.
Entropy is not a confidence interval.
Ensemble and parameter-label information also depend on model and posterior assumptions.

## Interview procedure

Name the variables, probability law, and log base.
Rebuild marginals, then conditionals, before calculating removed uncertainty.
Check the entropy and joint-ratio forms agree.
State sampling, causation, and confidence limitations separately from population identities.

## Common mistakes

- Missing units or child weights changes the reported quantity.
- Symmetry is not causation, and zero correlation is not independence.
- Discrete and differential entropies have different scale behavior.
- InfoNCE and plug-in estimates are not exact population measurements.
- Low predictive entropy is not calibrated confidence.
- Dependent observations cannot contribute independent information merely because they occupy different rows.

## Changed-assumption practice

At flip probability $0.5$, every joint cell is $0.25$ and information is zero.
At $0.9$, invert the output to recover the reference information.
With prior $0.9$, recompute the output mass $0.82$ instead of assuming fairness.
A constant transformation discards all information; inversion preserves it.
Two conditionally independent $0.1$-flip copies yield about $0.742086$ bits jointly, not twice $0.531004$.
Their outputs are dependent through the shared input.
A unique-per-row tree feature can memorize labels; fresh-data evaluation or leaf constraints are still required.

## Related links

- [Probability distributions used in ML](/concepts/probability-distributions-in-ml/)
- [KL divergence](/concepts/kl-divergence/)
- [Decision trees](/concepts/decision-trees/)
- [Contrastive and self-supervised learning](/concepts/contrastive-self-supervised-learning/)
- [Calibration](/concepts/calibration/)

</div>
