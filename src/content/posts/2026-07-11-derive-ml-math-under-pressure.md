---
title: "Defend an ML derivation when its assumptions change"
description: "A strong derivation states assumptions, exposes the key identity, checks the result, and explains what it means for model behavior."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

Derive an attention-score variance from stated random-variable assumptions and test whether the reasoning survives a change in dependence.

## Problem

**Context.** The fixture below defines four-dimensional query and key vectors using independent random signs, then changes their within-vector dependence.

**Task.** Find the unscaled and scaled dot-product variances in both cases, showing the assumptions needed at each step.
Explain what the result does and does not establish about attention scaling in a learned model.

**Constraints.** Use the specified zero-mean sign distributions rather than assuming arbitrary learned coordinates are independent.
Check covariance terms when independence changes; a familiar formula is not a substitute for the derivation.

Let $d=4$ and $z=q^Tk$.
All eight coordinates of $q$ and $k$ independently take $+1$ or $-1$, each with probability $1/2$.
Find the variance of $z$ and of $z/\sqrt d$.
Then decide whether the same conclusion holds when every query coordinate shares one sign and every key coordinate shares another.

## Picture

Each product $X_i=q_i k_i$ is another independent sign: mean zero, variance one.
Four independent products add variance; halving their sum multiplies variance by one quarter.

<!-- visual:attention-scaling-variance-trace -->
<figure class="learning-figure plot-panel" aria-labelledby="attention-scaling-variance-title">
	<p class="visual-kicker">An exact four-coordinate distribution</p>
	<p class="visual-title" id="attention-scaling-variance-title">Square-root scaling stabilizes variance only under the stated dependence assumptions</p>
	<svg viewBox="0 0 360 390" role="img" aria-labelledby="attention-scaling-variance-svg-title attention-scaling-variance-svg-desc">
		<title id="attention-scaling-variance-svg-title">Parallel trace of an attention logit and its variance</title>
		<desc id="attention-scaling-variance-svg-desc">Four query and four key coordinates are independent signs with equal positive and negative probabilities. Each product has variance one. The top row follows four products into their sum z and then z divided by two. The aligned bottom row follows variance one per product, total variance four, and scaled variance one. The scale is squared when computing variance. Sharing one query sign and one key sign across coordinates instead gives raw variance sixteen and scaled variance four.</desc>
		<defs><marker id="attention-scaling-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
		<rect class="viz-node viz-node--input" x="14" y="12" width="332" height="62" rx="4"></rect>
		<text class="viz-label" x="26" y="31">ASSUMPTIONS</text>
		<text class="viz-callout" x="26" y="49">4 query signs + 4 key signs: all independent</text>
		<text class="viz-callout" x="26" y="65">Each sign: +1 or −1, each with probability 1/2</text>
		<text class="viz-axis-label" x="60" y="100" text-anchor="middle">1 - PRODUCTS</text>
		<text class="viz-axis-label" x="180" y="100" text-anchor="middle">2 - SUM</text>
		<text class="viz-axis-label" x="300" y="100" text-anchor="middle">3 - SCALE</text>
		<text class="viz-label" x="14" y="120">RANDOM VARIABLE</text>
		<rect class="viz-node" x="14" y="130" width="92" height="54" rx="4"></rect>
		<text class="viz-callout" x="60" y="152" text-anchor="middle">X_i = q_i k_i</text>
		<text class="viz-label" x="60" y="172" text-anchor="middle">4 terms</text>
		<path d="M108 157H126" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#attention-scaling-arrow)"></path>
		<rect class="viz-node" x="132" y="130" width="96" height="54" rx="4"></rect>
		<text class="viz-callout" x="180" y="152" text-anchor="middle">z = sum X_i</text>
		<text class="viz-label" x="180" y="172" text-anchor="middle">add products</text>
		<path d="M230 157H248" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#attention-scaling-arrow)"></path>
		<rect class="viz-node viz-node--output" x="254" y="130" width="92" height="54" rx="4"></rect>
		<text class="viz-callout" x="300" y="152" text-anchor="middle">z / 2</text>
		<text class="viz-label" x="300" y="172" text-anchor="middle">scaled logit</text>
		<text class="viz-label" x="14" y="216">VARIANCE</text>
		<rect class="viz-node" x="14" y="226" width="92" height="54" rx="4"></rect>
		<text class="viz-callout" x="60" y="248" text-anchor="middle">Var(X_i) = 1</text>
		<text class="viz-label" x="60" y="268" text-anchor="middle">each term</text>
		<path d="M108 253H126" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#attention-scaling-arrow)"></path>
		<rect class="viz-node" x="132" y="226" width="96" height="54" rx="4"></rect>
		<text class="viz-callout" x="180" y="248" text-anchor="middle">Var(z) = 4</text>
		<text class="viz-label" x="180" y="268" text-anchor="middle">sum 4 ones</text>
		<path d="M230 253H248" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#attention-scaling-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="254" y="226" width="92" height="54" rx="4"></rect>
		<text class="viz-callout" x="300" y="248" text-anchor="middle">(1/4) × 4</text>
		<text class="viz-label" x="300" y="268" text-anchor="middle">Var = 1</text>
		<rect class="viz-node viz-node--focus" x="14" y="298" width="332" height="42" rx="4"></rect>
		<text class="viz-label" x="26" y="315">DECISIVE MOVE: SQUARE THE SCALE FACTOR</text>
		<text class="viz-callout" x="180" y="332" text-anchor="middle">(1/2)^2 × 4 = 1</text>
		<text class="viz-label" x="14" y="362">CHECK: d = 1 needs no scaling</text>
		<text class="viz-label" x="14" y="380">SHARED SIGNS: Var(z) = 16; Var(z/2) = 4</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> follow the random variable and its variance together. Scaling the variable by one half scales variance by one quarter, not one half. <span class="visual-reader-check" data-reader-check="attention-scaling-variance-trace"><strong>Check:</strong> Why does dividing by 2 yield variance 1 for independent terms but variance 4 for shared terms?</span></figcaption>
</figure>

<p class="diagram-source">References: <a href="https://arxiv.org/abs/1706.03762">Vaswani et al.</a> and <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html">PyTorch scaled-dot-product attention</a>.</p>

## Worked example: attention scaling

The decisive identity retains covariance until the assumptions remove it:

$$
\operatorname{Var}(z)=\sum_i\operatorname{Var}(X_i)+2\sum_{i<j}\operatorname{Cov}(X_i,X_j).
$$

Here the covariance terms vanish, so $\operatorname{Var}(z)=4$ and $\operatorname{Var}(z/2)=(1/2)^2\times4=1$.
The $2^8=256$ equally likely sign assignments give an exact check:

| $z$ | $-4$ | $-2$ | $0$ | $2$ | $4$ |
| --- | --- | --- | --- | --- | --- |
| Assignment count | $16$ | $64$ | $96$ | $64$ | $16$ |

Check also $d=1$: no scale change is needed.
Dividing variance by $\sqrt d$ would incorrectly give two at $d=4$; variance squares the multiplier.

Now set every $q_i=Q$ and every $k_i=K$, with independent signs $Q,K$.
All products equal $QK$, so $z=4QK$ has variance 16.
Six pair covariances contribute $2\times6=12$ beyond the four individual variances.
Dividing by two leaves variance four.

## Change the assumptions or object

### 1. Change coordinate variances

**New task:** independent zero-mean query coordinates have variance four; independent key coordinates have variance nine.

**Picture and change:** retain independence of vectors and products, but each product now has variance $4\times9=36$.
For $d=4$, the sum has variance 144.

**Check:** dividing by two leaves variance 36.
Dividing by $\sigma_q\sigma_k\sqrt d=12$ restores one in this stylized distribution.

**Use when:** testing the unit-variance premise, not claiming exact whitening of trained heads.

### 2. Average correlated worker errors

**New task:** average four unit-variance errors with pairwise correlation $0.5$.

**Picture and change:** apply the same covariance identity to their average.

$$
\operatorname{Var}(\bar e)=\frac{4+2\times6\times0.5}{16}=0.625.
$$

**Check:** independence gives $0.25$; perfect correlation gives one.

**Use when:** shared worker errors invalidate independent-noise averaging.

## Transfer: recover a missing identity

You forget the importance-sampling identity, and the proposal misses some target support.
What can you derive safely?

<details>
<summary>Check your answer</summary>

Start from the target integral and insert $q/q$ only where $q>0$.
Identify any missing contribution; support mismatch can block equality.
Large weights or an effective-sample-size number cannot recover omitted support.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Use the same four-part structure

### 1. Setup

Define variables, dimensions, probability direction and assumptions.
For gradients, state the differentiated variable and row/column convention.

### 2. Key identity

Expose the decisive step: chain rule, log-ratio expansion, log-sum-exp derivative, Jensen's inequality or a variance identity.
Justify it rather than hiding it behind a name.

### 3. Sanity checks

Use at least two: dimensions, sign, equality case, scalar case, limit, symmetry or numerical scale.

### 4. Interpretation

Explain the model or system consequence, then change an assumption and identify what survives.

Coordinate independence is sufficient, not necessary, for the attention result.
For independent zero-mean vectors with covariance matrices $\Sigma_q,\Sigma_k$:

$$
\operatorname{Var}(q^Tk)=\operatorname{tr}(\Sigma_q\Sigma_k).
$$

This conditional result does not establish universal softmax saturation or training stability.
Learned projections and normalization alter empirical head distributions.

<span id="what-an-l4-answer-sounds-like"></span>

## A remembered formula is not a proof

“It prevents large values” does not justify $1/\sqrt d$ or identify when the variance calculation changes.

<span id="what-an-l5-answer-adds"></span>

## Eight proof checkpoints

These are separate derivations connected by a presentation protocol, not one common mathematical proof.

| Prompt | Decisive step and check |
| --- | --- |
| Softmax cross-entropy | Log-sum-exp differentiation gives $p-y$ for normalized targets; gradients sum to zero; common logit shifts preserve probabilities. |
| KL direction | $D(P\Vert Q)=\mathbb E_P\log(P/Q)$; check support, equality and asymmetry. |
| ELBO | $\log p(x)=\operatorname{ELBO}(q)+D(q(z)\Vert p(z\mid x))$, when well-defined; equality uses the exact posterior. |
| L1 versus L2 | $|w|$ has subgradient interval $[-1,1]$ at zero; $w^2/2$ has continuous gradient $w$. |
| Geometric attempts | Independent identical success trials counted from one give $\mathbb E[T]=1/p$; $p=1$ gives one attempt. |
| Averaging | Independent errors give $\sigma^2/n$; equal pair correlation gives $\sigma^2[1+(n-1)\rho]/n$. |
| Importance sampling | Insert $q/q$ in the target integral only where the proposal supports the contributing integrand. |
| Attention | Keep covariance terms until assumptions justify removing them; square any scale factor in variance. |

For Bernoulli parameters $0.8$ and $0.5$, the KL is about $0.192745$ nats; reversing direction gives $0.223144$.
The unequal values check asymmetry, not an error in the expectation direction.
At geometric success probability zero, success never occurs and the expectation is not finite.
Importance weights can have infinite variance despite a valid support identity.
Effective sample size is a diagnostic, not a finite-variance or support certificate.

<span id="what-an-l6-answer-adds"></span>

## Which conclusion survives?

Independence, convexity, asymptotics, unbiasedness and support overlap are different premises.
State which identity uses each one.
The ELBO gap concerns posterior approximation, not an unspecified reconstruction-plus-regularization slogan.

<span id="tells-that-get-you-a-strong-hire-vote"></span>

## Evidence in a derivation

Defined notation, justified identities, two checks and explicit error recovery make the reasoning inspectable.
No format guarantees a hiring outcome.

<span id="tells-that-get-you-down-leveled"></span>

## Errors to expose

Undefined notation, missing support and unjustified algebra remain errors even when the final formula looks familiar.
Stop at the missing step rather than inventing one.

## Common follow-up

"You cannot remember the identity needed for the next step. What do you do?"

State what is known, derive the missing piece from a definition or simpler case, and identify the exact unresolved step.

Use the [timed math oral](/prep/labs/math-oral/) with an observer and a changed-assumption follow-up.

*Related: [derive logistic regression](/questions/derive-logistic-regression/), [softmax and cross-entropy](/questions/softmax-cross-entropy-pairing/), and [KL divergence](/concepts/kl-divergence/).*

</div>
