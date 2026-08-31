---
title: "Derive ML math under oral-interview pressure"
description: "A strong derivation states assumptions, exposes the key identity, checks the result, and explains what it means for model behavior."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> Derive one result at the board, explain every non-obvious step, and handle a follow-up that changes an assumption.

Do not rush to the remembered last line. Oral math is scored on setup, progression, sanity checks, and interpretation. A correct formula reached through unexplained jumps is fragile evidence.

## Use the same four-part structure

### 1. Setup

Define the object, dimensions, probability direction, and assumptions. For KL divergence, state which distribution is inside the expectation. For a gradient, state whether vectors are columns and which variable is differentiated.

### 2. Key identity

Name the one step that carries the derivation: log-ratio expansion, chain rule, log-sum-exp derivative, Jensen's inequality, variance of an independent sum, or geometric-series identity. Do not spend equal time on routine algebra and the decisive move.

### 3. Sanity checks

Use at least two:

- dimensions;
- sign or non-negativity;
- equality case;
- one-dimensional case;
- limiting behavior;
- symmetry or expected asymmetry;
- numerical scale.

### 4. Interpretation

Explain what changes when one term grows and how that affects optimization, uncertainty, model behavior, or a system decision.

## Worked example: attention scaling

Assume query and key coordinates are independent, zero mean, unit variance. For

$$
z = q^T k = \sum_{i=1}^{d} q_i k_i,
$$

each product has mean zero and variance one under the assumptions. Independence gives:

$$
\operatorname{Var}(z) = \sum_{i=1}^{d} \operatorname{Var}(q_i k_i) = d.
$$

Dividing by $\sqrt{d}$ gives unit variance:

$$
\operatorname{Var}\left(\frac{q^T k}{\sqrt{d}}\right) = 1.
$$

<!-- visual:attention-scaling-variance-trace -->
<figure class="learning-figure plot-panel" aria-labelledby="attention-scaling-variance-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="attention-scaling-variance-title">Trace why dividing the attention logit by <code>sqrt(d)</code> restores unit variance.</p>
	<svg viewBox="0 0 360 390" role="img" aria-labelledby="attention-scaling-variance-svg-title attention-scaling-variance-svg-desc">
		<title id="attention-scaling-variance-svg-title">Parallel trace of an attention logit and its variance</title>
		<desc id="attention-scaling-variance-svg-desc">Assume the d products X sub i equal q sub i times k sub i are independent across coordinates, with independent zero-mean, unit-variance factors. Each product has variance one. In the random-variable row, d products are summed to form z, then z is divided by square root of d. In the aligned variance row, d unit variances add to d. Dividing z by square root of d multiplies its variance by the square of that factor, one over d, so the result has variance one. If coordinates are correlated, covariance terms remain.</desc>
		<defs><marker id="attention-scaling-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
		<rect class="viz-node viz-node--input" x="14" y="12" width="332" height="62" rx="4"></rect>
		<text class="viz-label" x="26" y="31">ASSUMPTIONS</text>
		<text class="viz-callout" x="26" y="49">q_i and k_i: independent, mean 0, variance 1</text>
		<text class="viz-callout" x="26" y="65">products X_i: independent across coordinates</text>
		<text class="viz-axis-label" x="60" y="100" text-anchor="middle">1 - PRODUCTS</text>
		<text class="viz-axis-label" x="180" y="100" text-anchor="middle">2 - SUM</text>
		<text class="viz-axis-label" x="300" y="100" text-anchor="middle">3 - SCALE</text>
		<text class="viz-label" x="14" y="120">RANDOM VARIABLE</text>
		<rect class="viz-node" x="14" y="130" width="92" height="54" rx="4"></rect>
		<text class="viz-callout" x="60" y="152" text-anchor="middle">X_i = q_i k_i</text>
		<text class="viz-label" x="60" y="172" text-anchor="middle">d terms</text>
		<path d="M108 157H126" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#attention-scaling-arrow)"></path>
		<rect class="viz-node" x="132" y="130" width="96" height="54" rx="4"></rect>
		<text class="viz-callout" x="180" y="152" text-anchor="middle">z = sum X_i</text>
		<text class="viz-label" x="180" y="172" text-anchor="middle">add products</text>
		<path d="M230 157H248" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#attention-scaling-arrow)"></path>
		<rect class="viz-node viz-node--output" x="254" y="130" width="92" height="54" rx="4"></rect>
		<text class="viz-callout" x="300" y="152" text-anchor="middle">z / sqrt(d)</text>
		<text class="viz-label" x="300" y="172" text-anchor="middle">scaled logit</text>
		<text class="viz-label" x="14" y="216">VARIANCE</text>
		<rect class="viz-node" x="14" y="226" width="92" height="54" rx="4"></rect>
		<text class="viz-callout" x="60" y="248" text-anchor="middle">Var(X_i) = 1</text>
		<text class="viz-label" x="60" y="268" text-anchor="middle">each term</text>
		<path d="M108 253H126" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#attention-scaling-arrow)"></path>
		<rect class="viz-node" x="132" y="226" width="96" height="54" rx="4"></rect>
		<text class="viz-callout" x="180" y="248" text-anchor="middle">Var(z) = d</text>
		<text class="viz-label" x="180" y="268" text-anchor="middle">sum d ones</text>
		<path d="M230 253H248" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#attention-scaling-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="254" y="226" width="92" height="54" rx="4"></rect>
		<text class="viz-callout" x="300" y="248" text-anchor="middle">(1/d) x d</text>
		<text class="viz-label" x="300" y="268" text-anchor="middle">Var = 1</text>
		<rect class="viz-node viz-node--focus" x="14" y="298" width="332" height="42" rx="4"></rect>
		<text class="viz-label" x="26" y="315">DECISIVE MOVE: VARIANCE SQUARES THE SCALE FACTOR</text>
		<text class="viz-callout" x="180" y="332" text-anchor="middle">(1/sqrt(d))^2 x d = (1/d) x d = 1</text>
		<text class="viz-label" x="14" y="362">CHECK: d = 1 needs no scaling</text>
		<text class="viz-label" x="14" y="380">IF CORRELATED: covariance terms remain</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> follow the top and bottom rows together. Summing <code>d</code> independent products creates a logit with variance <code>d</code>. The next operation multiplies the logit by <code>1/sqrt(d)</code>, so it multiplies the variance by the square of that factor, <code>1/d</code>. The result is unit variance. Correlated coordinates break the simple sum because covariance terms remain. Original derivation checked against <a href="https://arxiv.org/abs/1706.03762">Vaswani et al.</a> and the <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html">PyTorch scaled-dot-product attention documentation</a>.</figcaption>
</figure>

Without scaling, larger head dimension produces wider logits, saturates softmax, and weakens useful gradients. The follow-up should challenge the assumptions: correlated or non-unit coordinates change the exact variance, while normalization and learned projections affect the empirical distribution.

## What an L4 answer sounds like

The candidate remembers $1/\sqrt{d}$ but says only "it prevents large values." Symbols appear without assumptions, an algebra step is skipped, and no boundary case checks the result.

## What an L5 answer adds

An L5 candidate defines variables, derives the key identity cleanly, checks dimensions and a special case, and interprets the result. When stuck, they state the exact missing step instead of producing random algebra.

They can handle a prompt set spanning:

- softmax cross-entropy gradient;
- KL between simple distributions;
- ELBO identity;
- L1 versus L2 gradient behavior;
- expected attempts under geometric success;
- variance reduction from averaging;
- importance-sampling estimator and support condition;
- attention logit scaling.

## What an L6 answer adds

An L6 candidate makes assumptions visible enough to change them. They know which conclusion is robust and which is an artifact of independence, asymptotics, convexity, unbiasedness, or support overlap.

They connect derivation to practice without hand-waving. For importance sampling, support mismatch and heavy weights become effective sample size and unstable evaluation. For averaged worker noise, correlation prevents the expected $1/n$ variance reduction. For ELBO, the gap identifies approximation error rather than becoming a generic reconstruction-plus-regularization slogan.

They also control the room. They signpost, invite correction on notation, and preserve a clean thread under interruption.

## Tells that get you a strong-hire vote

- Symbols, dimensions, direction, and assumptions come first.
- The key identity is named and justified.
- Algebra is paced around the difficult step.
- Dimensions and a boundary case check the result.
- Interpretation connects to model or system behavior.
- A changed assumption produces a changed conclusion.
- You recover from a mistake explicitly rather than hiding it.

## Tells that get you down-leveled

- Writing the remembered final formula immediately.
- Undefined notation or KL direction.
- "By the chain rule" over the only step being tested.
- No sanity check.
- Correct algebra with an incorrect interpretation.
- Treating every assumption as harmless.
- Continuing silently after losing the derivation.

## Common follow-up

"You cannot remember the identity needed for the next step. What do you do?"

State what you know, derive the missing piece from a definition or simpler case, and make the uncertainty explicit. A clean partial derivation with the exact blocker gives more signal than confident fabrication. The interviewer can help once the gap is localized.

Use the [timed math oral](/prep/labs/math-oral/) with an observer and a changed-assumption follow-up.

*Related: [derive logistic regression](/questions/derive-logistic-regression/), [softmax and cross-entropy](/questions/softmax-cross-entropy-pairing/), and [KL divergence](/concepts/kl-divergence/).*
