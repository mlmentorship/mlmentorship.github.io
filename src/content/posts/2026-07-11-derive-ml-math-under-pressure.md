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
