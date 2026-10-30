---
title: "Why is softmax + cross-entropy the right pairing?"
description: "The gradient simplifies to (p - y), and that's not a coincidence. The senior answer derives this and connects to GLMs and numerical stability."
date: "2026-04-26"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth, math-screen, and LLM internals interviews.*

Derive the loss and logit gradient for mutually exclusive classification and explain why implementations combine softmax with cross-entropy.

## Problem

**Context.** A classifier produces one finite logit per class and observes one correct class index for each example.
Softmax defines normalized class probabilities, and the objective is the negative log-probability of the observed class.

**Task.** Derive the gradient with respect to every logit, then explain a numerically stable logits-based computation.
Connect the result to the categorical likelihood and compare separate versus fused implementations.

**Constraints.** Begin with one-hot targets and mutually exclusive classes; smoothing, weighting, and multilabel objectives require stated changes.
Distinguish algebraic equivalence from floating-point behavior, and do not feed probabilities into an API that expects logits.

## The setup

C-class classification. Logits `z = (z_1, ..., z_C)`. Softmax produces probabilities:

```
p_i = exp(z_i) / sum_j exp(z_j)
```

Cross-entropy with one-hot label `y`:

```
L = -sum_i y_i * log p_i = -log p_y
```

(where `y` is the index of the true class).

## The gradient simplification

Compute `dL / dz_k`:

```
dL / dz_k = -d log p_y / dz_k
         = -(1 / p_y) * dp_y / dz_k
```

Two cases for `dp_y / dz_k`:
- If `k == y`: `dp_y / dz_k = p_y * (1 - p_y)`.
- If `k != y`: `dp_y / dz_k = -p_y * p_k`.

Substituting:
- `dL / dz_y = -(1 / p_y) * p_y * (1 - p_y) = -(1 - p_y) = p_y - 1`
- `dL / dz_k = -(1 / p_y) * (-p_y * p_k) = p_k`

Combining: `dL / dz_k = p_k - y_k` where `y_k = 1` for `k = y`, `0` otherwise.

The full gradient is `p - y` (predicted probabilities minus the one-hot true label). Three lines of algebra; the cleanest gradient in deep learning.

<!-- visual:softmax-ce-logit-update -->
<figure class="learning-figure" aria-labelledby="softmax-ce-update-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="softmax-ce-update-title">Gradient descent subtracts the gradient, raising the underpredicted true-class logit</p><div class="visual-argument" data-visual-argument="softmax-ce-logit-update" data-argument-kind="ledger"><p class="visual-argument-premise">Descent subtracts the gradient, so the true logit rises</p><svg viewBox="0 0 360 322" role="img" aria-label="Descent subtracts the gradient, so the true logit rises. For three equal logits and target class A, gradient is (-2/3,1/3,1/3). A learning rate of 0.3 produces logits (0.2,-0.1,-0.1)."><title>Descent subtracts the gradient, so the true logit rises</title><desc>For three equal logits and target class A, gradient is (-2/3,1/3,1/3). A learning rate of 0.3 produces logits (0.2,-0.1,-0.1).</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Initial logits</tspan></text><rect data-proof-token="0" x="20" y="58" width="102.66666666666667" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="71.33333333333334" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="71.33333333333334" dy="0">0</tspan></text><rect data-proof-token="0" x="126.66666666666667" y="58" width="102.66666666666667" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="178" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">0</tspan></text><rect data-proof-token="0" x="233.33333333333334" y="58" width="102.66666666666667" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="284.6666666666667" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="284.6666666666667" dy="0">0</tspan></text><text x="20" y="126" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Cross-entropy gradient</tspan></text><rect data-proof-token="-2/3" x="20" y="156" width="102.66666666666667" height="34" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="71.33333333333334" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="71.33333333333334" dy="0">-2/3</tspan></text><rect data-proof-token="+1/3" x="126.66666666666667" y="156" width="102.66666666666667" height="34" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="178" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">+1/3</tspan></text><rect data-proof-token="+1/3" x="233.33333333333334" y="156" width="102.66666666666667" height="34" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="284.6666666666667" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="284.6666666666667" dy="0">+1/3</tspan></text><text x="20" y="224" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">After subtracting .3 times gradient</tspan></text><rect data-proof-token="+.2" x="20" y="254" width="102.66666666666667" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="71.33333333333334" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="71.33333333333334" dy="0">+.2</tspan></text><rect data-proof-token="-.1" x="126.66666666666667" y="254" width="102.66666666666667" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="178" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">-.1</tspan></text><rect data-proof-token="-.1" x="233.33333333333334" y="254" width="102.66666666666667" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="284.6666666666667" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="284.6666666666667" dy="0">-.1</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 300" role="img" aria-labelledby="softmax-ce-update-svg-title softmax-ce-update-svg-desc">
		<title id="softmax-ce-update-svg-title">A three-class softmax cross-entropy gradient and its opposite-signed logit update</title>
		<desc id="softmax-ce-update-svg-desc">For predicted probabilities A 0.70, B 0.20, and C 0.10 with B as the true class, subtracting the one-hot target gives gradient components positive 0.70, negative 0.80, and positive 0.10. A gradient-descent step moves in the opposite direction: A left by 0.70 eta, B right by 0.80 eta, and C left by 0.10 eta. The two competitor logits decrease, the true-class logit increases, and all three changes sum to zero.</desc>
		<rect class="viz-plot-bg" x="12" y="30" width="336" height="226" rx="4"></rect>
		<path class="viz-gridline" d="M12 65H348M12 119H348M12 173H348M12 227H348M55 30V227M111 30V227M158 30V227M216 30V256"></path>
		<text class="viz-axis-label" x="33" y="50" text-anchor="middle">CLASS</text>
		<text class="viz-axis-label" x="83" y="50" text-anchor="middle">p</text>
		<text class="viz-axis-label" x="134" y="50" text-anchor="middle">y</text>
		<text class="viz-axis-label" x="187" y="44" text-anchor="middle">g = p − y</text>
		<text class="viz-axis-label" x="282" y="44" text-anchor="middle">LOGIT STEP −ηg</text>
		<text class="viz-label" x="33" y="92" text-anchor="middle">A</text>
		<text class="viz-label" x="83" y="92" text-anchor="middle">0.70</text>
		<text class="viz-label" x="134" y="92" text-anchor="middle">0</text>
		<text class="viz-callout" x="187" y="92" text-anchor="middle">+0.70</text>
		<path d="M320 86H244" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:3;stroke-linecap:round"></path>
		<path d="M244 86L253 80V92Z" style="fill:var(--viz-warning-stroke)"></path>
		<text class="viz-callout" x="282" y="108" text-anchor="middle">lower A: −0.70η</text>
		<text class="viz-label" x="33" y="146" text-anchor="middle">B (true)</text>
		<text class="viz-label" x="83" y="146" text-anchor="middle">0.20</text>
		<text class="viz-label" x="134" y="146" text-anchor="middle">1</text>
		<text class="viz-callout" x="187" y="146" text-anchor="middle">−0.80</text>
		<path d="M238 140H326" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:4;stroke-linecap:round"></path>
		<path d="M326 140L317 134V146Z" style="fill:var(--viz-output-stroke)"></path>
		<text class="viz-callout" x="282" y="162" text-anchor="middle">raise B: +0.80η</text>
		<text class="viz-label" x="33" y="200" text-anchor="middle">C</text>
		<text class="viz-label" x="83" y="200" text-anchor="middle">0.10</text>
		<text class="viz-label" x="134" y="200" text-anchor="middle">0</text>
		<text class="viz-callout" x="187" y="200" text-anchor="middle">+0.10</text>
		<path d="M285 194H274" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:3;stroke-linecap:round"></path>
		<path d="M274 194L283 188V200Z" style="fill:var(--viz-warning-stroke)"></path>
		<text class="viz-callout" x="314" y="200" text-anchor="middle">−0.10η</text>
		<text class="viz-axis-label" x="18" y="246">CHECK</text>
		<text class="viz-label" x="70" y="246">Σp = 1</text>
		<text class="viz-label" x="123" y="246">Σy = 1</text>
		<text class="viz-callout" x="181" y="246" text-anchor="middle">Σg = 0</text>
		<text class="viz-callout" x="282" y="246" text-anchor="middle">Σ(−ηg) = 0</text>
		<text class="viz-axis-label" x="180" y="284" text-anchor="middle">left = lower logit · right = raise logit · η &gt; 0</text>
	</svg>
	<div class="visual-derivation-caption">subtract the one-hot target row by row, then reverse each sign for gradient descent. The two wrong-class logits move left, in proportion to their current probabilities; the true-class logit moves right by <code>0.80η</code>. The updates sum to zero, matching softmax's invariance to a shared logit shift. These are logit changes, not direct probability changes. Original worked example checked against <a href="https://www.deeplearningbook.org/contents/mlp.html">Goodfellow, Bengio, and Courville</a> and the <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html">PyTorch CrossEntropyLoss documentation</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="softmax-ce-logit-update"><strong>Check:</strong> Why does the true class's negative gradient produce a positive logit step under gradient descent?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="softmax-ce-logit-update">For three equal logits and target class A, gradient is (-2/3,1/3,1/3). A learning rate of 0.3 produces logits (0.2,-0.1,-0.1).</span></figcaption>
</figure>

## Benefits of the fused operation

> "Three reasons the joint operation is preferred:
>
> **1. Numerical stability.** Computing softmax then cross-entropy separately involves taking `log(exp(...))`, which can overflow or underflow. The joint operation uses log-sum-exp:
>
> ```
> log p_y = z_y - log sum_j exp(z_j) = z_y - max_j z_j - log sum_j exp(z_j - max_j z_j)
> ```
>
> Subtracting `max_j z_j` keeps the largest exponentiated argument at zero, avoiding overflow. Frameworks (PyTorch's `nn.CrossEntropyLoss`) accept logits directly and apply this internally.
>
> **2. Computational efficiency.** The joint operation skips computing the explicit probabilities (since the gradient `p - y` only needs `p`, computed on demand). Saves memory and a few flops.
>
> **3. The gradient is exact and stable.** The `p - y` form has bounded magnitude (each element is in [-1, 1]), so gradients don't explode at the loss layer."

## The L6 connection: GLM

> "Softmax + cross-entropy is the multiclass generalization of sigmoid + binary cross-entropy, both of which are GLMs under their canonical link functions. The gradient simplification `(predicted - true) * input` is a property of *all* canonical-link GLMs, not just classification. Linear regression with MSE has the same gradient form (because MSE on a Gaussian noise model is the GLM with identity link).
>
> Modern deep nets use sigmoid + BCE for binary classification and softmax + CE for multiclass classification because the pairings give simple gradients and stable computation."

## Tells that get you a strong-hire vote

- You **derive the gradient** cleanly.
- You name the **log-sum-exp trick** for numerical stability.
- You connect to **GLMs and canonical links**.
- You explain why frameworks **fuse the operations**.

## Tells that get you down-leveled

- "It just works" without derivation.
- Computing softmax explicitly in code (in real systems, you should pass logits to the loss function).
- No mention of numerical stability.
- Confusion about which axis softmax operates on.

## Common follow-up

"What's wrong with using MSE for classification?"

The L6 answer:

> "Two related problems. (1) Vanishing gradients on confident-wrong predictions: MSE gradient under sigmoid is proportional to `(p - y) * p * (1 - p)`. When the model is very confident and wrong (`p ≈ 1` for the wrong class), the `p * (1 - p)` term vanishes; the model can't learn its way out. Cross-entropy's gradient is `p - y` directly, which stays large precisely when the model is most wrong. (2) MSE assumes Gaussian noise; classification labels are categorical. MLE under the wrong noise model gives the wrong objective. Cross-entropy is MLE under the right (categorical) noise model."

---

*Related: [entropy and mutual information](/concepts/entropy-mutual-information/), [cross-entropy and softmax](/concepts/cross-entropy-softmax/), [derive logistic regression from MLE](/questions/derive-logistic-regression/), and [choose a loss function](/questions/how-to-choose-loss-function/).*
