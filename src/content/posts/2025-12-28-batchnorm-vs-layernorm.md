---
title: "BatchNorm and LayerNorm: which values share statistics?"
description: "Normalize the same four values by columns and rows, then change companion examples, inference state, and RMS scaling."
date: "2025-12-28"
draft: false
tags: ["concepts"]
category: "concepts"
---


Normalization rescales groups of activations using statistics shared within each group.
Which other values can change one activation's output?

## Summary

Specify both the reduction axes and the axes sharing learned affine parameters.
They answer different questions: which values determine the mean, and which outputs share a scale or shift?
BatchNorm usually groups a channel across examples and spatial positions.
Transformer LayerNorm usually groups features within one token.

## The general form

For centered normalization, use this reference expression:

```
y = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta
```

The group supplies mean $\mu$ and population variance $\sigma^2$.
Learned $\gamma,\beta$ apply afterward; positive $\epsilon$ protects the denominator.

For `(N, C, H, W)`, the dimensions denote batch, channel, height, and width.
For transformer `(B, T, D)`, they denote batch, token, and embedding feature.

| Norm | Stats computed across | &gamma;, &beta; shape |
|---|---|---|
| **BatchNorm (BN)** | N, H, W (per channel) | C |
| **LayerNorm in CNNs (LN-CNN)** | C, H, W (per sample) | C, H, W |
| **LayerNorm in transformers (LN-TX)** | D (per token) | D |
| **InstanceNorm** | H, W (per sample, per channel) | C |
| **GroupNorm** | C/G channels within one group, H, W | C |

These affine shapes are common configurations, not properties forced by the reduction alone.
Framework settings can disable affine parameters or choose a different normalized shape.

<!-- visual:normalization-shared-statistics -->
<figure class="learning-figure" aria-labelledby="normalization-axes-title">
	<p class="visual-kicker">Spatial intuition</p>
	<p class="visual-title" id="normalization-axes-title">Color groups values that share one mean and variance.</p>
	<div class="visual-grid--two">
		<section class="visual-panel" aria-labelledby="batchnorm-panel-title">
			<h4 id="batchnorm-panel-title">BatchNorm on (N, C, H, W)</h4>
			<p>Each column is one channel across samples. Every square also represents all H×W positions.</p>
			<div class="norm-axis-key"><strong>columns:</strong> C0 · C1 · C2 <strong>rows:</strong> sample N</div>
			<div class="norm-matrix norm-matrix--batch" role="img" aria-label="Four samples by three channels. Values in each channel column share one mean and variance across batch and spatial positions.">
				<span class="norm-cell norm-c0">N0 C0</span><span class="norm-cell norm-c1">N0 C1</span><span class="norm-cell norm-c2">N0 C2</span>
				<span class="norm-cell norm-c0">N1 C0</span><span class="norm-cell norm-c1">N1 C1</span><span class="norm-cell norm-c2">N1 C2</span>
				<span class="norm-cell norm-c0">N2 C0</span><span class="norm-cell norm-c1">N2 C1</span><span class="norm-cell norm-c2">N2 C2</span>
				<span class="norm-cell norm-c0">N3 C0</span><span class="norm-cell norm-c1">N3 C1</span><span class="norm-cell norm-c2">N3 C2</span>
			</div>
			<p class="norm-brace">One (μ<sub>c</sub>, σ<sub>c</sub>) per colored channel, shared across N, H, and W.</p>
			<p class="norm-behavior">Train: batch statistics. Eval: stored running statistics.</p>
		</section>
		<section class="visual-panel" aria-labelledby="layernorm-panel-title">
			<h4 id="layernorm-panel-title">LayerNorm on transformer (B, T, D)</h4>
			<p>Each row is one token. Its embedding dimensions share statistics only with each other.</p>
			<div class="norm-axis-key"><strong>columns:</strong> D0…D5 <strong>rows:</strong> token (b, t)</div>
			<div class="norm-matrix norm-matrix--layer" role="img" aria-label="Four token rows by six embedding dimensions. Each token row computes its own mean and variance across embedding dimensions, independently of other tokens and samples.">
				<span class="norm-cell norm-row-a">D0</span><span class="norm-cell norm-row-a">D1</span><span class="norm-cell norm-row-a">D2</span><span class="norm-cell norm-row-a">D3</span><span class="norm-cell norm-row-a">D4</span><span class="norm-cell norm-row-a">D5</span>
				<span class="norm-cell norm-row-b">D0</span><span class="norm-cell norm-row-b">D1</span><span class="norm-cell norm-row-b">D2</span><span class="norm-cell norm-row-b">D3</span><span class="norm-cell norm-row-b">D4</span><span class="norm-cell norm-row-b">D5</span>
				<span class="norm-cell norm-row-c">D0</span><span class="norm-cell norm-row-c">D1</span><span class="norm-cell norm-row-c">D2</span><span class="norm-cell norm-row-c">D3</span><span class="norm-cell norm-row-c">D4</span><span class="norm-cell norm-row-c">D5</span>
				<span class="norm-cell norm-row-d">D0</span><span class="norm-cell norm-row-d">D1</span><span class="norm-cell norm-row-d">D2</span><span class="norm-cell norm-row-d">D3</span><span class="norm-cell norm-row-d">D4</span><span class="norm-cell norm-row-d">D5</span>
			</div>
			<p class="norm-brace">One (μ<sub>b,t</sub>, σ<sub>b,t</sub>) per colored token row, computed across D.</p>
			<p class="norm-behavior">Train and eval: the same per-token computation.</p>
		</section>
	</div>
	<p>The numeric trace below uses rows [1, 3] and [5, 7]. The blocks illustrate grouping, not those four values.</p>
	<figcaption><strong>Read it this way:</strong> BatchNorm shares statistics down a channel column and therefore couples examples. Transformer LayerNorm shares statistics across one token row and never mixes tokens or samples. <span class="visual-reader-check" data-reader-check="normalization-shared-statistics"><strong>Check:</strong> Which other sample can change a BatchNorm value's statistics but not a LayerNorm token's statistics?</span></figcaption>
</figure>

### Normalize four values

Use $X=\begin{bmatrix}1&3\\5&7\end{bmatrix}$ with affine scale one, shift zero, and $\epsilon=0$ for hand arithmetic.
All reference groups have positive variance.

Column means are $[3,5]$, with variances $[4,4]$.
Row means are $[2,6]$, with variances $[1,1]$.
Therefore

$$
\operatorname{BN}(X)=\begin{bmatrix}-1&-1\\1&1\end{bmatrix},\qquad
\operatorname{LN}(X)=\begin{bmatrix}-1&1\\-1&1\end{bmatrix}.
$$

Each normalized group has mean zero and second moment one before the affine step.
With positive epsilon, its second moment is $\sigma^2/(\sigma^2+\epsilon)$, slightly below one here.

### Change the companion example

Keep the first row $[1,3]$ but replace the second with $[0,2]$.
Column means become $[0.5,2.5]$ and variances become $[0.25,0.25]$.
BatchNorm's first row flips to $[1,1]$.
Its LayerNorm output remains $[-1,1]$ because its row group did not change.
This distinguishes batch-composition sensitivity from token-local normalization.

### Carry statistics into evaluation

BatchNorm normally stores running statistics during training and uses them during evaluation.
Those stored statistics need not equal a current batch's statistics.
With stored mean $[3,5]$ and variance $[4,4]$, the first row remains $[-1,-1]$ despite changed companions.
Check framework settings and running-variance conventions; disabling stored statistics can retain batch dependence at evaluation.
LayerNorm recomputes its configured groups in both modes.

BatchNorm's reduction count is $NHW$ per channel, not just $N$.
Correlation, group size, and distribution shift matter; no universal $N<16$ failure threshold follows.
An implementation may reject a training group containing only one scalar.

## The transformer-specific wrinkle most people miss

With normalized shape `D`, each `(b, t)` token owns its statistics.
Other tokens, examples, and padding tokens do not enter that reduction.
Padding still needs correct attention and loss masks.
Reducing over `(T, D)` instead mixes tokens, including padding or future positions.
Batch reductions over sequence positions require a deliberately compatible masking and causality policy; packing alone does not prove failure.

## RMSNorm: the modern variant

Remove centering and normalize by mean square instead:

```
y = gamma * x / sqrt(mean(x^2) + eps)
```

For token $[1,3]$ with unit scale and zero epsilon, mean square is $5$.
RMSNorm returns $[1/\sqrt5,3/\sqrt5]\approx[0.4472,1.3416]$, not $[-1,1]$.
Adding a constant to every feature generally changes that result.
Centered LayerNorm is translation invariant before its affine step; RMSNorm is not.
Choose the operation specified by the architecture, not a claimed universal speed or quality improvement.

### Failure and transfer

A constant group has zero centered variance.
Positive epsilon prevents division by zero but cannot recover feature differences; a one-scalar LayerNorm returns its affine shift.

**Reconstruct:** does LayerNorm with normalized shape `(T, D)` remain token-local?

<details><summary>Answer</summary>

No. The reduction includes tokens as well as features.
Replacing another token changes the group's mean and variance, as the companion-row trace demonstrates.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

## What an interviewer expects you to say

Declare tensor shape, reduction groups, affine sharing, and train/evaluation state.
Compute one group's statistics and ask which input changes can affect it.
Then distinguish the normalization operation from its position in a larger block.

## Pre-norm vs post-norm

For residual branch $F$, these are different computations:

$$
{}\text{post-norm: }y=\operatorname{LN}(x+F(x)),\qquad
{}\text{pre-norm: }y=x+F(\operatorname{LN}(x)).
$$

Pre-norm leaves an unnormalized identity path across this block.
That changes gradient propagation independently of the reduction axes.
Depth, optimizer, warmup, residual scaling, and final normalization affect the architecture comparison.
Neither equation proves universal quality or stability dominance.

## Common confusions

- LayerNorm's configured feature group need not equal an entire sample.
- Batch size one does not turn BatchNorm into LayerNorm; reduction axes still differ.
- Changing reduction axes changes the model's batch coupling, not merely notation.
- Batch-statistic noise can regularize, but its benefit is workload dependent.
- Internal covariate shift was an early motivation, not a settled universal explanation of BatchNorm's optimization effects.

## Why interviewers care

The reconstruction tests shapes, broadcasting, shared state, and inference behavior.
Architecture names or memorized adoption trends cannot replace that calculation.

---

*Related: [FlashAttention](/concepts/flashattention/) for the other transformer-specific optimization that's often paired with this in questions.*

</div>
