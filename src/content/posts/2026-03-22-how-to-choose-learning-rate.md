---
title: "How do you choose a learning rate?"
description: "Check update stability, propose a learning-rate peak, then test its schedule from a clean starting state."
date: "2026-03-22"
draft: false
tags: ["questions"]
category: "questions"
---


A learning rate controls the scale of an optimizer update.
A short sweep can propose a candidate, but it does not establish a complete training recipe.

## Problem

Use plain SGD on $L(w)=5w^2$, starting at $w_0=1$ with exact gradient $10w$.
Compare constant rates $0.1$, $0.2$ and $0.25$.
Which rates reduce the loss, and what changes when the intended run uses a schedule?

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-sounds-like"></span>

<!-- visual:learning-rate-range-to-schedule -->
<figure class="learning-figure" aria-labelledby="learning-rate-procedure-title">
	<p class="visual-kicker">Candidate first, controlled run second</p>
	<p class="visual-title" id="learning-rate-procedure-title">Turn a range test into a peak learning rate, then schedule around it</p>
	<div class="visual-grid--two" role="group" aria-label="Two-stage learning-rate selection procedure">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 238" role="img" aria-labelledby="lr-range-title lr-range-desc">
				<title id="lr-range-title">Read a learning-rate range test from too small through useful descent to instability</title>
				<desc id="lr-range-desc">An original qualitative plot shows training loss against learning rate on a logarithmic horizontal axis. At very small rates the loss is nearly flat. It then falls, with a hatched candidate band around the sustained steep descent. Farther right the curve bottoms and turns sharply upward in an instability region marked with crosses. The selected candidate is before the minimum and blow-up, not at either one.</desc>
				<defs>
					<pattern id="lr-candidate-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
						<line x1="0" y1="0" x2="0" y2="6" style="stroke:var(--viz-focus-stroke);stroke-width:2"></line>
					</pattern>
				</defs>
				<rect class="viz-plot-bg" x="38" y="28" width="248" height="158" rx="4"></rect>
				<rect x="125" y="28" width="64" height="158" style="fill:url(#lr-candidate-hatch);opacity:.24"></rect>
				<path class="viz-gridline" d="M38 81H286M38 133H286M100 28V186M162 28V186M224 28V186"></path>
				<path class="viz-axis" d="M38 28V186H286"></path>
				<path class="viz-pr-curve" d="M45 61C76 62 94 60 112 70C134 83 151 112 171 139C187 160 205 169 218 160C235 149 246 103 259 57C267 35 275 30 282 34"></path>
				<circle class="viz-operating-point" cx="158" cy="121" r="5"></circle>
				<path class="viz-operating-guide" d="M158 121V186"></path>
				<path d="M252 47L262 57M262 47L252 57M267 31L277 41M277 31L267 41" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2"></path>
				<text class="viz-axis-label" x="42" y="18">1 · SWEEP LR UP; WATCH LOSS</text>
				<text class="viz-label" x="69" y="48" text-anchor="middle">too small</text>
				<text class="viz-callout" x="157" y="102" text-anchor="middle">candidate</text>
				<text class="viz-callout" x="157" y="115" text-anchor="middle">steep descent</text>
				<text class="viz-label" x="247" y="82" text-anchor="middle">unstable</text>
				<text class="viz-label" x="38" y="204">10⁻⁷</text>
				<text class="viz-label" x="151" y="204">10⁻⁴</text>
				<text class="viz-label" x="268" y="204">10⁻¹</text>
				<text class="viz-axis-label" x="162" y="228" text-anchor="middle">learning rate (log scale) →</text>
				<text class="viz-axis-label" transform="translate(14 132) rotate(-90)">training loss</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 238" role="img" aria-labelledby="lr-schedule-title lr-schedule-desc">
				<title id="lr-schedule-title">Use the selected candidate as the peak of a warmup and decay schedule</title>
				<desc id="lr-schedule-desc">A learning-rate schedule begins low, rises linearly during a short warmup to the candidate peak selected by the range test, then follows a curved decay over the remaining training steps to a lower final rate. A vertical dotted guide marks the end of warmup, and the peak has a double-ring marker. The figure emphasizes that the selected value is a peak, not a constant rate used for every step.</desc>
				<rect class="viz-plot-bg" x="38" y="28" width="248" height="158" rx="4"></rect>
				<path class="viz-gridline" d="M38 81H286M38 133H286M100 28V186M162 28V186M224 28V186"></path>
				<path class="viz-axis" d="M38 28V186H286"></path>
				<path d="M45 174L89 58C132 60 174 70 211 92C241 110 263 133 282 154" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5"></path>
				<path class="viz-operating-guide" d="M89 42V186"></path>
				<circle class="viz-node viz-node--focus" cx="89" cy="58" r="8"></circle>
				<circle class="viz-node" cx="89" cy="58" r="4"></circle>
				<text class="viz-axis-label" x="42" y="18">2 · USE IT AS THE PEAK</text>
				<text class="viz-callout" x="65" y="104" text-anchor="middle">warmup</text>
				<text class="viz-callout" x="89" y="43" text-anchor="middle">selected peak</text>
				<text class="viz-callout" x="205" y="78" text-anchor="middle">decay</text>
				<text class="viz-label" x="57" y="204" text-anchor="middle">start</text>
				<text class="viz-label" x="89" y="204" text-anchor="middle">warmup end</text>
				<text class="viz-label" x="274" y="204" text-anchor="middle">finish</text>
				<text class="viz-axis-label" x="162" y="228" text-anchor="middle">training step →</text>
				<text class="viz-axis-label" transform="translate(14 132) rotate(-90)">learning rate</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> the first plot proposes a rate before instability. The second assigns that candidate to a schedule. These qualitative curves are not measured results or proof of an optimal peak.</figcaption>
</figure>
<p class="diagram-source">Range-test references: <a href="https://arxiv.org/abs/1506.01186">Smith (2017)</a> and the <a href="https://docs.fast.ai/callback.schedule.html#lrfind">fastai learning-rate finder documentation</a>.</p>

## Calculate the updates

For plain SGD, $w_{t+1}=w_t-\eta\nabla L(w_t)=(1-10\eta)w_t$.
Contraction requires $|1-10\eta|<1$, so $0<\eta<0.2$.

| Rate | $w_0,w_1,w_2$ | $L(w_0),L(w_1),L(w_2)$ | Decision |
| --- | --- | --- | --- |
| $0.1$ | $1,0,0$ | $5,0,0$ | Reaches the minimum |
| $0.2$ | $1,-1,1$ | $5,5,5$ | Oscillates without progress |
| $0.25$ | $1,-1.5,2.25$ | $5,11.25,25.3125$ | Diverges |

Finite values alone do not establish progress. The boundary rate $0.2$ never improves this loss.
This calculation is not an Adam benchmark or a universal neural-network bound.

## Test a candidate

Record the optimizer, loss reduction, batch, initialization, precision and intended training length.
Use a disposable run to increase rates while recording loss and update behavior.
For $N\ge2$ sweep steps, a geometric sweep can use

$$
\eta_j=\eta_{\mathrm{low}}\left(\frac{\eta_{\mathrm{high}}}{\eta_{\mathrm{low}}}\right)^{j/(N-1)},\qquad j=0,\ldots,N-1.
$$

Choose a candidate in sustained descent, with margin before instability.
Restore the starting model and optimizer state, then test the intended schedule on controlled runs.
Compare representative validation results and stability at the same budget.
A sweep changes weights along its path; its points are not independent rate trials.

## Change the run

### 1. Double the curvature

**New task:** replace the loss by $10w^2$.

**Picture and change:** the gradient becomes $20w$. Recompute contraction from $|1-20\eta|<1$.

**Check:** the interval becomes $0<\eta<0.1$. The old rate $0.1$ now oscillates without progress.

**Use when:** loss scaling, parameterization or optimizer changes the effective update.

### 2. Specify a shorter training horizon

**New task:** use $S=1000$ steps, $W=100$ warmup steps, peak $0.001$ and final rate $0.0001$.

**Picture and change:** choose the schedule separately from the peak.
At schedule position $t\in[0,S]$, use

$$
\eta(t)=\begin{cases}
0.001t/W,&0\le t\le W,\\
0.0001+\frac{0.0009}{2}\left[1+\cos\left(\pi\frac{t-W}{S-W}\right)\right],&W<t\le S.
\end{cases}
$$

**Check:** positions $50$, $100$, $550$ and $1000$ give rates $0.0005$, $0.001$, $0.00055$ and $0.0001$.

**Use when:** the step budget changes. These fractions describe this fixture, not a universal schedule.

## Do not continue from a failed sweep

Diverged weights and accumulated Adam moments are not a clean starting state.
Reload model, optimizer and schedule state; control data order and randomness for the comparison.
Otherwise a failed sweep can invalidate the candidate test.

## Try changing the batch

Doubling batch size also changes loss reduction from mean to sum. What must be checked first?

<details>
<summary>Check your answer</summary>

Restore the intended reduction or account for its doubled gradient scale.
Only then test a batch-scaling heuristic under controlled conditions.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

<span id="what-an-l6-answer-sounds-like"></span>
<span id="the-tells-that-get-you-a-strong-hire-vote"></span>
<span id="the-tells-that-get-you-down-leveled"></span>

## Optimizer and training regime

Adam uses accumulated, coordinate-dependent moments, so its effective update is not simply plain SGD's gradient times a rate.
Batch-scaling rules require assumptions. Neither linear scaling for SGD nor square-root scaling for Adam is universal.
The [large-minibatch SGD recipe](https://arxiv.org/abs/1706.02677) combines scaling with specific training conditions and warmup.

Fine-tuning must balance adaptation against preserving useful behavior.
Compare layer updates and held-out behavior instead of assuming pretrained weights are well-conditioned.
Fixed fine-tuning, SFT or preference-training rate ranges do not transfer automatically between models and objectives.

## Schedules and diagnostics

Warmup limits early updates; its usefulness and mechanism depend on optimizer, parameterization and model.
The [Transformer schedule](https://arxiv.org/abs/1706.03762), cosine decay, step decay and one-cycle use different time profiles.
Compare them at a stated training budget rather than declaring one always superior.

If every reasonable rate fails, inspect data, initialization, gradient scale, loss implementation and numerical precision.
Reducing the rate does not repair incorrect labels or a broken backward pass.

Related: [debug a model that is not learning](/questions/debug-model-not-learning/) and [bias and variance](/questions/bias-variance-tradeoff/).

</div>
