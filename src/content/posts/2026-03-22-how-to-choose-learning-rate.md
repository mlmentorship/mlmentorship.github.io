---
title: "How do you choose a learning rate?"
description: "Check update stability, propose a learning-rate peak, then test its schedule from a clean starting state."
date: "2026-03-22"
draft: false
tags: ["questions"]
category: "questions"
---


Choose a learning-rate policy by comparing update behavior and validating a complete training schedule rather than a single sweep result.

## Problem

**Context.** The exact one-dimensional SGD fixture below isolates learning-rate behavior before considering noisy, scheduled training.

**Task.** Compare the three constant rates, explain convergence or divergence, and propose a controlled learning-rate selection procedure.
Describe how warmup, decay, optimizer choice, and intended run length affect the final comparison.

**Constraints.** Use the stated loss, gradient, and starting point for the fixture; do not add momentum or stochastic noise.
A short range test proposes candidates rather than proving the best full-run schedule.

Use plain SGD on $L(w)=5w^2$, starting at $w_0=1$ with exact gradient $10w$.
Compare constant rates $0.1$, $0.2$ and $0.25$.
Which rates reduce the loss, and what changes when the intended run uses a schedule?

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-sounds-like"></span>

<!-- visual:learning-rate-range-to-schedule -->
<figure class="learning-figure" aria-labelledby="learning-rate-procedure-title">
	<p class="visual-kicker">Candidate first, controlled run second</p>
	<p class="visual-title" id="learning-rate-procedure-title">Choose the useful descent region, not the unstable end of the sweep</p><div class="visual-argument" data-visual-argument="learning-rate-range-to-schedule" data-argument-kind="curve" data-argument-composition="iterative-step"><svg viewBox="0 0 360 750" role="img" aria-label="Choose the useful descent region before instability. For loss x squared, the update multiplies x by one minus twice the learning rate. These three-step fixtures contract, oscillate, or grow. A range test still requires observed task evidence."><title>Choose the useful descent region before instability</title><desc>For loss x squared, the update multiplies x by one minus twice the learning rate. These three-step fixtures contract, oscillate, or grow. A range test still requires observed task evidence.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Same quadratic; different update</tspan><tspan x="20" dy="18">distances</tspan></text><text x="20" y="89" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">Rate 0.1: multiply the coordinate by</tspan><tspan x="20" dy="18">0.8</tspan></text><path d="M35 171L325 171" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M180 121L180 222" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><g data-picture-update-step="0,0" data-picture-coordinate="1"><circle cx="256" cy="141" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-update-step="0,1" data-picture-coordinate="0.8"><circle cx="240.8" cy="161" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path d="M256 141L240.8 161M242.45601736377606 152.21036937688095L240.8 161L248.8253149167609 157.05103551714942" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-update-step="0,2" data-picture-coordinate="0.6400000000000001"><circle cx="228.64000000000001" cy="181" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path d="M240.8 161L228.64000000000001 181M229.37825555242662 172.08624777440437L228.64000000000001 181L236.2139551119325 176.24235310658392" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-update-step="0,3" data-picture-coordinate="0.5120000000000001"><circle cx="218.912" cy="201" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path d="M228.64000000000001 181L218.912 201M218.81415989382407 192.05626323544664L218.912 201L226.00828534793715 195.5554858563272" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="20" y="300" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Rate 0.9: multiply the coordinate by</tspan><tspan x="20" dy="18">-0.8</tspan></text><path d="M35 382L325 382" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M180 332L180 433" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><g data-picture-update-step="1,0" data-picture-coordinate="1"><circle cx="256" cy="352" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-update-step="1,1" data-picture-coordinate="-0.8"><circle cx="119.19999999999999" cy="372" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><path d="M256 352L119.19999999999999 372M126.53720617961626 366.8847868590057L119.19999999999999 372L127.6944942205652 374.80063705909646" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-update-step="1,2" data-picture-coordinate="0.6400000000000001"><circle cx="228.64000000000001" cy="392" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><path d="M119.19999999999999 372L228.64000000000001 392M220.05124739639422 394.4966635163864L228.64000000000001 392L221.48941762472742 386.6269960269472" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-update-step="1,3" data-picture-coordinate="-0.5120000000000001"><circle cx="141.088" cy="412" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><path d="M228.64000000000001 392L141.088 412M147.99630188178085 406.31885881972795L141.088 412L149.77789407328615 414.11795679726146" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="20" y="511" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Rate 1.1: multiply the coordinate by</tspan><tspan x="20" dy="18">-1.2</tspan></text><path d="M35 593L325 593" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M180 543L180 644" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><g data-picture-update-step="2,0" data-picture-coordinate="1"><circle cx="256" cy="563" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle></g><g data-picture-update-step="2,1" data-picture-coordinate="-1.2000000000000002"><circle cx="88.79999999999998" cy="583" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle></g><path d="M256 563L88.79999999999998 583M96.26829165938665 578.0781487537375L88.79999999999998 583L97.21845599264196 586.0215225797518" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-update-step="2,2" data-picture-coordinate="1.4400000000000004"><circle cx="289.44000000000005" cy="603" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle></g><path d="M88.79999999999998 583L289.44000000000005 603M281.082693662641 606.1867586641506L289.44000000000005 603L281.8762092662642 598.226210128603" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-update-step="2,3" data-picture-coordinate="-1.7280000000000006"><circle cx="48.67199999999994" cy="623" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle></g><path d="M289.44000000000005 603L48.67199999999994 623M56.313411571862424 618.3514702120557L48.67199999999994 623L56.97567077347292 626.3240113847233" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="20" y="700" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Oscillation can contract; oversized</tspan><tspan x="20" dy="18">steps grow.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption">the first plot proposes a rate before instability. The second assigns that candidate to a schedule. These qualitative curves are not measured results or proof of an optimal peak.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="learning-rate-range-to-schedule"><strong>Check:</strong> Which LR from the sweep becomes the schedule peak? Why is the unstable endpoint not the choice?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="learning-rate-range-to-schedule">For loss x squared, the update multiplies x by one minus twice the learning rate. These three-step fixtures contract, oscillate, or grow. A range test still requires observed task evidence. <br><strong>Does crossing the minimum necessarily mean the iterations diverge?</strong></span></figcaption>
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
