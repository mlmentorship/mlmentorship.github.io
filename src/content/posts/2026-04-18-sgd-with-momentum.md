---
title: "SGD with momentum: keep the update history"
description: "Trace the saved gradient sum from startup to steady behavior, then change history length and learning-rate placement."
date: "2026-04-18"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Momentum saves a decaying sum of gradients and uses it to choose the next parameter step.
It is useful when successive gradients contain persistent direction, but history can also delay a necessary reversal.

**Task:** filter four prescribed descent directions $(1,2),(1,-2),(1,2),(1,-2)$ with retention $\beta=1/2$ and learning rate $\eta=1$.
Start the parameter and buffer at zero.
What changes relative to ordinary SGD?

Write gradient $g_t$, descent direction $d_t=-g_t$, gradient buffer $v_t$, and descent buffer $u_t=-v_t$.
The figure shows the warmed-up response, not the first four updates from zero.
These are prescribed inputs, not gradients reevaluated along two measured optimization trajectories.

<!-- visual:momentum-alternating-components -->
<figure class="learning-figure" aria-labelledby="momentum-components-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="momentum-components-title">In this recurrence, momentum reinforces persistent motion and damps alternating motion</p><div class="visual-argument" data-visual-argument="momentum-alternating-components" data-argument-kind="curve" data-argument-composition="landmark"><svg viewBox="0 0 360 439" role="img" aria-label="Persistent directions accumulate while alternating directions cancel. With beta=0.5 and v=beta v+g, constant gradients approach 2. Alternating gradients remain bounded and repeatedly reverse direction."><title>Persistent directions accumulate while alternating directions cancel</title><desc>With beta=0.5 and v=beta v+g, constant gradients approach 2. Alternating gradients remain bounded and repeatedly reverse direction.</desc><g opacity=".35"><path d="M48 177.5L330 177.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 135L330 135" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 92.5L330 92.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">0</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">4</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">2</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">-1</tspan></text><path data-proof-landmark="4" d="M330 40V220" style="stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:2 4"></path><path data-proof-series="Constant gradients +1" d="M48 163.33333333333334L118.5 106.66666666666667L189 78.33333333333331L259.5 64.16666666666669L330 57.083333333333314" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3"></path><g data-proof-observation="0" data-proof-x="4" data-proof-y="1.875"><circle cx="330" cy="57.083333333333314" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path data-proof-series="Alternating gradients +1,-1" d="M48 163.33333333333334L118.5 106.66666666666667L189 191.66666666666666L259.5 120.83333333333333L330 198.75" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:7 4"></path><g data-proof-observation="1" data-proof-x="4" data-proof-y="-0.625"><circle cx="330" cy="198.75" r="5" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle></g><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">update number</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">velocity under the stated recurrence</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Constant gradients +1</tspan></text><text x="60" y="333" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="60" dy="0">1.875 at 4</tspan></text><path d="M20 368L48 368" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="372" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Alternating gradients +1,-1</tspan></text><text x="60" y="395" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="60" dy="0">-0.625 at 4</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-grid--two" role="group" aria-label="Raw SGD and momentum trajectories on the same coordinate scale">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 235" role="img" aria-labelledby="momentum-raw-title momentum-raw-desc">
				<title id="momentum-raw-title">Raw SGD follows alternating descent directions</title>
				<desc id="momentum-raw-desc">On a common coordinate scale, four raw SGD updates alternate between the vectors one comma two and one comma negative two. The path zigzags from zero comma zero through one comma two, two comma zero, three comma two, and four comma zero.</desc>
				<rect class="viz-plot-bg" x="30" y="28" width="244" height="152" rx="5"></rect>
				<text class="viz-axis-label" x="152" y="17" text-anchor="middle">RAW SGD · SAME SCALE</text>
				<path d="M40 130H262M40 42V174" class="viz-axis"></path>
				<path d="M40 66H262M40 98H262M40 130H262M40 162H262" class="viz-gridline"></path>
				<text class="viz-label" x="25" y="70" text-anchor="end">+2</text>
				<text class="viz-label" x="25" y="134" text-anchor="end">0</text>
				<text class="viz-label" x="25" y="166" text-anchor="end">−1</text>
				<text class="viz-label" x="40" y="194" text-anchor="middle">0</text>
				<text class="viz-label" x="94" y="194" text-anchor="middle">2</text>
				<text class="viz-label" x="148" y="194" text-anchor="middle">4</text>
				<text class="viz-label" x="202" y="194" text-anchor="middle">6</text>
				<text class="viz-label" x="256" y="194" text-anchor="middle">8</text>
				<path d="M40 130L67 66L94 130L121 66L148 130" style="fill:none;stroke:var(--viz-edge);stroke-width:3;stroke-dasharray:6 4;stroke-linejoin:round"></path>
				<g style="fill:var(--viz-neutral-bg);stroke:var(--viz-edge);stroke-width:2"><circle cx="40" cy="130" r="4"></circle><circle cx="67" cy="66" r="4"></circle><circle cx="94" cy="130" r="4"></circle><circle cx="121" cy="66" r="4"></circle><circle cx="148" cy="130" r="4"></circle></g>
				<text class="viz-callout" x="40" y="219">Δθ = (1, +2), (1, −2), …</text>
				<text class="viz-label" x="164" y="115">ends at x = 4</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 235" role="img" aria-labelledby="momentum-smoothed-title momentum-smoothed-desc">
				<title id="momentum-smoothed-title">Momentum amplifies consistent progress relative to oscillation</title>
				<desc id="momentum-smoothed-desc">For beta one half after the momentum buffer reaches its steady alternating regime, the update vectors alternate between two comma four thirds and two comma negative four thirds. On the same scale as raw SGD, four updates trace zero comma zero, two comma four thirds, four comma zero, six comma four thirds, and eight comma zero.</desc>
				<rect class="viz-plot-bg" x="30" y="28" width="244" height="152" rx="5"></rect>
				<text class="viz-axis-label" x="152" y="17" text-anchor="middle">MOMENTUM · β=0.5 · WARMED UP</text>
				<path d="M40 130H262M40 42V174" class="viz-axis"></path>
				<path d="M40 66H262M40 98H262M40 130H262M40 162H262" class="viz-gridline"></path>
				<text class="viz-label" x="25" y="70" text-anchor="end">+2</text>
				<text class="viz-label" x="25" y="134" text-anchor="end">0</text>
				<text class="viz-label" x="25" y="166" text-anchor="end">−1</text>
				<text class="viz-label" x="40" y="194" text-anchor="middle">0</text>
				<text class="viz-label" x="94" y="194" text-anchor="middle">2</text>
				<text class="viz-label" x="148" y="194" text-anchor="middle">4</text>
				<text class="viz-label" x="202" y="194" text-anchor="middle">6</text>
				<text class="viz-label" x="256" y="194" text-anchor="middle">8</text>
				<path d="M40 130L94 87.33L148 130L202 87.33L256 130" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:4;stroke-linejoin:round"></path>
				<g style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"><rect x="36" y="126" width="8" height="8"></rect><rect x="90" y="83.33" width="8" height="8"></rect><rect x="144" y="126" width="8" height="8"></rect><rect x="198" y="83.33" width="8" height="8"></rect><rect x="252" y="126" width="8" height="8"></rect></g>
				<text class="viz-callout" x="40" y="219">Δθ = (2, +4/3), (2, −4/3), …</text>
				<text class="viz-label" x="190" y="148">ends at x = 8</text>
			</svg>
		</section>
	</div>
	<div class="visual-derivation-caption">use the same axes in both panels. Raw SGD follows descent directions <var>d<sub>t</sub></var> = −<var>g<sub>t</sub></var> = (1, ±2). Each step moves forward by 1 and across the valley by 2. With <var>u<sub>t</sub></var> = 0.5<var>u<sub>t−1</sub></var> + <var>d<sub>t</sub></var> after warm-up, the alternating steady-state updates are (2, ±4/3). The consistent component doubles; the cross-valley-to-forward ratio falls from 2 to 2/3. This is a filter response, not evidence of measured convergence. Compare learning rates as well as paths. This original trace was checked against <a href="https://doi.org/10.1016/0041-5553(64)90137-5">Polyak (1964)</a>, <a href="https://proceedings.mlr.press/v28/sutskever13.html">Sutskever et al. (2013)</a>, and the <a href="https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html">PyTorch SGD definition</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="momentum-alternating-components"><strong>Check:</strong> Why is the persistent component amplified while the alternating component is damped in this specific warmed-up recurrence?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="momentum-alternating-components">With beta=0.5 and v=beta v+g, constant gradients approach 2. Alternating gradients remain bounded and repeatedly reverse direction.</span></figcaption>
</figure>

### Start from zero

Compute $u_t=\beta u_{t-1}+d_t$, then $\theta_t=\theta_{t-1}+\eta u_t$:

| Step | $d_t$ | $u_t$ | $\theta_t$ |
| --- | --- | --- | --- |
| 1 | $(1,2)$ | $(1,2)$ | $(1,2)$ |
| 2 | $(1,-2)$ | $(3/2,-1)$ | $(5/2,1)$ |
| 3 | $(1,2)$ | $(7/4,3/2)$ | $(17/4,5/2)$ |
| 4 | $(1,-2)$ | $(15/8,-5/4)$ | $(49/8,5/4)$ |

Ordinary SGD ends at $(4,0)$; the startup momentum path ends at $(6.125,1.25)$.
The warmed-up figure instead ends at $(8,0)$.

For constant input $a$, the limiting buffer solves $u=\beta u+a$, giving $a/(1-\beta)$.
For alternating input $+a,-a$, let the positive-phase limit be $b$.
Then $b=\beta(-b)+a$, so $b=a/(1+\beta)$.
Here the steady components are $2$ and $\pm4/3$.

## Two formulations

### Classical momentum (Polyak, 1964)

$$
g_t=\nabla L(\theta_{t-1}),\qquad
v_t=\beta v_{t-1}+g_t,\qquad
{}\theta_t=\theta_{t-1}-\eta v_t.
$$

Assume $0\le\beta<1$, zero initial buffer, and no dampening.
Expanding the recurrence gives $v_t=\sum_{j=1}^t\beta^{t-j}g_j$.
This is a decaying sum, not a normalized exponential moving average.

For the average $m_t=\beta m_{t-1}+(1-\beta)g_t$, matching initialization gives $m_t=(1-\beta)v_t$.
An update using $m_t$ needs learning rate $\eta/(1-\beta)$ to match the sum-based update at constant $\beta$.

### Nesterov momentum (Nesterov, 1983)

**New task:** evaluate the gradient at a lookahead point.
Replace $g_t$ above with $\nabla L(\theta_{t-1}-\eta\beta v_{t-1})$; keep the parameter update based at $\theta_{t-1}$.

**Check:** for $L(\theta)=\theta^2/2$, current $\theta=2$, previous $v=1$, $\eta=0.1$, and $\beta=0.5$, lookahead is $1.95$.
The new buffer is $2.45$ and parameter $1.755$, versus classical $2.5$ and $1.75$.
**Use when:** the specified algorithm calls for lookahead; this is not merely a changed label on the same gradient.

## Picking $\beta$

**New task:** respond faster when older directions become stale.
Reduce $\beta$ and reconsider the learning rate together.

**Check:** constant gradient one gives limiting buffers $10$ at $\beta=0.9$ and $2$ at $\beta=0.5$.
At learning rate $0.1$, limiting step magnitudes are $1$ and $0.2$.
Using $0.02$ with $\beta=0.9$ matches the latter scale, but not its response speed.
**Use when:** history length changes; equal learning rates do not isolate smoothing.

### Change the learning-rate placement

**New task:** resume across a learning-rate reduction from $0.1$ to $0.05$.
With $v_{old}=2$, $g=1$, and $\beta=0.5$, the gradient buffer stays $2$; the new step magnitude is $0.1$.
The stored gradients do not contain the old learning rate.

For a displacement buffer $p_t=\beta p_{t-1}+\eta_tg_t$, old $p=0.2$ instead gives $p_t=0.15$.
**Use when:** resuming or translating implementations; preserve state according to the actual recurrence, not a universal reset rule.

### Failure and transfer

**Reconstruct:** why is $(2,\pm4/3)$ not the zero-buffer startup trace?

<details><summary>Answer</summary>

Those values solve steady-state equations after transients decay.
The four-row ledger retains the missing initial history explicitly; its final parameter differs from the warmed-up figure.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

## When SGD+momentum vs. Adam

Momentum changes the history filter; [Adam](/concepts/adam-and-adamw/) also tracks coordinatewise squared gradients.
Neither mechanism establishes a universal speed or generalization ordering.
Compare tuned learning rates, schedules, regularization, stopping criteria, compute budgets, and repeated seeds on the target workload.

The scale $1/(1-\beta)$ is a useful history horizon, not an exact count of equally weighted observations.
For independent scalar gradient noise of variance $\sigma^2$, the stationary sum has variance $\sigma^2/(1-\beta^2)$.
The normalized average instead has variance $\sigma^2(1-\beta)/(1+\beta)$.
These noise assumptions do not describe every minibatch process.

## Common pitfalls

- A zero buffer creates startup transients, not an error that every momentum implementation must bias-correct.
- A zero current gradient does not imply a zero update while history remains.
- Larger retention can amplify overshoot as well as persistent progress.
- Frameworks differ in dampening, first-buffer initialization, Nesterov parameterization, and learning-rate placement.
- Algebraically related Nesterov forms require matching state definitions and schedule assumptions before comparing saved buffers.
- Resume checkpoints must preserve parameters, optimizer buffers, step count, schedule state, and any random or sampling state needed for replay.

</div>
