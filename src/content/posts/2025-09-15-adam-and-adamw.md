---
title: "Adam: trace direction and coordinate scale"
description: "Compute two moment updates, reconstruct a step with zero current gradient, and separate bias correction, epsilon, decay, and state memory."
date: "2025-09-15"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Adam saves a smoothed gradient and a smoothed squared gradient for each parameter.
It divides directional history by a coordinate-scale estimate.
Bias correction compensates for zero-initialized buffers, not gradient noise in general.

**Task:** start $\theta_0=[1,1]$, $m_0=v_0=[0,0]$, $\beta_1=\beta_2=0.5$, and learning rate $\eta=0.1$.
Apply prescribed gradients $[2,4]$, then $[2,0]$.
Use $\epsilon=0$ only for this nonzero-denominator fixture; production implementations use positive epsilon.
Does the second coordinate stop when its current gradient becomes zero?

## The mechanism

All operations below are coordinatewise; the step counter $t$ counts optimizer updates, starting at one.
[Momentum](/concepts/sgd-with-momentum/) saves a decaying sum; Adam's first moment includes the additional factor $(1-\beta_1)$.
Its second moment averages $g_t^2$, not squared deviations from a mean.

| Step | $g_t$ | $m_t$ | $v_t$ | $\widehat m_t$ | $\widehat v_t$ |
| --- | --- | --- | --- | --- | --- |
| 1 | $[2,4]$ | $[1,2]$ | $[2,8]$ | $[2,4]$ | $[4,16]$ |
| 2 | $[2,0]$ | $[3/2,1]$ | $[3,4]$ | $[2,4/3]$ | $[4,16/3]$ |

At step one, divide each buffer by $1-0.5=0.5$.
Both update magnitudes are $0.1$, giving $\theta_1=[0.9,0.9]$.
At step two, divide by $1-0.5^2=0.75$.
The second displacement is

$$
0.1\frac{4/3}{\sqrt{16/3}}=\frac{0.1}{\sqrt3}\approx0.057735027.
$$

Thus $\theta_2\approx[0.8,0.842264973]$.
The vanished current gradient does not erase saved directional history.

The complete state update is:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2

# bias correction (for early steps when m, v are biased toward 0)
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)

theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
```

The [Adam paper](https://arxiv.org/abs/1412.6980) uses this outside-the-square-root epsilon convention.
Actual APIs may expose additional choices; inspect the specified implementation.

Why those correction factors?
For constant gradient mean $\mu$ and zero initial state, $\mathbb E[m_t]=(1-\beta_1^t)\mu$.
Likewise, constant raw second moment $\nu$ gives $\mathbb E[v_t]=(1-\beta_2^t)\nu$.
The denominators remove those initialization factors.
Under changing distributions, corrected buffers still represent weighted history, not unbiased estimates of every current quantity.
The ratio of corrected moments is not thereby an unbiased update.

### Remove first-moment memory

**New task:** set $\beta_1=0$, retaining the same gradients and $\beta_2=0.5$.
Now $\widehat m_2=[2,0]$, while $\widehat v_2=[4,16/3]$ is unchanged.
**Check:** $\theta_2=[0.8,0.9]$; the second coordinate no longer moves on step two.
**Use when:** testing whether saved directional history causes movement after a gradient disappears.

### Enter epsilon's scale

**New task:** use first gradient $[10^{-9},1]$ and $\epsilon=10^{-8}$ with the same learning rate.
The first-step coordinate ratio is $g/(|g|+\epsilon)$, not merely its sign.
**Check:** $\theta_1\approx[0.990909091,0.900000001]$, rather than $[0.9,0.9]$.
**Use when:** gradient magnitudes approach epsilon; scale cancellation is then only an approximation.

### Failure and transfer

Clipping the current gradient does not erase $m_t$ or directly bound the complete Adam displacement.
Even a wholly zero current gradient can leave a nonzero moment-driven step.
Normalization also does not guarantee decreasing loss on each update.

**Reconstruct:** why is $v_t$ not the gradient variance?

<details><summary>Answer</summary>

It averages raw squares.
Variance would subtract a squared mean under compatible weighting and assumptions; Adam's two histories can use different decay rates.

</details>

## What an interviewer expects you to say

Reconstruct buffers, step count, correction factors, and the denominator before discussing optimizer preferences.
Two FP32 moment arrays require $8N$ bytes: $560$ decimal GB for $70$ billion parameters.
That excludes weights, gradients, master copies, activations, and communication buffers.

<!-- learning:core-end -->
<div data-learning-extension>

## AdamW: the fix that matters

AdamW applies decay directly to parameters, separately from adaptive gradient normalization:

```
theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps) - lr * weight_decay * theta_{t-1}
```

<!-- visual:adamw-decoupled-shrinkage -->
<figure class="learning-figure" aria-labelledby="adamw-shrinkage-title">
	<p class="visual-kicker">Worked comparison</p>
	<p class="visual-title" id="adamw-shrinkage-title">Do equal parameters shrink equally when their gradient histories differ?</p>
	<p>Hold the adaptive preconditioner fixed to isolate regularization. Both parameters start at θ = 1, with learning rate α = 0.01 and regularization rate λ = 0.1.</p>
	<div class="visual-grid--two" role="group" aria-label="Comparison of regularization contributions for two equal parameters with adaptive history scales of one and ten">
		<section class="visual-panel" aria-labelledby="adamw-parameter-a-title">
			<h4 id="adamw-parameter-a-title">Parameter A · history scale s = 1</h4>
			<p>The adaptive denominator does not reduce the coupled penalty.</p>
			<table class="cm-grid" aria-label="Parameter A regularization contributions">
				<thead><tr><th scope="col">Method</th><th scope="col">Shrinkage this step</th></tr></thead>
				<tbody>
					<tr><th scope="row">Adam + L2</th><td><strong>0.001</strong> αλθ / s</td></tr>
					<tr><th scope="row">AdamW</th><td><strong>0.001</strong> αλθ</td></tr>
				</tbody>
			</table>
		</section>
		<section class="visual-panel" aria-labelledby="adamw-parameter-b-title">
			<h4 id="adamw-parameter-b-title">Parameter B · history scale s = 10</h4>
			<p>The larger adaptive denominator makes the coupled penalty ten times smaller.</p>
			<table class="cm-grid" aria-label="Parameter B regularization contributions">
				<thead><tr><th scope="col">Method</th><th scope="col">Shrinkage this step</th></tr></thead>
				<tbody>
					<tr><th scope="row">Adam + L2</th><td><strong>0.0001</strong> αλθ / s</td></tr>
					<tr><th scope="row">AdamW</th><td><strong>0.001</strong> αλθ</td></tr>
				</tbody>
			</table>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> hold the adaptive denominator fixed for this illustration. Equal weights then receive different scaled penalty contributions, while AdamW shrinks both by 0.001. This is not a full Adam recurrence: coupled penalties also enter the moment buffers.</figcaption>
</figure>

See [the decay derivation](/concepts/weight-decay-vs-l2/) and [Loshchilov and Hutter](https://arxiv.org/abs/1711.05101).
Check API decay mode and excluded parameter groups rather than inferring behavior from the optimizer's name alone.

## When Adam goes wrong

Warmup changes the learning-rate schedule; bias correction changes moment initialization factors.
Neither guarantees stability, and no fixed warmup percentage suits every workload.
Inspect actual update norms, nonfinite values, loss scale, and sampling changes.
Sparse-gradient support and inactive-coordinate updates depend on the optimizer implementation.

## Modern optimizer choices

| Choice | Distinct computation or state tradeoff |
| --- | --- |
| LAMB | Adds a layerwise trust ratio to an adaptive update. |
| AdaFactor | Factors matrix second-moment statistics; optional first moments add separate state. |
| Lion | Uses signed updates and directional momentum without Adam's squared-gradient buffer. |
| Sophia | Uses estimated diagonal curvature in its preconditioner. |
| Shampoo | Uses matrix statistics and Kronecker-structured preconditioning, not an exact full Hessian. |
| Muon | Approximately orthogonalizes matrix momentum updates; other parameter groups need their specified treatment. |

Memory, communication, numerical precision, and matched empirical quality determine suitability, not a universal production ranking.

## Common confusions

Adam still needs a learning rate.
RMSProp-like squared-gradient normalization does not by itself specify Adam's first moment and bias correction.
Save moments, update count, parameter groups, and schedule state when resuming.

## Why interviewers ask

The useful test is whether you can reconstruct an update and separate assumptions from an empirical optimizer preference.

---

*Related interview: [How do you choose a learning rate?](/questions/how-to-choose-learning-rate/), [How would you debug a model that's not learning?](/questions/debug-model-not-learning/).*

</div>
