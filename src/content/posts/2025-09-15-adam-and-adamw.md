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
<figure class="learning-figure editorial-illustration" aria-labelledby="adamw-decoupled-shrinkage-edition-title"><span class="visual-fragment-anchor" id="adamw-shrinkage-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="adamw-parameter-a-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="adamw-parameter-b-title" aria-hidden="true"></span><p class="visual-title" id="adamw-decoupled-shrinkage-edition-title">Adam carries direction, rescales coordinates, and keeps decay separate</p><div class="visual-argument" data-visual-argument="adamw-decoupled-shrinkage" data-argument-kind="bars" data-argument-composition="adam-memory-and-scale"><svg viewBox="0 0 360 2120" role="img" aria-label="Adam carries direction, rescales coordinates, and keeps decay separate. Both buffers blend history: direction and raw squared gradients. Startup correction compensates for zero initialization; root-size then rescales each direction. A vanished gradient can leave an update because saved direction remains. Removing directional memory stops it. Epsilon limits cancellation for tiny gradients. Direct AdamW decay bypasses this route."><title>Adam carries direction, rescales coordinates, and keeps decay separate</title><desc>Both buffers blend history: direction and raw squared gradients. Startup correction compensates for zero initialization; root-size then rescales each direction. A vanished gradient can leave an update because saved direction remains. Removing directional memory stops it. Epsilon limits cancellation for tiny gradients. Direct AdamW decay bypasses this route.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Store history before rescaling the</tspan><tspan x="20" dy="18">step</tspan></text><text x="20" y="72" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Zero start; both buffers keep half</tspan><tspan x="20" dy="18">their past</tspan></text><text x="20" y="109" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Learning rate 0.1; epsilon zero</tspan></text><text x="20" y="155" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">First update: different gradient</tspan><tspan x="20" dy="18">sizes</tspan></text><g data-picture-adam-coordinate="0" data-picture-adam-update="1" data-picture-adam-gradient="2" data-picture-adam-direction="1" data-picture-adam-squared="2" data-picture-adam-step="0.1"><circle cx="160" cy="202" r="17" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="160" y="208" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="160" dy="0">2</tspan></text><text x="20" y="250" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">stored</tspan><tspan x="20" dy="18">direction</tspan></text><text x="160" y="250" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="160" dy="0">1</tspan></text><text x="20" y="295" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">squared</tspan><tspan x="20" dy="18">history</tspan></text><text x="160" y="295" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="160" dy="0">2</tspan></text><text x="20" y="377" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">corrected</tspan><tspan x="20" dy="18">direction</tspan></text><text x="160" y="377" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="160" dy="0">2</tspan></text><text x="20" y="417" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">root-size</tspan></text><text x="160" y="417" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="160" dy="0">2</tspan></text><rect x="132" y="440" width="56" height="100" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></rect><text x="160" y="572" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="160" dy="0">0.1</tspan></text></g><g data-picture-adam-coordinate="1" data-picture-adam-update="1" data-picture-adam-gradient="4" data-picture-adam-direction="2" data-picture-adam-squared="8" data-picture-adam-step="0.1"><circle cx="280" cy="202" r="17" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="280" y="208" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="280" dy="0">4</tspan></text><text x="280" y="250" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="280" dy="0">2</tspan></text><text x="280" y="295" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="280" dy="0">8</tspan></text><text x="280" y="377" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="280" dy="0">4</tspan></text><text x="280" y="417" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="280" dy="0">4</tspan></text><rect x="252" y="440" width="56" height="100" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><text x="280" y="572" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="280" dy="0">0.1</tspan></text></g><text x="20" y="205" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">gradient</tspan></text><text x="20" y="339" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Correct the zero-start bias</tspan></text><text x="20" y="516" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">step size</tspan></text><path d="M132 540L308 540" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="20" y="625" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Next update: second gradient vanishes</tspan></text><g data-picture-adam-coordinate="0" data-picture-adam-update="2" data-picture-adam-gradient="2" data-picture-adam-direction="1.5" data-picture-adam-squared="3" data-picture-adam-step="0.1"><circle cx="160" cy="672" r="17" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="160" y="678" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="160" dy="0">2</tspan></text><text x="20" y="720" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">stored</tspan><tspan x="20" dy="18">direction</tspan></text><text x="160" y="720" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="160" dy="0">1.5</tspan></text><text x="20" y="765" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">squared</tspan><tspan x="20" dy="18">history</tspan></text><text x="160" y="765" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="160" dy="0">3</tspan></text><text x="20" y="847" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">corrected</tspan><tspan x="20" dy="18">direction</tspan></text><text x="160" y="847" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="160" dy="0">2</tspan></text><text x="20" y="887" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">root-size</tspan></text><text x="160" y="887" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="160" dy="0">2</tspan></text><rect x="132" y="910" width="56" height="100" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></rect><text x="160" y="1042" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="160" dy="0">0.1</tspan></text></g><g data-picture-adam-coordinate="1" data-picture-adam-update="2" data-picture-adam-gradient="0" data-picture-adam-direction="1" data-picture-adam-squared="4" data-picture-adam-step="0.05773502691896258"><circle cx="280" cy="672" r="17" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="280" y="678" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="280" dy="0">0</tspan></text><text x="280" y="720" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="280" dy="0">1</tspan></text><text x="280" y="765" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="280" dy="0">4</tspan></text><text x="280" y="847" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="280" dy="0">1.333</tspan></text><text x="280" y="887" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="280" dy="0">2.309</tspan></text><rect x="252" y="952.2649730810374" width="56" height="57.73502691896258" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><text x="280" y="1042" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="280" dy="0">0.05774</tspan></text></g><text x="20" y="675" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">gradient</tspan></text><text x="20" y="809" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Correct the zero-start bias</tspan></text><text x="20" y="986" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">step size</tspan></text><path d="M132 1010L308 1010" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M305 246H340V709H306" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2"></path><path d="M340 709L306 709M314 705L306 709L314 713" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="20" y="601" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Half the saved direction remains</tspan></text><text x="20" y="1095" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Remove direction memory: this step</tspan><tspan x="20" dy="18">vanishes</tspan></text><text x="20" y="1172" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Same tiny input: epsilon prevents</tspan><tspan x="20" dy="18">scale cancellation</tspan></text><text x="20" y="1229" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Input one-billionth; epsilon</tspan><tspan x="20" dy="18">ten-billionths</tspan></text><rect data-picture-adam-tiny-epsilon="0" data-picture-adam-tiny-step="0.1" x="72" y="1275" width="56" height="125" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></rect><text x="100" y="1430" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="100" dy="0">no epsilon</tspan></text><text x="100" y="1470" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="100" dy="0">0.1</tspan></text><rect data-picture-adam-tiny-epsilon="1" data-picture-adam-tiny-step="0.009090909090909092" x="232" y="1388.6363636363637" width="56" height="11.363636363636365" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><text x="260" y="1430" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="260" dy="0">with epsilon</tspan></text><text x="260" y="1470" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="260" dy="0">0.009091</tspan></text><path d="M70 1400L290 1400" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g transform="translate(0 1540)"><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Shrinkage: coupled or direct</tspan></text><text x="20" y="54" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Same weights; frozen preconditioners</tspan></text><text x="20" y="76" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Frozen gradient scale: 1</tspan></text><circle cx="48" cy="131" r="21" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="48" y="136" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="48" dy="0">w</tspan></text><path d="M128 108h70l18 23-18 23h-70l-18-23Z" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><text x="163" y="136" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="163" dy="0">same</tspan></text><path d="M70 131L108 131M100 135L108 131L100 127" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><path d="M216 131L260 131M252 135L260 131L252 127" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="340" y="136" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--viz-input-stroke)"><tspan x="340" dy="0">0.001</tspan></text><path d="M48 154V201H291" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2.5"></path><text x="340" y="207" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--viz-output-stroke)"><tspan x="340" dy="0">0.001</tspan></text><text x="20" y="265" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Frozen gradient scale: 10</tspan></text><circle cx="48" cy="320" r="21" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="48" y="325" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="48" dy="0">w</tspan></text><path d="M128 297h70l18 23-18 23h-70l-18-23Z" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><text x="163" y="325" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="163" dy="0">tenth</tspan></text><path d="M70 320L108 320M100 324L108 320L100 316" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><path d="M216 320L260 320M252 324L260 320L252 316" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="340" y="325" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--viz-input-stroke)"><tspan x="340" dy="0">0.0001</tspan></text><path d="M48 343V390H291" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2.5"></path><text x="340" y="396" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--viz-output-stroke)"><tspan x="340" dy="0">0.001</tspan></text><text x="20" y="491" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">Blue: coupled through the gradient</tspan><tspan x="20" dy="18">scale</tspan></text><text x="20" y="536" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Green: direct decay bypasses that</tspan><tspan x="20" dy="18">scale</tspan></text></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 529" role="img" aria-labelledby="adamw-decoupled-shrinkage-edition-svg-title adamw-decoupled-shrinkage-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="adamw-decoupled-shrinkage-edition-svg-title">Equal weights, unequal coupled penalties</title><desc id="adamw-decoupled-shrinkage-edition-svg-desc">With theta one, alpha .01, lambda .1, and fixed adaptive scales one and ten, coupled L2 contributes shrinkage .001 and .0001. AdamW contributes equal .001 shrinkage. Leftward arrows share the same parameter-distance scale; this isolates the penalty and is not a full Adam recurrence.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fixed preconditioner; not the full Adam</tspan><tspan x="180" dy="19">recurrence</tspan></text><text x="180" y="64" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Same start: theta = 1</tspan></text><text x="180" y="89" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">alpha = 0.01; lambda = 0.1</tspan></text><text x="18" y="130" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Adam + coupled L2</tspan></text><text x="18" y="162" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">A: s = 1</tspan></text><path d="M70 174L320 174" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M320 164L320 184" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-shrink="0.001" data-start="320" data-end="70"><path d="M320 174L70 174" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M76.1430779332326,170.64402122977057L70,174L76.1430779332326,177.35597877022943" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><text x="195" y="200" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="195" dy="0">shrink 0.001; end 0.999</tspan></text><text x="18" y="226" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">B: s = 10</tspan></text><path d="M70 238L320 238" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M320 228L320 248" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-shrink="0.0001" data-start="320" data-end="295"><path d="M320 238L295 238" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M301.14307793323263,234.64402122977057L295,238L301.14307793323263,241.35597877022943" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><text x="195" y="264" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="195" dy="0">shrink 0.0001; end 0.9999</tspan></text><text x="18" y="310" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">AdamW: decoupled</tspan></text><text x="18" y="342" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">A: s = 1</tspan></text><path d="M70 354L320 354" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M320 344L320 364" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-shrink="0.001" data-start="320" data-end="70"><path d="M320 354L70 354" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M76.1430779332326,350.6440212297706L70,354L76.1430779332326,357.3559787702294" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path></g><text x="195" y="380" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="195" dy="0">shrink 0.001; end 0.999</tspan></text><text x="18" y="406" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">B: s = 10</tspan></text><path d="M70 418L320 418" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M320 408L320 428" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-shrink="0.001" data-start="320" data-end="70"><path d="M320 418L70 418" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M76.1430779332326,414.6440212297706L70,418L76.1430779332326,421.3559787702294" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path></g><text x="195" y="444" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="195" dy="0">shrink 0.001; end 0.999</tspan></text><text x="180" y="493" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Every arrow uses the same parameter-distance</tspan><tspan x="180" dy="19">scale.</tspan></text></svg><div class="visual-derivation-caption"><strong>Read it this way:</strong> Compare arrows on one zoomed parameter scale. Coupled L2 is divided by the fixed adaptive scale. AdamW is not. Coupled penalties also affect moment buffers.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="adamw-decoupled-shrinkage"><strong>Check:</strong> Which method gives equal weights equal explicit decay despite different history scales?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="adamw-decoupled-shrinkage">Both buffers blend history: direction and raw squared gradients. Startup correction compensates for zero initialization; root-size then rescales each direction. A vanished gradient can leave an update because saved direction remains. Removing directional memory stops it. Epsilon limits cancellation for tiny gradients. Direct AdamW decay bypasses this route. <strong>Does zero current gradient clear the second-coordinate direction buffer?</strong></span></figcaption></figure>

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
