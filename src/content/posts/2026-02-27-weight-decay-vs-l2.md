---
title: "L2 and weight decay: follow the shrinkage path"
description: "Derive the vanilla-SGD equivalence, isolate adaptive scaling, and work changes to parameter groups and cumulative decay exposure."
date: "2026-02-27"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

L2 adds a penalty to the objective.
Decoupled weight decay shrinks parameters outside the optimizer's gradient history.
Their updates coincide for vanilla SGD, but that algebra does not automatically extend to momentum or Adam.

**Task:** start two parameters at $[1,1]$, with data gradient $[0.2,0.2]$, learning rate $\eta=0.01$, and $\lambda=0.1$.
Freeze adaptive denominators at $[1,10]$ to isolate one step.
Does routing $\lambda\theta$ through those denominators give the same result as direct decay?

## The two formulations

### L2 (penalty added to loss)

$$
L_\text{total}(\theta) = L_\text{data}(\theta) + \tfrac{\lambda}{2} \|\theta\|^2
$$

Writing $g=\nabla L_\text{data}(\theta)$, the penalized gradient is $g+\lambda\theta$.
Vanilla SGD gives the exact expansion

$$
{}\theta-\eta(g+\lambda\theta)=(1-\eta\lambda)\theta-\eta g.
$$

### Weight decay (multiplicative shrink)

Direct decay uses the expression on the right.
Equality requires the same learning rate, penalty convention, parameter groups, and old-state evaluation, with no momentum or adaptive scaling.

## Why they diverge under Adam

Coupled L2 feeds $g+\lambda\theta$ into both [Adam moment buffers](/concepts/adam-and-adamw/).
Decoupled AdamW feeds only $g$ into them.
The history itself changes, not merely a final division.

For the frozen-denominator fixture, isolate the paths without updating moments:

| Quantity | Coordinate 1 | Coordinate 2 |
| --- | --- | --- |
| Coupled data-plus-penalty step $\eta(g+\lambda\theta)/s$ | $0.003$ | $0.0003$ |
| Coupled next parameter | $0.997$ | $0.9997$ |
| Direct decay $\eta\lambda\theta$ | $0.001$ | $0.001$ |
| Decoupled data step $\eta g/s$ | $0.002$ | $0.0002$ |
| Decoupled next parameter | $0.997$ | $0.9988$ |

The equal weights receive equal fractional direct decay despite unequal adaptive denominators.
This illustration is not a complete coupled-Adam trajectory; real coupled penalties also change both moments.

<!-- visual:weight-decay-adaptive-paths -->
<figure class="learning-figure" aria-labelledby="weight-decay-paths-title" aria-describedby="weight-decay-paths-description"><p class="visual-kicker">Learning objective</p><p class="visual-title" id="weight-decay-paths-title">Adam rescales coupled L2 through its moments; AdamW applies decay directly</p><div class="visual-argument" data-visual-argument="weight-decay-adaptive-paths" data-argument-kind="bars" data-argument-composition="endpoint"><svg viewBox="0 0 360 394" role="img" aria-label="Adaptive rescaling changes coupled L2, not the direct decay fraction. For equal weights, coupled L2 passes through different fixed preconditioners. The illustrative decoupled decay remains 0.001 for either history."><title>Adaptive rescaling changes coupled L2, not the direct decay fraction</title><desc>For equal weights, coupled L2 passes through different fixed preconditioners. The illustrative decoupled decay remains 0.001 for either history.</desc><text x="18" y="23" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Coupled L2; preconditioner 1</tspan></text><path d="M28 64L332 64" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="0.001" data-proof-min="0" data-proof-max="0.001" data-proof-zero="28" x="28" y="46" width="304" height="35" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></rect><circle cx="332" cy="63.5" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M332 69L332 86" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="233" y="106" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="233" dy="0">0.001 weight reduction</tspan></text><text x="18" y="140" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Coupled L2; preconditioner 10</tspan></text><path d="M28 181L332 181" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="0.0001" data-proof-min="0" data-proof-max="0.001" data-proof-zero="28" x="28" y="163" width="30.400000000000006" height="35" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></rect><circle cx="58.400000000000006" cy="180.5" r="4" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><path d="M58.400000000000006 186L58.400000000000006 203" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="131.5" y="223" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="131.5" dy="0">0.0001 weight reduction</tspan></text><text x="18" y="257" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Decoupled decay; either history</tspan></text><path d="M28 298L332 298" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="0.001" data-proof-min="0" data-proof-max="0.001" data-proof-zero="28" x="28" y="280" width="304" height="35" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><circle cx="332" cy="297.5" r="4" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><path d="M332 303L332 320" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><text x="233" y="340" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="233" dy="0">0.001 weight reduction</tspan></text><text x="28" y="374" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="28" dy="0">0</tspan></text><text x="332" y="374" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="332" dy="0">0.001</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><p id="weight-decay-paths-description">Follow λθ: through Adam's denominator in L2, around it in AdamW.</p><svg class="weight-decay-path-visual" viewBox="0 0 900 420" role="img" aria-labelledby="weight-decay-svg-title weight-decay-svg-description" width="900" height="420"><title id="weight-decay-svg-title">Coupled L2 and decoupled AdamW route the regularization term through different paths</title><desc id="weight-decay-svg-description">The left panel routes lambda theta through gradient joining, moment tracking, and two adaptive denominators, producing different shrinkage for coordinates with small and large denominators. The right panel routes lambda theta directly around Adam, producing the same fractional decay for both coordinates.</desc><defs><marker id="wd-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="var(--viz-focus-stroke)" /></marker><marker id="wd-data-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="var(--viz-input-stroke)" /></marker></defs><text x="44" y="42" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="20" font-weight="700">Adam + L2: coupled</text><text x="44" y="66" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="13">lambda theta becomes part of the adapted gradient</text><rect x="46" y="105" width="102" height="52" rx="8" fill="var(--viz-focus-bg)" stroke="var(--viz-focus-stroke)" stroke-width="2" /><text x="97" y="136" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="16" font-weight="700">λθ</text><rect x="46" y="205" width="102" height="52" rx="8" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)" stroke-width="2" /><text x="97" y="236" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="15" font-weight="700">data grad</text><line x1="148" y1="131" x2="202" y2="160" stroke="var(--viz-focus-stroke)" stroke-width="3" marker-end="url(#wd-arrow)" /><line x1="148" y1="231" x2="202" y2="184" stroke="var(--viz-input-stroke)" stroke-width="3" marker-end="url(#wd-data-arrow)" /><rect x="204" y="136" width="118" height="72" rx="8" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><text x="263" y="166" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="14" font-weight="700">join</text><text x="263" y="187" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="12">g + λθ</text><line x1="322" y1="172" x2="376" y2="172" stroke="var(--viz-focus-stroke)" stroke-width="3" marker-end="url(#wd-arrow)" /><rect x="378" y="136" width="120" height="72" rx="8" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><text x="438" y="166" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="14" font-weight="700">Adam moments</text><text x="438" y="187" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="12">m and v include λθ</text><line x1="438" y1="208" x2="438" y2="256" stroke="var(--viz-focus-stroke)" stroke-width="3" marker-end="url(#wd-arrow)" /><rect x="315" y="258" width="92" height="54" rx="8" fill="var(--viz-warning-bg)" stroke="var(--viz-warning-stroke)" /><rect x="426" y="258" width="92" height="54" rx="8" fill="var(--viz-warning-bg)" stroke="var(--viz-warning-stroke)" /><text x="361" y="280" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="12" font-weight="700">coord 1</text><text x="361" y="300" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="11">small sqrt(v)</text><text x="472" y="280" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="12" font-weight="700">coord 2</text><text x="472" y="300" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="11">large sqrt(v)</text><text x="318" y="348" fill="var(--viz-warning-stroke)" font-family="Lato, sans-serif" font-size="13" font-weight="700">same λθ, different effective shrinkage</text><line x1="545" y1="70" x2="545" y2="365" stroke="var(--c-rule)" stroke-width="1.5" /><text x="590" y="42" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="20" font-weight="700">AdamW: decoupled</text><text x="590" y="66" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="13">data gradient adapts; decay bypasses moments</text><rect x="592" y="105" width="102" height="52" rx="8" fill="var(--viz-focus-bg)" stroke="var(--viz-focus-stroke)" stroke-width="2" /><text x="643" y="136" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="16" font-weight="700">λθ</text><rect x="592" y="205" width="102" height="52" rx="8" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)" stroke-width="2" /><text x="643" y="236" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="15" font-weight="700">data grad</text><line x1="694" y1="231" x2="748" y2="231" stroke="var(--viz-input-stroke)" stroke-width="3" marker-end="url(#wd-data-arrow)" /><rect x="750" y="195" width="116" height="72" rx="8" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><text x="808" y="225" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="14" font-weight="700">Adam step</text><text x="808" y="246" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="12">m, v see data only</text><path d="M694 131 C760 92 848 110 865 178" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="3" marker-end="url(#wd-arrow)" /><line x1="808" y1="267" x2="808" y2="305" stroke="var(--viz-input-stroke)" stroke-width="3" marker-end="url(#wd-data-arrow)" /><line x1="865" y1="178" x2="865" y2="305" stroke="var(--viz-focus-stroke)" stroke-width="3" marker-end="url(#wd-arrow)" /><rect x="735" y="308" width="130" height="54" rx="8" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)" stroke-width="2" /><text x="800" y="331" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">parameter update</text><text x="800" y="350" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="11">minus ηλθ directly</text><text x="590" y="348" fill="var(--viz-output-stroke)" font-family="Lato, sans-serif" font-size="13" font-weight="700">same fractional decay for both coordinates</text></svg><svg class="weight-decay-path-mobile" viewBox="0 0 360 560" role="img" aria-labelledby="weight-decay-mobile-title weight-decay-mobile-description" width="360" height="560"><title id="weight-decay-mobile-title">Mobile view of coupled L2 and decoupled AdamW paths</title><desc id="weight-decay-mobile-description">Two stacked panels show lambda theta merging with the Adam path in L2 and bypassing it in AdamW.</desc><defs><marker id="wd-mobile-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="var(--viz-focus-stroke)" /></marker><marker id="wd-mobile-data-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="var(--viz-input-stroke)" /></marker></defs><text x="18" y="30" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="17" font-weight="700">Adam + L2: λθ joins Adam</text><rect x="22" y="60" width="88" height="44" rx="8" fill="var(--viz-focus-bg)" stroke="var(--viz-focus-stroke)" stroke-width="2" /><text x="66" y="87" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="15" font-weight="700">λθ</text><rect x="22" y="132" width="88" height="44" rx="8" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)" stroke-width="2" /><text x="66" y="159" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">data grad</text><line x1="110" y1="82" x2="160" y2="111" stroke="var(--viz-focus-stroke)" stroke-width="3" marker-end="url(#wd-mobile-arrow)" /><line x1="110" y1="154" x2="160" y2="129" stroke="var(--viz-input-stroke)" stroke-width="3" marker-end="url(#wd-mobile-data-arrow)" /><rect x="162" y="96" width="78" height="50" rx="8" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><text x="201" y="118" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">join</text><text x="201" y="136" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="11">g + λθ</text><line x1="240" y1="121" x2="286" y2="121" stroke="var(--viz-focus-stroke)" stroke-width="3" marker-end="url(#wd-mobile-arrow)" /><rect x="288" y="92" width="56" height="58" rx="8" fill="var(--viz-warning-bg)" stroke="var(--viz-warning-stroke)" /><text x="316" y="116" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="12" font-weight="700">Adam</text><text x="316" y="134" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="10">divides</text><text x="22" y="214" fill="var(--viz-warning-stroke)" font-family="Lato, sans-serif" font-size="13" font-weight="700">same λθ, but each coordinate history changes shrinkage</text><line x1="18" y1="260" x2="342" y2="260" stroke="var(--c-rule)" stroke-width="1.5" /><text x="18" y="306" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="17" font-weight="700">AdamW: λθ bypasses Adam</text><rect x="22" y="336" width="88" height="44" rx="8" fill="var(--viz-focus-bg)" stroke="var(--viz-focus-stroke)" stroke-width="2" /><text x="66" y="363" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="15" font-weight="700">λθ</text><rect x="22" y="418" width="88" height="44" rx="8" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)" stroke-width="2" /><text x="66" y="445" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">data grad</text><line x1="110" y1="440" x2="166" y2="440" stroke="var(--viz-input-stroke)" stroke-width="3" marker-end="url(#wd-mobile-data-arrow)" /><rect x="168" y="410" width="84" height="60" rx="8" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><text x="210" y="434" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">Adam</text><text x="210" y="454" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="11">data only</text><path d="M110 358 C190 318 284 330 306 404" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="3" marker-end="url(#wd-mobile-arrow)" /><line x1="252" y1="440" x2="292" y2="440" stroke="var(--viz-input-stroke)" stroke-width="3" marker-end="url(#wd-mobile-data-arrow)" /><rect x="294" y="412" width="50" height="56" rx="8" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)" stroke-width="2" /><text x="319" y="435" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="12" font-weight="700">θ</text><text x="319" y="453" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="10">update</text><text x="22" y="520" fill="var(--viz-output-stroke)" font-family="Lato, sans-serif" font-size="13" font-weight="700">one direct fractional shrinkage, independent of v</text></svg>
<div class="visual-derivation-caption">follow λθ. In Adam with L2, it joins the gradient before the adaptive machinery and changes the moment histories. AdamW's upper bypass instead applies direct fractional decay. Numerical denominator comparisons freeze the preconditioner; they are not full coupled-Adam trajectories. Original schematic checked against <a href="https://arxiv.org/abs/1711.05101">Loshchilov and Hutter (2019)</a> and the <a href="https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html">PyTorch AdamW algorithm</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="weight-decay-adaptive-paths"><strong>Check:</strong> Why does Adam scale equal L2 penalties differently, while AdamW applies equal direct decay fractions?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="weight-decay-adaptive-paths">For equal weights, coupled L2 passes through different fixed preconditioners. The illustrative decoupled decay remains 0.001 for either history.</span></figcaption>
</figure>

For AdamW, compute corrected moments from the data gradients and evaluate both contributions against the old parameter:

$$
{}\theta_{t+1}=(1-\eta_t\lambda)\theta_t
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

Epsilon is outside the square root in this convention.
Shrinking an already updated parameter instead introduces a cross-term; it is not the displayed update.

### Remove adaptive scaling

**New task:** replace denominators $[1,10]$ with $[1,1]$, without momentum.
**Picture and change:** both paths reduce to the vanilla-SGD expansion.
**Check:** both produce $[0.997,0.997]$.
**Use when:** the actual optimizer satisfies those assumptions, not merely because its name contains SGD.

### Exempt one parameter

**New task:** exclude coordinate two from direct decay while retaining its data gradient.
**Picture and change:** set only its shrink contribution to zero.
**Check:** the decoupled result becomes $[0.997,0.9998]$, instead of $[0.997,0.9988]$.
**Use when:** the architecture explicitly exempts that parameter role; tied tensors must not receive duplicate updates.

### Failure and transfer

With momentum, even zero current data gradient can leave a nonzero history-driven update.
The vanilla proof cannot identify that movement with direct shrinkage.
With zero adaptive updates, cumulative decay is $\prod_t(1-\eta_t\lambda)$, so the schedule matters.

**Reconstruct:** what happens when $\eta$ halves from $0.01$ at $\theta=1$, $\lambda=0.1$?

<details><summary>Answer</summary>

One direct-decay step changes from $0.999$ to $0.9995$.
The decay exposure halves for that step; this says nothing about stored-gradient movement.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

## Empirical impact

[Loshchilov and Hutter](https://arxiv.org/abs/1711.05101) distinguish decoupled decay from adaptive L2 and report experiments.
Their results are not a guarantee of improved generalization on every architecture or training regime.
Compare tuned coefficients and schedules under a matched evaluation contract.

## What to skip

Review parameter roles rather than applying a universal no-decay list.
Biases and normalization scales are often treated separately; embedding tables are usually matrices, not one-dimensional parameters.
An embedding can also be tied to an output projection, so membership must follow tensor identity.
Assign every trainable parameter exactly once and retain the architecture's chosen exclusions.

## Common pitfalls

- Inspect the installed API's decay option; some Adam implementations expose a decoupled mode.
- Loss reduction conventions change the relative scale of an objective penalty.
- A copied coefficient does not preserve exposure when step count or learning-rate schedule changes.
- The term is ordinary fractional shrinkage when $0\le\eta_t\lambda<1$; larger products can erase or flip a parameter.

## Related

- [Adam and AdamW](/concepts/adam-and-adamw/). For the optimizer-side derivation.
- [Regularization](/concepts/regularization/). Broader survey of regularization techniques.

</div>
