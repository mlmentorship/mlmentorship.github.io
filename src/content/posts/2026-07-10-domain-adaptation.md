---
title: "Domain adaptation"
description: "Transfer a model across related but shifted data distributions without assuming unlabeled target data makes the problem identifiable."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Domain adaptation transfers a predictor between source and target populations under explicit assumptions about what changed.
Specify the shared label space, sampling unit, available target supervision, time period, and deployment slices.
Unlabeled target inputs can reveal an input mixture without identifying a changed labeling rule.

Our teaching fixture has two input bins, fixed predictor errors $(0.1,0.4)$, and different population mixtures.
Compute target risk before choosing a correction.

## Shift types

- **Covariate shift:** $P(X)$ changes while $P(Y\mid X)$ is stable.
- **Label shift:** $P(Y)$ changes while $P(X\mid Y)$ is stable.
- **Concept shift:** $P(Y\mid X)$ changes; unlabeled target data alone is generally not enough.

Multiple factors can move together; a drift test on inputs does not establish an unchanged conditional distribution.

<!-- visual:domain-adaptation-shift-assumptions -->
<figure class="learning-figure editorial-illustration" aria-labelledby="domain-adaptation-shift-assumptions-edition-title"><span class="visual-fragment-anchor" id="domain-adaptation-visual-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="domain-adaptation-svg-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="domain-adaptation-svg-desc" aria-hidden="true"></span><span class="visual-fragment-anchor" id="domain-adaptation-arrow" aria-hidden="true"></span><p class="visual-title" id="domain-adaptation-shift-assumptions-edition-title">Which factor moved? Inspect the same joint-probability cells</p><svg viewBox="0 0 360 806" role="img" aria-labelledby="domain-adaptation-shift-assumptions-edition-svg-title domain-adaptation-shift-assumptions-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="domain-adaptation-shift-assumptions-edition-svg-title">Which factor moved? Inspect the same joint-probability cells</title><desc id="domain-adaptation-shift-assumptions-edition-svg-desc">Toy joint tables change only the declared factor. Covariate shift moves P(X=1) from .2 to .8 with fixed label conditionals. Label shift moves P(Y=1) from .2 to .8 with fixed input conditionals. Two concept-shift worlds then share identical balanced inputs but opposite deterministic label rules.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Toy joint probabilities P(X, Y); each panel sums</tspan><tspan x="180" dy="19">to 1</tspan></text><text x="180" y="66" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">COVARIATE: only P(X) changes</tspan></text><text x="99" y="96" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="99" dy="0">SOURCE</tspan></text><text x="75" y="123" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="75" dy="0">Y=0</tspan></text><text x="129" y="123" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="129" dy="0">Y=1</tspan></text><rect x="48" y="134" width="54" height="37" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="75" y="158" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="75" dy="0">0.72</tspan></text><rect x="102" y="134" width="54" height="37" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="129" y="158" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="129" dy="0">0.08</tspan></text><rect x="48" y="171" width="54" height="37" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="75" y="195" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="75" dy="0">0.02</tspan></text><rect x="102" y="171" width="54" height="37" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="129" y="195" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="129" dy="0">0.18</tspan></text><text x="273" y="96" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="273" dy="0">TARGET</tspan></text><text x="249" y="123" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="249" dy="0">Y=0</tspan></text><text x="303" y="123" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="303" dy="0">Y=1</tspan></text><rect x="222" y="134" width="54" height="37" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="249" y="158" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="249" dy="0">0.18</tspan></text><rect x="276" y="134" width="54" height="37" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="303" y="158" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="303" dy="0">0.02</tspan></text><rect x="222" y="171" width="54" height="37" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="249" y="195" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="249" dy="0">0.08</tspan></text><rect x="276" y="171" width="54" height="37" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="303" y="195" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="303" dy="0">0.72</tspan></text><text x="15" y="171" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="15" dy="0">X=0</tspan></text><text x="15" y="208" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="15" dy="0">X=1</tspan></text><text x="180" y="243" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">P(Y=1 | X=0)=.1; P(Y=1 | X=1)=.9</tspan></text><text x="180" y="273" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">P(X=1): .2 -&gt; .8; support overlaps</tspan></text><text x="180" y="307" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">LABEL: only P(Y) changes</tspan></text><text x="99" y="337" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="99" dy="0">SOURCE</tspan></text><text x="75" y="364" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="75" dy="0">Y=0</tspan></text><text x="129" y="364" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="129" dy="0">Y=1</tspan></text><rect x="48" y="375" width="54" height="37" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="75" y="399" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="75" dy="0">0.60</tspan></text><rect x="102" y="375" width="54" height="37" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="129" y="399" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="129" dy="0">0.05</tspan></text><rect x="48" y="412" width="54" height="37" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="75" y="436" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="75" dy="0">0.20</tspan></text><rect x="102" y="412" width="54" height="37" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="129" y="436" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="129" dy="0">0.15</tspan></text><text x="273" y="337" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="273" dy="0">TARGET</tspan></text><text x="249" y="364" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="249" dy="0">Y=0</tspan></text><text x="303" y="364" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="303" dy="0">Y=1</tspan></text><rect x="222" y="375" width="54" height="37" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="249" y="399" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="249" dy="0">0.15</tspan></text><rect x="276" y="375" width="54" height="37" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="303" y="399" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="303" dy="0">0.20</tspan></text><rect x="222" y="412" width="54" height="37" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="249" y="436" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="249" dy="0">0.05</tspan></text><rect x="276" y="412" width="54" height="37" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="303" y="436" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="303" dy="0">0.60</tspan></text><text x="15" y="412" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="15" dy="0">X=0</tspan></text><text x="15" y="449" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="15" dy="0">X=1</tspan></text><text x="180" y="484" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">P(X=1 | Y=0)=.25; P(X=1 | Y=1)=.75</tspan></text><text x="180" y="514" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">P(Y=1): .2 -&gt; .8; identifiable statistic needed</tspan></text><text x="180" y="554" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">CONCEPT: same target inputs, incompatible</tspan><tspan x="180" dy="19">labels</tspan></text><g data-editorial-node><rect x="25" y="586" width="310" height="52" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="617" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Both worlds: P(X=0)=P(X=1)=0.5</tspan></text></g><g data-editorial-node><rect x="15" y="662" width="150" height="70" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="90" y="692.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">World A</tspan><tspan x="90" dy="19">Y = X</tspan></text></g><g data-editorial-node><rect x="195" y="662" width="150" height="70" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="692.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">World B</tspan><tspan x="270" dy="19">Y = 1 - X</tspan></text></g><text x="180" y="770" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Unlabeled X cannot distinguish these</tspan><tspan x="180" dy="19">worlds.</tspan></text></svg><figcaption><strong>Read it this way:</strong> Rows index X and columns index Y. Check the stated conditional against the cells. Identical target input marginals cannot identify which opposite label rule holds. <span class="visual-reader-check" data-reader-check="domain-adaptation-shift-assumptions"><strong>Check:</strong> How can two targets have identical observed X distributions but require different predictions?</span></figcaption></figure>

### Work the target risk

| Input bin | Source mass | Target mass | Conditional error | Target/source weight |
| --- | --- | --- | --- | --- |
| 0 | 0.8 | 0.2 | 0.1 | 0.25 |
| 1 | 0.2 | 0.8 | 0.4 | 4 |

Source risk is $0.8(0.1)+0.2(0.4)=0.16$.
Target risk is $0.2(0.1)+0.8(0.4)=0.34$.
Under stable $P(Y\mid X)$ and target support contained in source support,

$$
R_t(f)=\mathbb E_s\left[\frac{p_t(X)}{p_s(X)}\ell(f(X),Y)\right].
$$

The weighted source calculation gives $0.8(0.25)(0.1)+0.2(4)(0.4)=0.34$.
This verifies the algebra, not the real-world invariance assumption.
For 80/20 source rows, the weight-based effective sample size is

$$\frac{(80(0.25)+20(4))^2}{80(0.25)^2+20(4)^2}=\frac{400}{13}\approx30.77.$$

Large weights can reduce precision; clipping them changes the estimate and can introduce bias.

### Receive target labels

**New task:** use a small representative labeled target sample to test the assumed relationship.

**Picture and change:** reserve separate target evaluation; use adaptation labels for calibration or fitting, not both fitting and final scoring.

**Check:** target conditional error rates $(0.1,0.7)$ would instead give risk $0.2(0.1)+0.8(0.7)=0.58$.
Input reweighting alone still returns $0.34$, so it cannot repair that changed error relationship.
Allow sampling uncertainty before concluding that a finite sample establishes concept drift.

**Use when:** labeled evidence can discriminate assumptions that target inputs alone cannot test.

### Lose support or the labeling invariant

**New task:** deploy where the source has no observations or the target label rule differs.

**Picture and change:** restrict claims, collect justified labels, or abstain rather than inventing density ratios.

**Check:** if source mass on bin 1 is zero, its target mass $0.8$ has no source-label support.
With error $0.1$ in bin 0 and bounded loss, target risk can range from $0.02$ to $0.82$.
Separately, uniform inputs can accompany either $Y=X$ or $Y=1-X$.
Predictor $f(X)=X$ then has risk zero or one despite identical input distributions.

**Use when:** overlap or conditional invariance fails; matching feature distributions is not proof of transfer.

<!-- learning:core-end -->

<div data-learning-extension>

## Approaches

- Importance weighting requires the named invariant and support.
- Target fine-tuning requires representative supervision and separate evaluation.
- Feature alignment requires preserving label-relevant distinctions.
- Self-training can reinforce confident mistakes; validate its assumptions and calibration.
- Domain-specific normalization or adapters change the fitted model, not identifiability by themselves.
- Robust objectives compare the environments actually observed, not every possible future shift.

Under label shift, a fixed classifier statistic can satisfy $q_t=C\pi_t$.
Columns of $C$ are its stable prediction distributions given each class; $\pi_t$ contains target class priors.
For binary true-positive/false-positive rates $0.8/0.2$ and target positive-prediction rate $0.5$, the target prior is

$$\pi_t(1)=\frac{0.5-0.2}{0.8-0.2}=0.5.$$

If the two conditional rates coincide, this statistic cannot identify that prior.

## Evaluation

Compare source-only and target-supervised baselines on a true target holdout.
Report target slices, calibration, source retention, uncertainty, and negative transfer.
Reusing target evaluation labels for adaptation requires a new evaluation split.

## In an interview

Can unlabeled target inputs distinguish the two opposite label rules above?

<details><summary>Answer</summary>

No. Their input likelihoods agree exactly.
Target labels or additional structural assumptions are necessary.

</details>

## Common confusions

Aligned features or lower adaptation loss do not establish preserved class structure.
Biased target labels can worsen transfer even when their training loss decreases.

</div>

*Related: [cross-validation strategies](/concepts/cross-validation-strategies/), [epistemic versus aleatoric uncertainty](/concepts/epistemic-vs-aleatoric-uncertainty/), and [calibration](/concepts/calibration/).*
