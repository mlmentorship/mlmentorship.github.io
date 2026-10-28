---
title: "Critique an ML paper you have not seen before"
description: "A research-depth framework for claims, baselines, evidence, leakage, compute fairness, ablations, and generalization."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

Assess an ML paper's central claim by identifying what its reported experiment establishes and which alternative explanation remains unresolved.

## Problem

**Context.** You have ten minutes and only an abstract, method figure, and main table from a hypothetical routing paper.
The table reports accuracy and total tuning/training cost for two recipes.

**Task.** Separate the empirical observation from the mechanism and efficiency claims, then propose one feasible deciding experiment.
Explain how its possible outcomes would change your conclusion.

**Constraints.** Do not invent missing implementation details, variance estimates, datasets, or unpublished results.
Account for unequal resource budgets and distinguish absence of evidence from evidence that the mechanism cannot work.

You have ten minutes to inspect an abstract, method figure and main result table.
An explicitly hypothetical paper claims: “Our routing mechanism improves accuracy efficiently.”

| Main-table recipe | Accuracy | Total tuning/training cost |
| --- | --- | --- |
| Baseline | $80\%$ | $10$ GPU-hours |
| Routing | $84\%$ | $20$ GPU-hours |

Choose the most important unresolved explanation and propose one feasible next experiment.
Do not assume anything else about the unpublished method or evaluation.

## Picture

The reported four-percentage-point gain is a real strength within this hypothetical table.
But routing and resource advantage changed together.
Matched resources could distinguish a resource-only explanation from a gain that survives the comparison.

<!-- visual:paper-critique-prioritization-funnel -->
<figure class="learning-figure editorial-illustration" aria-labelledby="paper-critique-prioritization-funnel-edition-title"><p class="visual-title" id="paper-critique-prioritization-funnel-edition-title">Four more accuracy points came with twice the compute</p><svg viewBox="0 0 360 441" role="img" aria-labelledby="paper-critique-prioritization-funnel-edition-svg-title paper-critique-prioritization-funnel-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="paper-critique-prioritization-funnel-edition-svg-title">Four more accuracy points came with twice the compute</title><desc id="paper-critique-prioritization-funnel-edition-svg-desc">The reported baseline uses 10 GPU-hours for 80 percent accuracy; the new recipe uses 20 for 84 percent. Separate zero-based common scales show accuracy and compute. Three proposed matched-compute paired runs have question marks because their outcomes are not known.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Reported comparison; not matched compute</tspan></text><text x="15" y="60" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="15" dy="0">accuracy / 100%</tspan></text><text x="15" y="98" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="15" dy="0">base</tspan></text><rect x="62" y="84" width="176" height="20" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="332" y="99" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:end"><tspan x="332" dy="0">80%</tspan></text><text x="15" y="134" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="15" dy="0">new</tspan></text><rect x="62" y="120" width="184.8" height="20" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="332" y="135" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:end"><tspan x="332" dy="0">84%</tspan></text><text x="15" y="179" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="15" dy="0">GPU-hours / 20</tspan></text><text x="15" y="217" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="15" dy="0">base</tspan></text><rect x="62" y="203" width="110" height="20" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="332" y="218" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:end"><tspan x="332" dy="0">10h</tspan></text><text x="15" y="253" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="15" dy="0">new</tspan></text><rect x="62" y="239" width="220" height="20" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="332" y="254" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:end"><tspan x="332" dy="0">20h</tspan></text><text x="180" y="316" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Next: 3 paired runs at 10 hours per</tspan><tspan x="180" dy="19">recipe</tspan></text><text x="180" y="376" style="font-size:25px;font-weight:400;fill:var(--viz-state-stroke);text-anchor:middle"><tspan x="180" dy="0">? ? ?</tspan></text><text x="180" y="415" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Matched-compute gain is not known yet.</tspan></text></svg><figcaption><strong>Read it this way:</strong> The report changes accuracy and compute together. Three paired, matched-compute runs test whether a useful gain survives. Their results are unknown, not implied by the drawing. <span class="visual-reader-check" data-reader-check="paper-critique-prioritization-funnel"><strong>Check:</strong> What comparison could separate a routing benefit from the extra compute budget?</span></figcaption></figure>

## Write the decision record

| Field | This hypothetical decision |
| --- | --- |
| Claim and evidence | Exact abstract quote; main-table rows; versioned paper copy |
| Selected alternative | More resources, rather than routing, explain the useful gain |
| Comparison | Same hardware, data, evaluation and total budget |
| Resource plan | Three paired runs per recipe; each gets two tuning hours and eight final-training hours |
| Primary effect | Paired accuracy difference in percentage points |
| Useful threshold | At least two points, declared before confirmation |
| Opposing predictions | Gain disappears under matching, or a useful gain remains |
| Uncertainty | Declare seed pairing, evaluation sampling unit and analysis method |
| Stop rule | Fixed budget; retain failed attempts and all outcomes |

The six training/tuning allocations total $3\times2\times(2+8)=60$ GPU-hours.
Declare tuning opportunities and account for failed attempts within each allocation.
Budget evaluation overhead separately.
Equal FLOPs, wall time, GPU-hours and tuning opportunities are different contracts; choose one matching the claim.

Suppose a justified analysis reports $+0.2$ points with interval $[-0.6,+1.0]$.
This is an interpretation fixture, not an interval calculated from three seeds.
Its upper endpoint remains below the declared useful threshold.
That outcome would not justify a larger reproduction based on this efficiency claim.

A supported retained three-point gain would weaken the resource-only explanation.
It would not prove the proposed routing mechanism or generalization beyond the evaluated setting.
Training-seed variation and uncertainty over evaluation items need separate treatment.

## Change the threat or decision

### 1. Suspect shared source documents

**New task:** budgets already match, but evaluation items may share training-document ancestry.

**Picture and change:** freeze the original result and build a source-disjoint confirmation set from provenance.
Keep tuning away from it.

**Check:** compare matched clean and overlapping strata.
A clean-set decline alone cannot prove leakage; difficulty or domain may also change.

**Use when:** source overlap directly threatens out-of-source generalization.

### 2. Judge operating value

**New task:** slower batches reduce failure recovery work.

**Picture and change:** measure end-to-end cost, reliability and intervention time, not kernel speed alone.

**Check:** compare the same workload and service objective over the same observation window.
Operational value can justify adoption without establishing a novel learning mechanism.

**Use when:** the decision is deployment rather than scientific novelty.

## Failure and transfer

Requesting five datasets is not discriminating if every experiment retains the same confound.
Which competing predictions would the added observations separate?

Does a fair reproduction retaining the gain prove the mechanism?

<details>
<summary>Check your answer</summary>

No. It supports the observed comparison and weakens the selected resource explanation.
Mechanism evidence still needs an intervention with competing predictions.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## A reading order

1. State the claim precisely.
2. Locate its supporting evidence.
3. Check baseline and resource fairness.
4. Identify leakage, selection and variance threats.
5. Ask what isolates the mechanism.
6. Bound generalization and failure settings.
7. Compare value with compute and complexity.
8. Choose one belief-changing experiment.

## Be fair before you criticize

Distinguish an overstated correct result, useful engineering without novel science, an unsupported mechanism, and an invalid evaluation.
Absence of evidence is not evidence of failure.

<span id="what-an-l4-answer-sounds-like"></span>

## Unfocused critique

“Small gain” needs uncertainty and practical-cost context.
“More baselines” needs a comparison that tests the claim.

<span id="what-an-l5-answer-adds"></span>

## Claim-specific critique

Match the measurement to efficiency, robustness or mechanism, then explain the result that would alter the conclusion.

<span id="what-an-l6-answer-adds"></span>

## Strategic value

Ask what survives scale, what reproduction requires, and whether the contribution is a primitive, recipe or local optimization.

<span id="tells-that-get-you-a-strong-hire-vote"></span>

## Useful evidence

Prioritize impact, acknowledge a genuine strength and make the next test falsifiable.
No response format guarantees a hiring outcome.

<span id="tells-that-get-you-down-leveled"></span>

## Avoid unsupported judgments

Judge evidence, not authors or venue.
Avoid long criticism lists without a decision.

## Common follow-ups

- Would you spend a month reproducing this paper?
- Which claim would you narrow?
- What is the strongest alternative explanation?
- How would you test whether the result survives scale?
- What if the method is slower but easier to operate?

## Sources

Checklist items cover claims, uncertainty and compute; they are not a mechanical accept/reject rubric.

- [NeurIPS 2025 reviewer guidance](https://neurips.cc/Conferences/2025/ReviewerGuidelines).
- [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist).
- [Pineau et al., JMLR reproducibility-program report](https://jmlr.org/papers/v22/20-303.html).

*Related: [design an ablation study](/questions/design-ablation-study/), [bias-variance of estimators](/concepts/bias-variance-of-estimators/), and [lessons from Marin 8B](/guides/lessons-from-marin-8b/).*

</div>
