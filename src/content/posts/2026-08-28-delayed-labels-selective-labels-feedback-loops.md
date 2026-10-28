---
title: "Delayed labels, selective labels, and feedback loops"
description: "Production policies control which labels arrive and when. Track label maturity, selection, action propensities, and feedback before training or evaluating a model."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["delayed labels", "selective labels", "censored labels", "ML feedback loops", "policy-induced data", "label maturity"]
roles: ["Applied Scientist", "Machine Learning Engineer", "Ranking MLE"]
rounds: ["ML system design", "Evaluation", "Experimentation"]
difficulty: "Intermediate"
priority: "Core"
prerequisites: ["data-leakage-point-in-time-correctness", "causal-inference-for-ml-decisions"]
---

## Summary

A production label can be pending, censored, observed, or unavailable under the current policy.
No recorded event does not automatically establish a negative outcome.
Ask whether the revealing action occurred and whether follow-up supports the chosen outcome definition.

Our hypothetical target is a confirmed chargeback within 45 days of an approved payment.
Keep decision-time features frozen; later outcome corrections must not introduce future information into those features.

<!-- visual:policy-label-observation-loop -->
<figure class="learning-figure editorial-illustration" aria-labelledby="policy-label-observation-loop-edition-title"><span class="visual-fragment-anchor" id="policy-label-visual-title" aria-hidden="true"></span><p class="visual-title" id="policy-label-observation-loop-edition-title">Pending, censored, and policy-unobserved are not negative labels</p><svg viewBox="0 0 360 542" role="img" aria-labelledby="policy-label-observation-loop-edition-svg-title policy-label-observation-loop-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="policy-label-observation-loop-edition-svg-title">Pending, censored, and policy-unobserved are not negative labels</title><desc id="policy-label-observation-loop-edition-svg-desc">Aligned schematic timelines share an evaluation cutoff. They distinguish an early known positive, a selected action whose outcome window remains open, a selected action whose observation ends early, and an action never selected by the policy. A task-specific maturity rule controls eligibility; unknown outcomes do not become negatives.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Four different reasons a label may not be</tspan><tspan x="180" dy="19">usable</tspan></text><text x="20" y="66" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">Selected; positive arrives early</tspan></text><path d="M35 94L325 94" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M251 76L251 118" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><text x="251" y="138" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="251" dy="0">cutoff</tspan></text><circle cx="35" cy="94" r="4" style="fill:var(--viz-input-stroke)"></circle><rect x="35" y="85" width="134" height="18" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="97" y="138" style="font-size:12px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="97" dy="0">KNOWN +</tspan></text><text x="20" y="166" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">Selected; window still open</tspan></text><path d="M35 194L325 194" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M251 176L251 218" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><text x="251" y="238" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="251" dy="0">cutoff</tspan></text><circle cx="35" cy="194" r="4" style="fill:var(--viz-input-stroke)"></circle><rect x="35" y="185" width="286" height="18" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="97" y="238" style="font-size:12px;font-weight:700;fill:var(--viz-state-stroke);text-anchor:middle"><tspan x="97" dy="0">PENDING</tspan></text><text x="20" y="266" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">Selected; observation ends early</tspan></text><path d="M35 294L325 294" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M251 276L251 318" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><text x="251" y="338" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="251" dy="0">cutoff</tspan></text><circle cx="35" cy="294" r="4" style="fill:var(--viz-input-stroke)"></circle><rect x="35" y="285" width="126" height="18" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3;stroke-dasharray:5 3"></rect><text x="97" y="338" style="font-size:12px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="97" dy="0">CENSORED</tspan></text><text x="20" y="366" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">Action not selected by policy</tspan></text><path d="M35 394L325 394" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M251 376L251 418" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><text x="251" y="438" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="251" dy="0">cutoff</tspan></text><circle cx="35" cy="394" r="4" style="fill:var(--viz-input-stroke)"></circle><text x="97" y="438" style="font-size:12px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="97" dy="0">UNOBSERVED</tspan></text><text x="180" y="483" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">A task-specific maturity rule decides</tspan><tspan x="180" dy="19">eligibility.</tspan></text><text x="180" y="520" style="font-size:13px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="180" dy="0">Unknown is never a synonym for negative.</tspan></text></svg><figcaption><strong>Read it this way:</strong> These schematic rows distinguish four observation states at one cutoff. An early positive may be known. Other eligibility depends on the declared maturity rule. <span class="visual-reader-check" data-reader-check="policy-label-observation-loop"><strong>Check:</strong> How do a pending outcome and a right-censored outcome differ even when neither is an observed negative?</span></figcaption></figure>

## Delayed labels

An event can occur or be reported after its prediction.
Group evaluation by decision cohort and retain event time, observation coverage, label arrival, and correction history separately.
A confirmed in-window event can establish a positive before day 45.
Absence establishes a negative only under the declared complete-follow-up and reporting rule.

## Worked example

Among 100 fully followed approved payments, two chargebacks are known at day 1 and ten by day 45.
Day 1 therefore has two positives and 98 pending rows, not 98 confirmed negatives.
At day 45, the mature cohort has ten positives and 90 negatives.
The shortcut would have mislabeled eight eventual positives; mature prevalence is $10/100=0.10$, not $2/100$.

| Record | Revealing action | Age / followed days | Observation status |
| --- | --- | --- | --- |
| Event confirmed day 1 | Approved | 1 / 1 | Known positive |
| No event yet | Approved | 7 / 7 | Pending |
| No event; observation stopped | Approved | 50 / 15 | Right-censored |
| Approval never occurred | Blocked | 50 / none | Policy-unobserved approval outcome |
| No event; complete window | Approved | 45 / 45 | Mature negative |

The day-50 row is not mature merely because it is old.

## Right censoring

Without a confirmed event, follow-up ending at day 15 leaves the 45-day binary outcome unresolved.
Survival methods can use event-time and censoring information under explicit assumptions.
Conditional independence of censoring is not automatic: outcome-related departures from coverage can violate it.
Closed cohorts sacrifice freshness and may still have selection bias.

## Selective labels

Repayment after approval, clicks after exposure, and review outcomes after selection have different observation support.
Approved-cohort risk is not automatically risk across all eligible applicants.
Neither chronological splitting nor waiting repairs missing counterfactual outcomes.

## Safe ways to gain labels

**New task:** estimate recent risk before 45-day outcomes mature.

**Picture and change:** validate an early proxy or specify an event-time model; retain mature evaluation as the reference.

**Check:** a seven-day proxy finding eight of ten mature positives has sensitivity $8/10$.
Without its false positives, precision remains unknown.
That proxy has not changed the ground-truth outcome definition.

**Use when:** waiting is costly, with proxy error and cohort age reported explicitly.

## Evaluation under policy change

**New task:** evaluate approvals where the old policy always rejected.

**Picture and change:** identify missing action support before choosing a counterfactual estimator.

**Check:** approval propensity zero makes an approval importance weight undefined, not zero.
Even propensity $0.01$ gives a selected approval weight $100$, creating variance risk.
Weighting requires adequate support and a defensible assignment/observation model.

**Use when:** a proposal changes observed actions; lawful independent evidence or tightly bounded safe collection may be necessary.
Adjudication can assess a proxy without revealing an unperformed action's actual outcome.

## Logging contract

For each decision, log:

- decision timestamp and entity;
- feature-snapshot identifier;
- candidate set and available actions;
- chosen action, score, threshold, model, and policy versions;
- action probability when randomized;
- label-window end and observation end;
- outcome timestamp, observation status, and corrections.

Backfilling labels while recomputing historical features from today's warehouse leaks future information despite correct maturity handling.

<!-- learning:core-end -->

<div data-learning-extension>

## Feedback loops

A deployed policy changes the evidence available to its successor:

```text
model score -> action or ranking -> user exposure -> behavior -> label -> retraining
```

Training on the selected subset can narrow future coverage even when loss on that subset decreases.

## Direct and indirect feedback

Direct feedback changes measured responses, such as ranking-induced clicks reused as labels.
Indirect feedback changes the population or behavior, such as which applicants return under stricter approvals.
Track coverage and action mixtures alongside outcome metrics; retain stable controls when long-term feedback matters.

Randomized audits, permitted independent reviews, and external adjudication can broaden evidence within lawful risk limits.
Shadow predictions alone do not reveal counterfactual actions' outcomes.
Positive-unlabeled methods need assumptions about positive-label selection; survival models need justified censoring assumptions.
Simulation and expert review can inform a decision without becoming measured deployment evidence.
Do not introduce unacceptable risk merely to collect labels.

## In an interview

A shown result receives no click; an unshown result has no click record. Are both negatives?

<details><summary>Answer</summary>

No. The first is a response under one exposure; the second lacks that exposure outcome.
Neither alone establishes universal relevance across positions, users, and contexts.

</details>

## Common mistakes

Report estimand, age, maturity coverage, selection policy, and remaining unknown outcomes together.
A well-defined mature complete-case estimate can still target the wrong population.
Preserve original scores and feature versions when later labels arrive.

</div>

## Practice next

Apply these ideas in [point-in-time correctness](/concepts/data-leakage-point-in-time-correctness/), [counterfactual ranking](/concepts/position-bias-counterfactual-learning-to-rank/), [fraud-system design](/questions/design-fraud-detection/), and [offline-online metric debugging](/questions/debug-offline-online-metric-gap/).
