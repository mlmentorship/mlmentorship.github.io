---
title: "Data leakage and point-in-time correctness"
description: "A feature is valid only if it was available when the prediction was made. Prevent target, temporal, group, preprocessing, and evaluation leakage."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["data leakage", "target leakage", "look-ahead bias", "point in time join", "as-of join", "train test leakage"]
roles: ["Applied Scientist", "Machine Learning Engineer", "Research Engineer"]
rounds: ["Evaluation", "ML system design", "Debugging"]
difficulty: "Intermediate"
priority: "Core"
prerequisites: ["cross-validation-strategies"]
---

## Summary

Data leakage occurs when training or evaluation uses information that would not be available for the intended prediction. The model then learns from the target, the future, a related entity, or the evaluation set.

Point-in-time correctness is the operational test: for a prediction made at time $t$, every feature value must come from information available by $t$. This rule must hold in dataset construction, feature computation, splitting, tuning, and final evaluation.

## Start with the prediction contract

Define four items before building the dataset:

1. **Entity:** what receives the prediction?
2. **Prediction time:** when is the score produced?
3. **Outcome window:** which future interval defines the label?
4. **Feature cutoff:** which events are allowed to contribute?

Example: predict whether an account will default within 90 days, using information available at the end of each month. A payment posted after month end cannot appear in that month's feature row, even if the warehouse table now contains it.

## Target leakage

Target leakage gives the model a direct or indirect copy of the outcome.

Examples:

- a fraud feature records the result of a later investigation;
- a churn model includes an account-closure code;
- a hospital model uses a treatment ordered after diagnosis;
- a loan model uses a collection status created after default.

Leakage can hide behind a plausible column name. Trace how and when each field is produced.

## Temporal leakage

Temporal leakage moves future information into the past.

Common causes include:

- random train/test splits for time-dependent data;
- aggregates computed over the full table rather than up to prediction time;
- backfilled records with current values;
- joins that select the latest row instead of the latest eligible row;
- labels whose observation window overlaps the feature window.

Use chronological evaluation when the deployment problem is chronological. Training rows may use history before their own prediction times, not history before the date when the dataset was exported.

## Point-in-time joins

A point-in-time join selects the newest feature row whose event time and availability time are no later than the prediction time.

Two timestamps may matter:

- **event time:** when the real-world event occurred;
- **availability time:** when the system received and could use it.

A transaction may occur on Monday but arrive on Wednesday. A Tuesday prediction cannot use it. Joining only on event time still leaks information.

<!-- visual:point-in-time-availability-boundary -->
<figure class="learning-figure editorial-illustration" aria-labelledby="point-in-time-availability-boundary-edition-title"><p class="visual-title" id="point-in-time-availability-boundary-edition-title">Availability, training-only fitting, and untouched evaluation are separate leakage boundaries</p><div class="visual-argument" data-visual-argument="point-in-time-availability-boundary" data-argument-kind="timeline" data-argument-composition="leakage-boundaries"><svg viewBox="0 0 360 1795" role="img" aria-label="Availability, training-only fitting, and untouched evaluation are separate leakage boundaries. Use only features available at prediction time. Keep related entities and future periods out of the training side. Fit preprocessing on training data alone. Validation may guide choices; an untouched test set checks the chosen pipeline. Feeding test outcomes back into selection leaks evaluation evidence. The noon example shows why an earlier event can still arrive too late."><title>Availability, training-only fitting, and untouched evaluation are separate leakage boundaries</title><desc>Use only features available at prediction time. Keep related entities and future periods out of the training side. Fit preprocessing on training data alone. Validation may guide choices; an untouched test set checks the chosen pipeline. Feeding test outcomes back into selection leaks evaluation evidence. The noon example shows why an earlier event can still arrive too late.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">One record, two different timestamps</tspan></text><rect x="35" y="181" width="290" height="42" style="fill:var(--viz-warning-bg);opacity:.7"></rect><path d="M35 202L325 202" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="35" y="108" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="35" dy="0">Event</tspan></text><circle cx="35" cy="202" r="7" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="35" y="265" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="35" dy="0">11:50</tspan></text><text x="180" y="108" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="180" dy="0">Prediction</tspan></text><circle cx="180" cy="202" r="7" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="180" y="265" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">12:00</tspan></text><text x="325" y="108" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--viz-input-stroke)"><tspan x="325" dy="0">Available</tspan></text><circle cx="325" cy="202" r="7" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="325" y="265" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--c-text)"><tspan x="325" dy="0">12:10</tspan></text><path d="M180 136L180 324" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="366" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">The noon decision lands</tspan><tspan x="20" dy="23">before availability</tspan></text><text x="20" y="430" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">An early event is not an available</tspan><tspan x="20" dy="18">feature.</tspan></text><text x="20" y="550" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Keep related records together</tspan></text><text x="65" y="612" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="65" dy="0">A: train</tspan></text><rect x="20" y="635" width="90" height="88" data-picture-leakage-group="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="44" y="687" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="44" dy="0">1</tspan></text><text x="86" y="687" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="86" dy="0">3</tspan></text><text x="180" y="612" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="180" dy="0">B: train</tspan></text><rect x="135" y="635" width="90" height="88" data-picture-leakage-group="1" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="159" y="687" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="159" dy="0">5</tspan></text><text x="201" y="687" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="201" dy="0">7</tspan></text><text x="295" y="612" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="295" dy="0">C: test</tspan></text><rect x="250" y="635" width="90" height="88" data-picture-leakage-group="2" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="274" y="687" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="274" dy="0">9</tspan></text><text x="316" y="687" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="316" dy="0">11</tspan></text><path d="M65 741L120 790M111.36589708558272 787.6649910357237L120 790L116.68754542277073 781.6917122899004" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><path d="M180 741L180 790M176 782L180 790L184 782" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><path d="M295 741L295 773" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M283 778L307 802" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M307 778L283 802" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><rect x="80" y="810" width="145" height="60" data-picture-leakage-fit-mean="4" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="152.5" y="846" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="152.5" dy="0">Train mean: 4</tspan></text><text x="340" y="842" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--viz-warning-stroke)"><tspan x="340" dy="0">Test does</tspan><tspan x="340" dy="18">not fit</tspan></text><text x="20" y="932" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Apply saved mean 4 to C</tspan></text><text x="20" y="985" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Held-out inputs</tspan></text><circle cx="85" cy="1038" r="24" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="85" y="1044" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="85" dy="0">9</tspan></text><path d="M85 1074L85 1120M81 1112L85 1120L89 1112" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="85" cy="1155" r="24" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="85" y="1161" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="85" dy="0">5</tspan></text><circle cx="265" cy="1038" r="24" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="265" y="1044" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="265" dy="0">11</tspan></text><path d="M265 1074L265 1120M261 1112L265 1120L269 1112" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="265" cy="1155" r="24" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="265" y="1161" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="265" dy="0">7</tspan></text><text x="20" y="1235" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">New C values do not refit the saved</tspan><tspan x="20" dy="18">mean.</tspan></text><text x="20" y="1320" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Choose first; test without choosing</tspan><tspan x="20" dy="18">again</tspan></text><circle cx="80" cy="1410" r="27" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="80" y="1416" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="80" dy="0">pick</tspan></text><path d="M120 1410L240 1410M232 1414L240 1410L232 1406" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="280" cy="1410" r="27" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="280" y="1416" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="280" dy="0">test</tspan></text><text x="80" y="1480" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="80" dy="0">Choose with</tspan><tspan x="80" dy="18">validation</tspan></text><text x="280" y="1480" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="280" dy="0">Untouched test</tspan></text><path d="M280 1530L280 1560" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M280 1560L196 1560" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M164 1560L80 1560" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M80 1560L80 1530" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M168 1548L192 1572" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M192 1548L168 1572" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="20" y="1640" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Test outcomes must not return to</tspan><tspan x="20" dy="18">choice.</tspan></text><text x="20" y="1720" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Future or related examples stay on</tspan><tspan x="20" dy="18">the held-out side.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 355" role="img" aria-labelledby="point-in-time-availability-boundary-edition-svg-title point-in-time-availability-boundary-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="point-in-time-availability-boundary-edition-svg-title">An earlier event can still be unavailable at prediction time</title><desc id="point-in-time-availability-boundary-edition-svg-desc">The Monday-noon cutoff divides knowable evidence from the future. Payment A occurs Monday at 09:00 and is available at 09:05. Payment B occurs at 10:00 but is unavailable until Wednesday. Its early event timestamp cannot authorize using its later-arriving information.</desc><text x="180" y="22" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Prediction cutoff: Monday 12:00</tspan></text><rect x="202" y="44" width="142" height="257" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3;stroke-dasharray:5 3"></rect><path d="M201 44L201 301" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><text x="105" y="69" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="105" dy="0">KNOWABLE</tspan></text><text x="272" y="69" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="272" dy="0">FUTURE</tspan></text><text x="20" y="111" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">Payment A</tspan></text><text x="20" y="137" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">event Mon 09:00</tspan></text><g data-editorial-node><rect x="25" y="151" width="160" height="49" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="105" y="171" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="105" dy="0">available Mon</tspan><tspan x="105" dy="19">09:05</tspan></text></g><text x="20" y="236" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">Payment B</tspan></text><text x="20" y="262" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">event Mon 10:00</tspan></text><g data-editorial-node><rect x="214" y="226" width="120" height="61" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="274" y="252" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="274" dy="0">available</tspan><tspan x="274" dy="19">Wednesday</tspan></text></g><path d="M138 269L211 260" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M205.31372515598005,264.0824353513884L211,260L204.49244169442863,257.4209139410269" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="180" y="330" style="font-size:14px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="180" dy="0">Earlier event != earlier availability</tspan></text></svg><div class="visual-derivation-caption"><strong>Read it this way:</strong> Payment B happened before noon but arrived on Wednesday. An event-time-only join leaks future availability. The layout encodes the cutoff, not proportional elapsed time.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="point-in-time-availability-boundary"><strong>Check:</strong> Why is Monday's 10:00 payment unavailable to a Monday-noon prediction?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="point-in-time-availability-boundary">Use only features available at prediction time. Keep related entities and future periods out of the training side. Fit preprocessing on training data alone. Validation may guide choices; an untouched test set checks the chosen pipeline. Feeding test outcomes back into selection leaks evaluation evidence. The noon example shows why an earlier event can still arrive too late. <strong>Can a later-arriving record be used because its event happened earlier?</strong></span></figcaption></figure>

A simplified rule is:

```text
feature.entity_id = example.entity_id
feature.available_at <= example.prediction_at
choose the latest eligible feature row
```

Test this rule with rows that arrive late, are corrected, or are backfilled.

## Group leakage

Related examples can leak identity or content across splits.

Examples:

- passages from one document appear in train and test;
- visits from one patient appear in both sets;
- near-duplicate images cross the boundary;
- future events from one user appear in training for an earlier test event;
- products with shared parent records cross a split.

Choose the split unit from the deployment claim. If the claim is generalization to new patients, split by patient. If the claim is future behavior for known users, use a temporal split within user.

## Preprocessing leakage

Fit learned preprocessing only on the training fold. This includes:

- normalization statistics;
- vocabulary and token frequency;
- imputation values;
- feature selection;
- dimensionality reduction;
- target encoding;
- synthetic oversampling.

A pipeline should fit each transform on training data and apply the frozen transform to validation or test data. Fitting once on the full dataset makes the held-out set influence training.

## Evaluation leakage

Repeated test-set use turns the test set into training information. Teams leak evaluation data when they:

- choose prompts or checkpoints from test results;
- inspect test errors and change the model repeatedly;
- publish the best of many unreported runs;
- train on benchmark solutions or near duplicates;
- let human raters see model identity or treatment.

Keep a validation set for iteration and a final test set for the last estimate. For long-running programs, rotate fresh held-out task families and record all evaluations.

## Worked example

A recommendation model predicts clicks at noon. Its seven-day item popularity feature is rebuilt overnight from the current warehouse table.

An offline row for Monday noon is exported on Friday. A naive query computes popularity from the seven days ending Friday and joins it to Monday. The feature includes events from Monday afternoon through Friday.

The correct query computes the aggregate using records available by Monday noon. Offline and online feature code should share this cutoff rule.

## Detection and tests

Use several checks:

1. Compare feature timestamps with prediction timestamps.
2. Remove suspicious high-performing features and measure the drop.
3. Train on past windows and evaluate on a later untouched window.
4. Search for columns created after the outcome.
5. Check duplicates and entity overlap across splits.
6. Replay online feature generation for historical examples.
7. Add unit tests for late arrival, backfill, and correction cases.

An unexpectedly large offline gain deserves a leakage review before model celebration.

## In an interview

Use this order:

1. Define entity, prediction time, outcome window, and feature cutoff.
2. Name target, temporal, group, preprocessing, and evaluation leakage.
3. Describe an as-of join using availability time.
4. Match the split unit to the deployment claim.
5. Add replay and boundary tests.
6. Explain how training and serving share feature definitions.

## Common mistakes

- Saying a random split is valid because rows are shuffled.
- Using event time when availability time controls deployment.
- Grouping by user but allowing future user events into past rows.
- Fitting scalers or vocabularies before cross-validation.
- Treating a feature store as automatic proof of point-in-time correctness.
- Reusing a benchmark until it no longer measures generalization.

## Practice next

Apply these rules in [cross-validation strategies](/concepts/cross-validation-strategies/), [feature-store design](/questions/design-feature-store/), and [debugging offline-online disagreement](/questions/debug-offline-online-metric-gap/).
