---
title: "Reproducibility and fair model comparison"
description: "Compare models with matched resources, repeated runs, complete records, and tests that another person can reproduce."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A model comparison needs complete runs and a declared resource budget.
A reproducible record lets another person rebuild those runs from code, data, configuration and environment.

## Problem

A candidate has a better best accuracy than the baseline. Should you recommend it?
Compare three hypothetical paired runs with equal tokens, tuning opportunity and evaluation settings.
The pair mapping was specified before scores were observed. None of these planned runs failed.

## Picture

Give every planned run a row before training begins.
Put the baseline and candidate side by side. Compare within each row, then summarize all rows.
Keep a failed run in its row with its status.

<!-- visual:fair-comparison-evidence-ledger -->
<figure class="learning-figure" aria-labelledby="fair-comparison-ledger-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="fair-comparison-ledger-title">Unequal tuning opportunities prevent the best observed scores from isolating method effects</p><div class="visual-argument" data-visual-argument="fair-comparison-evidence-ledger" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 176" role="img" aria-label="Unequal tuning opportunities confound the method comparison. Method B received ten times the search trials. Its best score cannot isolate the method effect without a matched tuning comparison."><title>Unequal tuning opportunities confound the method comparison</title><desc>Method B received ten times the search trials. Its best score cannot isolate the method effect without a matched tuning comparison.</desc><text x="141.43" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="141.43" dy="0">trials</tspan></text><text x="273.81" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="273.81" dy="0">reported</tspan></text><text x="63.24000000000001" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="63.24000000000001" dy="0">method</tspan><tspan x="63.24000000000001" dy="18">A</tspan></text><g data-proof-status="value"><rect data-proof-cell="0,0" x="75.24000000000001" y="36" width="127.38" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="138.93" y="74" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="138.93" dy="0">10</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="0,1" x="207.62" y="36" width="127.38" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="271.31" y="71" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="271.31" dy="0">best A</tspan></text></g><text x="63.24000000000001" y="118" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="63.24000000000001" dy="0">method</tspan><tspan x="63.24000000000001" dy="18">B</tspan></text><g data-proof-status="value"><rect data-proof-cell="1,0" x="75.24000000000001" y="100" width="127.38" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><text x="138.93" y="138" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="138.93" dy="0">100</tspan></text><path data-proof-warning="1,0" d="M79.24000000000001 108v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g><g data-proof-status="value"><rect data-proof-cell="1,1" x="207.62" y="100" width="127.38" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="271.31" y="135" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="271.31" dy="0">best B</tspan></text></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-grid--two" role="group" aria-label="Side-by-side comparison of an unfair best-run leaderboard and a fair complete experiment ledger">
		<section class="visual-panel" aria-labelledby="best-run-ledger-title">
			<h4 id="best-run-ledger-title">Best-run leaderboard</h4>
			<p>The methods receive different opportunities, then only each winner survives reporting.</p>
			<table class="cm-grid" aria-label="Unequal comparison that cannot isolate a method effect">
				<thead><tr><th scope="col">evidence choice</th><th scope="col">baseline</th><th scope="col">new method</th></tr></thead>
				<tbody>
					<tr><th scope="row">data and tokens</th><td>smaller set</td><td>larger set</td></tr>
					<tr><th scope="row">compute and capacity</th><td>lower</td><td>higher</td></tr>
					<tr><th scope="row">tuning opportunity</th><td>few trials</td><td>many trials</td></tr>
					<tr><th scope="row">selection rule</th><td>best observed</td><td class="cm-selected">best observed</td></tr>
					<tr><th scope="row">failures</th><td>omitted</td><td>omitted</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">conclusion allowed: one selected setup scored higher</p>
		</section>
		<section class="visual-panel" aria-labelledby="complete-ledger-title">
			<h4 id="complete-ledger-title">Matched, complete evidence</h4>
			<p>Predeclare one comparison contract, then retain every planned outcome under it.</p>
			<table class="cm-grid" aria-label="Matched comparison contract and complete repeated-run record">
				<thead><tr><th scope="col">record</th><th scope="col">baseline</th><th scope="col">new method</th></tr></thead>
				<tbody>
					<tr><th scope="row">data · tokens · evaluation</th><td class="cm-selected">matched</td><td class="cm-selected">matched</td></tr>
					<tr><th scope="row">capacity · compute · tuning</th><td class="cm-selected">matched</td><td class="cm-selected">matched</td></tr>
					<tr><th scope="row">selection rule</th><td class="cm-selected">predeclared</td><td class="cm-selected">same rule</td></tr>
					<tr><th scope="row">run 1</th><td>recorded</td><td>recorded</td></tr>
					<tr><th scope="row">run 2</th><td>recorded</td><td>recorded</td></tr>
					<tr><th scope="row">run 3</th><td>recorded</td><td>failed, retained</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">conclusion allowed: effect size + uncertainty + disclosed cost</p>
		</section>
	</div>
	<div class="visual-derivation-caption">compare the resource rows before the scores. Unequal data, compute or search can explain a win. The right panel declares the comparison before training and retains every planned run, including failures. Report the complete set's effect size, uncertainty and cost.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="fair-comparison-evidence-ledger"><strong>Check:</strong> Why does the best observed score alone not isolate a method effect when tuning opportunities differ?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="fair-comparison-evidence-ledger">Method B received ten times the search trials. Its best score cannot isolate the method effect without a matched tuning comparison.</span></figcaption>
</figure>
<p class="diagram-source">Original synthesis informed by the <a href="https://arxiv.org/abs/2103.03098">benchmark-variance study</a>, <a href="https://aclanthology.org/D19-1224/">experimental-reporting study</a>, and <a href="https://jmlr.org/papers/v22/20-303.html">NeurIPS reproducibility report</a>.</p>

## Repeat the sources of variation

Repeat the variation relevant to the claim: initialization, data order, training stochasticity or evaluation sampling.
Pairing means a declared correspondence between runs. Equal seed numbers alone do not guarantee useful pairing across different programs.
For this example, each row is one paired training repeat evaluated on the same held-out set.
Evaluation-sample uncertainty is a separate quantity.

## Small example

| Planned pair | Baseline accuracy | Candidate accuracy | Candidate minus baseline |
| --- | --- | --- | --- |
| 1 | 0.80 | 0.83 | 0.03 |
| 2 | 0.82 | 0.81 | -0.01 |
| 3 | 0.78 | 0.79 | 0.01 |

The baseline mean is $0.80$ and the candidate mean is $0.81$.
The mean paired difference is $0.01$: one percentage point.
One pair regresses. The best candidate's three-point paired gain is not the average gain.

Let $d_j$ be the difference in row $j$, and let $\bar d$ be their mean.
The centered differences are $[0.02,-0.02,0]$.
Their sample standard deviation is

$$
s_d=\sqrt{\frac{(0.02)^2+(-0.02)^2+0^2}{3-1}}=0.02.
$$

Report the complete ledger, mean difference, declared uncertainty method and costs.
Three rows describe this fixture; they are insufficient evidence for a general reliability claim.
If a fourth planned run fails, retain it and define how failures enter the comparison's outcome.

## Change the comparison

### 1. The candidate uses twice the compute

**New task:** isolate the method's gain at equal compute.

**Picture and change:** add a compute-matched baseline. Keep the larger-system comparison separately labeled.
Match tuning opportunity too.

**Check:** one versus two GPU-hours on the same hardware is unequal compute, even with identical seed labels.

**Use when:** architecture or training duration changes resource use.

### 2. Replay a failed training step

**New task:** reproduce a NaN in one distributed transition.

**Picture and change:** save the pre-failure checkpoint, next batch, optimizer, schedule, data position and random-number state.
Record rank mapping, world size, software, kernels and precision.

**Check:** loading the weights with a different batch does not replay the same transition.

**Use when:** the target is a failure mechanism. Keep this replay separate from representative performance repeats.

## Selection can survive repetition

Suppose the baseline receives two tuning trials and the candidate receives fifty.
Rerunning their selected winners leaves that unequal selection opportunity unchanged.
Define the tuning and selection rules before comparing the methods.

## Try a different claim

Does an exactly repeated result prove a useful performance gain?

<details>
<summary>Check your answer</summary>

It demonstrates repeatability of that setup.
A useful comparative gain needs matched opportunities, relevant variation and an outcome worth its cost.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

<a id="why-ai-labs-care"></a>

## Sources of variation

Initialization, data sampling, cleaning, tuning, training duration, hardware, precision, evaluation and code changes can explain a score change.
Control or record each factor relevant to the claim.

## Three useful terms

- **Repeatability:** rerun the same setup and obtain a compatible result.
- **Reproducibility:** rebuild the result from supplied records and artifacts.
- **Replicability:** test the claim with an independent implementation or dataset.

Terminology conventions differ. State the evidence your chosen term requires.

## Match the comparison

| Resource | Recorded comparison |
| --- | --- |
| Data | Examples, tokens and cleaning |
| Capacity | Model size and parameter use |
| Compute | Work, hardware, time and precision |
| Tuning | Search budget and selection rule |
| Implementation | Baseline correctness and optimization |
| Evaluation | Prompts, decoding, scorer and held-out set |

If exact matching is impossible, report the difference and narrow the claim.

## Keep an experiment record

Use immutable run IDs. Retain code, data/mixture, model configuration, optimizer, schedule, seeds, hardware, precision and dependency/kernel versions.
Include checkpoint lineage, evaluator, decoding, status, failure reason, logs and artifact locations.

## Reproducing a training failure

Keep logs on both sides of the first bad transition.
For expensive runs, repeat affordable scale tests or replay critical ranges.
Report which full-run variation remains unmeasured.
Deterministic replay helps debugging; representative repeats measure variability.

## In an interview

State the claim, its alternative explanation, the matched comparison and a result that would reject your explanation.

## Common mistakes

Do not tune on the final test set or omit failed attempts.
Changing data and model together needs additional controls to isolate their effects.

*Related: [ML data lineage and versioning](/concepts/ml-data-lineage-versioning/), [design an ablation study](/questions/design-ablation-study/), [debug a frontier LLM training run](/questions/debug-frontier-llm-training-run/), and [loss spikes at scale](/concepts/loss-spikes-at-scale/).*

</div>