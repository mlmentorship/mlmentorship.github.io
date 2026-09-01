---
title: "Reproducibility and fair model comparison"
description: "Compare models with matched resources, repeated runs, complete records, and tests that another person can reproduce."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A reproducible ML result can be rebuilt from recorded code, data, configuration, and environment. A fair comparison gives each method a matched and clearly reported chance to succeed.

## Why AI labs care

A model result can change because of:

- random initialization;
- data order and sampling;
- hyperparameter search;
- training duration;
- data cleaning;
- hardware and numerical precision;
- evaluation prompts and decoding settings;
- code or dependency changes.

If these factors are not controlled or recorded, a small gain may not come from the proposed method.

## Three useful terms

- **Repeatability:** the same team runs the same setup again and gets a compatible result.
- **Reproducibility:** another person can rebuild the result from the provided records and artifacts.
- **Replicability:** an independent implementation or dataset supports the same claim.

Interviewers often use these words loosely. Define what you mean and focus on the evidence.

## Match the comparison

A fair baseline should receive a reasonable version of the same resources:

| Resource | Question to ask |
| --- | --- |
| Data | Did both methods train on the same examples and tokens? |
| Parameters | Is the gain only due to a larger model? |
| Compute | Did one method train longer or use more hardware? |
| Tuning | Did both methods get a similar search budget? |
| Code quality | Is one baseline poorly implemented? |
| Evaluation | Were prompts, decoding, and scoring identical? |
| Selection | Were both results chosen using the same rule? |

Exact matching is not always possible. Report the difference and explain why it is needed.

<!-- visual:fair-comparison-evidence-ledger -->
<figure class="learning-figure" aria-labelledby="fair-comparison-ledger-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="fair-comparison-ledger-title">Distinguish a lucky leaderboard win from evidence that estimates a method effect.</p>
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
	<figcaption><strong>Read it this way:</strong> read down the left panel before comparing scores: unequal data, compute, and search make the winning method ambiguous, and selecting only each best run hides stability. On the right, fairness starts before training with one declared opportunity contract and continues after training by retaining every planned run, including failures. Summarize that complete set with effect size, uncertainty, and cost, not the luckiest cell. Original synthesis informed by the <a href="https://arxiv.org/abs/2103.03098">benchmark-variance study</a>, <a href="https://aclanthology.org/D19-1224/">experimental-reporting study</a>, and <a href="https://jmlr.org/papers/v22/20-303.html">NeurIPS reproducibility report</a>.</figcaption>
</figure>

## Repeat the sources of variation

A single seed hides training variance. Several seeds help estimate it.

The number of runs depends on cost and observed variance. Use cheap pilot runs to learn which factors matter. For an expensive frontier run, teams may not repeat the full run many times. They can still repeat smaller-scale tests, replay critical data ranges, compare checkpoints, and state uncertainty honestly.

Do not treat deterministic execution as the only goal. A fully deterministic run can prove that code is repeatable. It may hide the normal variation seen in production training. Use deterministic tests for debugging and representative repeated runs for scientific claims.

## Keep an experiment record

Each run should record:

- code commit;
- data and mixture version;
- model configuration;
- optimizer and schedule;
- random seeds;
- hardware and precision;
- dependency and kernel versions;
- checkpoint source;
- evaluation version;
- decoding settings;
- run status and failure reason;
- links to logs and artifacts.

Failed runs are part of the evidence. Removing them can make an unstable method look reliable.

## Small example

A new attention change improves validation loss in one run.

Before making a mechanism claim:

1. Run both models with several matched seeds at a smaller scale.
2. Match tokens, parameters, and tuning budget.
3. Check whether the gain is larger than run-to-run variation.
4. Measure the internal effect predicted by the claim.
5. Repeat on a second task or scale only if the claim requires it.
6. Save every run, including failures.

If the gain appears only for the best new-model seed, the evidence is weak.

## Reproducing a training failure

For a failure, preserve the first bad transition:

- exact input batch or task;
- checkpoint and optimizer state;
- random-number state;
- world size and rank mapping;
- software and kernel versions;
- logs before and after the event.

Restoring only model weights may not reproduce a failed training step.

## In an interview

A strong answer includes:

1. The claim and the main source of variation.
2. Matched data, compute, capacity, and tuning.
3. Repeated runs or a clear reason they are not feasible.
4. Effect size and uncertainty, not only the best score.
5. Complete run lineage and failed runs.
6. A test that measures the claimed mechanism.
7. A result that would reject the claim.

## Common mistakes

- Comparing a tuned new method with an untuned baseline.
- Reporting the best seed.
- Changing data and model code at the same time.
- Reusing the test set to choose a checkpoint.
- Recording model weights without optimizer, data, and environment state.
- Claiming reproducibility because a notebook ran once.
- Treating a tiny average gain as useful when its cost is much higher.

*Related: [ML data lineage and versioning](/concepts/ml-data-lineage-versioning/), [design an ablation study](/questions/design-ablation-study/), [debug a frontier LLM training run](/questions/debug-frontier-llm-training-run/), and [loss spikes at scale](/concepts/loss-spikes-at-scale/).*