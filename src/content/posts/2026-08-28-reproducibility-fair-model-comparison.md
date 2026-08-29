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