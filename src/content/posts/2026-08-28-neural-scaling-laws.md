---
title: "Neural scaling laws and compute-optimal training"
description: "Use small training runs to estimate how loss changes with model size, data, and compute, then choose a training plan within a fixed budget."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A neural scaling law is an empirical relation between model quality and resources such as parameters, training data, and compute. Compute-optimal training uses that relation to divide a fixed budget between model size and data.

## Why AI labs care

A frontier training run is too expensive for blind trial and error. Teams use smaller runs to answer practical questions:

- How large should the model be?
- How many tokens should it see?
- Is more data better than more parameters at this budget?
- Which run is likely to meet a target loss?
- Is the planned run limited by compute, data, or serving cost?

Scaling laws guide these choices. They do not remove the need for experiments.

## A common form

One simple model for validation loss is:

$$
L(N,D) = L_\infty + A N^{-\alpha} + B D^{-\beta},
$$

where:

- $N$ is the parameter count;
- $D$ is the number of training tokens;
- $L_\infty$ is an estimated lower limit;
- $A$, $B$, $\alpha$, and $\beta$ are fit from experiments.

The exact form and exponents depend on the model family, data, tokenizer, training recipe, and loss. Do not copy constants from another project without checking them.

For a dense transformer, training compute is roughly proportional to $N D$. This makes model size and token count compete for the same budget.

## Compute-optimal allocation

A model that is too large for its token budget is undertrained. A model that is too small may use many tokens with limited capacity.

The compute-optimal point balances these sources of loss for the chosen budget. The Chinchilla study showed that many earlier language models were too large and trained on too few tokens for their compute.

Fit the token-to-parameter trade-off to the current data and training recipe instead of copying a published ratio.

## How to fit a useful scaling study

1. Choose a stable model family and training recipe.
2. Run a grid of smaller model sizes and token budgets.
3. Keep data quality and evaluation fixed.
4. Train long enough to measure each curve cleanly.
5. Fit the relation on held-out runs.
6. Check prediction error, not only fit on the observed points.
7. Test one larger run before committing the full budget.

Record failed and unstable runs. Excluding them can make a plan look safer than it is.

## Small example

A team can afford one fixed amount of training compute. It is choosing between:

- a larger model trained for fewer tokens;
- a smaller model trained for more tokens.

Small-run curves show that the larger model is still improving quickly at the end of training, while the smaller model has started to flatten. The team should test an intermediate allocation instead of choosing either extreme.

The decision should also include serving cost. A slightly better large model may be too slow or expensive for the product.

## What can break the extrapolation

Scaling curves are local evidence. They can fail when:

- the data mixture changes;
- data quality falls at larger scale;
- the tokenizer changes;
- the architecture or optimizer changes;
- training becomes unstable;
- repeated data causes memorization;
- downstream capabilities appear at a different rate than validation loss;
- hardware efficiency changes with model size;
- post-training changes the ranking of base models.

A smooth loss curve does not guarantee a specific reasoning, safety, or tool-use capability.

## Data-constrained training

High-quality data may be limited. Repeating data can still help, though gains usually weaken with more repeats and memorization risk grows.

A data-constrained plan should track:

- effective repeats by source;
- train and validation loss gap;
- benchmark contamination;
- memorization probes;
- value of synthetic or newly collected data;
- whether data filtering removes useful diversity.

## In an interview

Use this order:

1. State the fixed resource and target metric.
2. Define the variables: parameters, tokens, and compute.
3. Explain why both model size and data matter.
4. Propose a small-run study with held-out validation.
5. Include stability, data quality, and serving cost.
6. State why extrapolation may fail.
7. Name the pilot that would reduce risk before the full run.

## Common mistakes

- Treating one published ratio as a universal law.
- Fitting a curve to too few runs.
- Changing model, data, and optimizer together.
- Ignoring failed runs.
- Optimizing pretraining loss while ignoring downstream behavior.
- Ignoring inference cost.
- Extrapolating far beyond the measured range without a pilot.

*Related: [train a 100B parameter model](/questions/train-100b-model/), [foundation-model data curation](/concepts/foundation-model-data-curation/), and [design an ML system under a fixed budget](/questions/design-ml-system-fixed-budget/).*