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

<!-- visual:scaling-fixed-compute-allocation -->
<figure class="learning-figure plot-panel" aria-labelledby="scaling-allocation-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="scaling-allocation-title">Why does one fixed compute budget have an interior best allocation?</p>
	<svg viewBox="0 0 360 300" role="img" aria-labelledby="scaling-allocation-svg-title scaling-allocation-svg-desc">
		<title id="scaling-allocation-svg-title">Validation loss across parameter and token allocations at fixed compute</title>
		<desc id="scaling-allocation-svg-desc">A U-shaped qualitative curve shows validation loss falling as allocation moves away from a small capacity-limited model trained on many tokens, reaching a compute-optimal allocation in the middle, then rising when a large model receives too few training tokens and is undertrained. The horizontal axis keeps compute approximately equal to a constant times parameters times tokens.</desc>
		<rect class="viz-plot-bg" x="43" y="24" width="287" height="198" rx="3"></rect>
		<path class="viz-gridline" d="M43 73H330 M43 122H330 M43 171H330"></path>
		<path class="viz-axis" d="M43 24V222H330"></path>
		<path class="viz-roc-curve" d="M55 52 C82 132 116 190 180 190 C244 190 283 133 324 52"></path>
		<path class="viz-operating-guide" d="M180 190V222"></path>
		<circle class="viz-operating-point" cx="180" cy="190" r="5"></circle>
		<text class="viz-callout" x="59" y="47">capacity-limited</text>
		<text class="viz-label" x="59" y="62">too few parameters</text>
		<text class="viz-callout" x="238" y="47">undertrained</text>
		<text class="viz-label" x="238" y="62">too few tokens</text>
		<text class="viz-callout" x="180" y="180" text-anchor="middle">lowest predicted loss</text>
		<text class="viz-label" x="180" y="206" text-anchor="middle">compute-optimal allocation</text>
		<text class="viz-axis-label" x="43" y="244">smaller N / more tokens</text>
		<text class="viz-axis-label" x="330" y="244" text-anchor="end">larger N / fewer tokens</text>
		<text class="viz-label" x="186" y="269" text-anchor="middle">allocation along fixed C ~ k N D</text>
		<text class="viz-axis-label" transform="translate(15 151) rotate(-90)">validation loss</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> move right while holding training compute fixed: parameter count rises, so token count must fall. Too far left, insufficient capacity dominates loss; too far right, insufficient training data dominates. Fit the interior minimum for the current model, data, and recipe rather than copying a universal ratio. This original qualitative curve was checked against <a href="https://arxiv.org/abs/2001.08361">Kaplan et al.'s scaling-law study</a> and <a href="https://arxiv.org/abs/2203.15556">Hoffmann et al.'s compute-optimal study</a>; it reproduces no paper data or figure.</figcaption>
</figure>

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