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
	<p class="visual-title" id="scaling-allocation-title">At fixed compute, more parameters leave fewer training tokens</p><div class="visual-argument" data-visual-argument="scaling-fixed-compute-allocation" data-argument-kind="curve"><p class="visual-argument-premise">At fixed compute, parameters compete with training tokens</p><svg viewBox="0 0 360 345" role="img" aria-label="At fixed compute, parameters compete with training tokens. In the simplified C proportional to N times D model, doubling parameters halves the token budget. This is a constraint curve, not an optimality claim."><title>At fixed compute, parameters compete with training tokens</title><desc>In the simplified C proportional to N times D model, doubling parameters halves the token budget. This is a constraint curve, not an optimality claim.</desc><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">1</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">8</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">8</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0</tspan></text><path data-proof-series="Fixed N x D = 8" d="M48 50L52.40625 66.7605633802817L56.8125 80.51282051282053L61.21875 92L65.625 101.73913043478261L70.03125 110.10101010101009L74.4375 117.35849056603774L78.84375 123.71681415929204L83.25 129.33333333333331L87.65625 134.3307086614173L92.0625 138.80597014925374L96.46875 142.83687943262413L100.875 146.48648648648648L105.28125 149.80645161290323L109.6875 152.8395061728395L114.09375 155.6213017751479L118.5 158.1818181818182L122.90625 160.54644808743168L127.3125 162.73684210526315L131.71875 164.7715736040609L136.125 166.66666666666666L140.53125 168.43601895734596L144.9375 170.09174311926606L149.34375 171.64444444444445L153.75 173.10344827586206L158.15625 174.47698744769875L162.5625 175.77235772357722L166.96875 176.99604743083003L171.375 178.15384615384616L175.78125 179.250936329588L180.1875 180.2919708029197L184.59375 181.28113879003558L189 182.22222222222223L193.40625 183.1186440677966L197.8125 183.97350993377484L202.21875 184.789644012945L206.625 185.56962025316454L211.03125 186.31578947368422L215.4375 187.03030303030303L219.84375 187.71513353115728L224.25 188.37209302325581L228.65625 189.002849002849L233.0625 189.60893854748605L237.46875 190.1917808219178L241.875 190.752688172043L246.28125 191.29287598944592L250.6875 191.8134715025907L255.09375 192.31552162849871L259.5 192.8L263.90625 193.26781326781327L268.3125 193.71980676328502L272.71875 194.15676959619952L277.125 194.57943925233644L281.53125 194.98850574712645L285.9375 195.3846153846154L290.34375 195.7683741648107L294.75 196.140350877193L299.15625 196.5010799136069L303.5625 196.85106382978722L307.96875 197.19077568134173L312.375 197.5206611570248L316.78125 197.84114052953157L321.1875 198.15261044176708L325.59375 198.45544554455446L330 198.75" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5"></path><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">relative parameter count N</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">relative token budget D</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Fixed N x D = 8</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> move right while holding training compute fixed: parameter count rises, so token count must fall. Too far left, insufficient capacity dominates loss; too far right, insufficient training data dominates. Fit the interior minimum for the current model, data, and recipe rather than copying a universal ratio. This original qualitative curve was checked against <a href="https://arxiv.org/abs/2001.08361">Kaplan et al.'s scaling-law study</a> and <a href="https://arxiv.org/abs/2203.15556">Hoffmann et al.'s compute-optimal study</a>; it reproduces no paper data or figure.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="scaling-fixed-compute-allocation"><strong>Check:</strong> When N increases at fixed C, what must happen to D? Why can that eventually worsen predicted loss?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="scaling-fixed-compute-allocation">In the simplified C proportional to N times D model, doubling parameters halves the token budget. This is a constraint curve, not an optimality claim.</span></figcaption>
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