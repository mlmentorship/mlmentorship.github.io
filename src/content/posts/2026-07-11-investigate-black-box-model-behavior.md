---
title: "Investigate a black-box model behavior in 90 minutes"
description: "Turn an observation into competing hypotheses, discriminating probes, boundary conditions, and one decision-changing result."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

Investigate why an undocumented model follows conflicting content using controlled queries, competing predictions, and a fixed investigation budget.

## Problem

**Context.** An undocumented API receives a user request for `BLUE` and conflicting content asking for `RED`.
The content's source, wording, order, and simulated consequence cues can vary.

**Task.** Design a query ledger within 90 minutes and 32 queries, test competing explanations, and report supported behavioral conclusions.
State which next observation would discriminate remaining hypotheses.

**Constraints.** Execute no external actions, including in simulated high-risk conditions.
Control query changes and uncertainty; observed outputs do not establish hidden architecture, training data, or internal reasoning.

You have an undocumented model API, 90 minutes and a ceiling of 32 queries.
The user requests the exact token `BLUE`; conflicting data requests `RED`.
Investigate whether following the conflict depends on its source rather than wording, simulated risk cues or order.
Execute no external actions, even in a condition labeled high risk.

## Begin with a behavioral claim

Bound the claim to model version, tested request family, tool schema and decoding settings.
Do not turn “this source contrast changes responses” into “we found an internal priority module.”
Repeated samples of one request are not independent tasks or a production prevalence estimate.

## Build competing hypotheses

Use deliberately simplified accounts:

- **H1, source:** conflict in tool text predicts `RED`, unlike the tested context placement.
- **H2, wording:** the exact trigger predicts `RED`, but its tested paraphrase does not.
- **H3, risk cue:** the low-risk label predicts `RED`, unlike the simulated high-risk label.
- **H4, recency:** conflict last predicts `RED`, unlike user instruction last.

These accounts can coexist; real hypotheses need not have these cleanly separated predictions.
Write predictions before inspecting outputs.

## Picture

The original setup combines tool source, exact wording, low-risk cue and late placement.
All four idealized accounts predict `RED`, so another identical success does little to separate them.

<!-- visual:black-box-discriminating-probes -->
<figure class="learning-figure" aria-labelledby="black-box-probes-title">
	<p class="visual-kicker">Predictions before observations</p>
	<p class="visual-title" id="black-box-probes-title">One RED response fits four accounts; choose a matched contrast.</p>
	<div class="visual-grid--two" role="group" aria-label="Comparison of an ambiguous observation and four matched discriminating probes">
		<section class="visual-panel" aria-labelledby="black-box-ambiguous-title">
			<h4 id="black-box-ambiguous-title">One observation fits every story</h4>
			<p>The exact conflict appears last in tool output with a simulated low-risk cue. The model returns RED.</p>
			<table class="cm-grid" aria-label="Four hypotheses that all predict the observed behavior">
				<thead><tr><th scope="col">hypothesis</th><th scope="col">prediction</th></tr></thead>
				<tbody>
					<tr><th scope="row">H1 source priority</th><td>follow conflict</td></tr>
					<tr><th scope="row">H2 lexical trigger</th><td>follow conflict</td></tr>
					<tr><th scope="row">H3 risk gate</th><td>follow conflict</td></tr>
					<tr><th scope="row">H4 recency</th><td>follow conflict</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">result: 4 explanations remain compatible</p>
		</section>
		<section class="visual-panel" aria-labelledby="black-box-discriminate-title">
			<h4 id="black-box-discriminate-title">Change one factor; predict before querying</h4>
			<p>Hold other factors fixed where the API permits. A switch matches one simplified prediction, not a unique internal mechanism.</p>
			<table class="cm-grid" aria-label="One-factor probes and idealized predictions when the response switches to BLUE">
				<thead><tr><th scope="col">one change</th><th scope="col">switch matches</th><th scope="col">idealized rivals</th></tr></thead>
				<tbody>
					<tr><th scope="row">tool to context, still last</th><td class="cm-selected"><strong>H1</strong>stop</td><td>follow</td></tr>
					<tr><th scope="row">paraphrase trigger</th><td class="cm-selected"><strong>H2</strong>stop</td><td>follow</td></tr>
					<tr><th scope="row">low to high risk cue</th><td class="cm-selected"><strong>H3</strong>stop</td><td>follow</td></tr>
					<tr><th scope="row">conflict first, user last</th><td class="cm-selected"><strong>H4</strong>stop</td><td>follow</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">Select a useful split within the query budget.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> change source, wording, risk cue or order while recording unavoidable serialization changes. A matched contrast tests a prediction; four samples do not establish a hidden mechanism.</figcaption>
</figure>

<p class="diagram-source">Method references: <a href="https://doi.org/10.1126/science.146.3642.347">Platt, Strong Inference</a> and <a href="https://aclanthology.org/2020.acl-main.442/">CheckList behavioral testing</a>.</p>

## Read an invented result ledger

Score exact `RED`, exact `BLUE` and other output separately.
The following counts are invented for teaching; they are not results from a queried model.
Each cell receives four samples; assume all non-RED responses in this fixture are `BLUE`.

| Condition | Intended change | RED / samples |
| --- | --- | --- |
| Baseline | Tool, exact phrase, low-risk cue, conflict last | $4/4$ |
| No conflict | Remove conflicting request | $0/4$ |
| Source swap | Context placement, still last | $0/4$ |
| Paraphrase | Change phrase, retain source and order | $4/4$ |
| Risk cue | Change only simulated risk label | $4/4$ |
| Order reversal | Conflict first, user last | $4/4$ |

This pattern favors H1's tested behavioral prediction, with substantial uncertainty.
It does not establish source priority inside the model or risk of deployed actions.
Inspect serialized messages: API role restrictions may prevent a clean source or order intervention.

## Keep the query budget explicit

Initial allocation uses $6\times4=24$ calls.
Reserve eight calls for a four-cell source-by-order pilot, two samples per cell.
Stop at 32 total; replacing that pilot with residual tests must not add uncounted calls.
Log actual request/output tokens and priced cost.
At caps of 512 input and eight output tokens per call, the ceilings are 16384 input and 256 output tokens.
Count the complete serialized request against the input cap.

Freeze model, prompts, tools, rubric and decoding; declare seed policy.
Same-seed repeats need not provide independent draws.
Retain refusals, errors and negative outcomes rather than selecting only successful conflicts.

## Change the tested factor or allocation

### 1. Cross source with order

**New task:** source and recency both explain the baseline.

**Picture and change:** cross tool/context placement with early/late conflict using the reserved four cells.
Keep semantics and scoring fixed; document serialization differences.

**Check:** a source effect at both orders supports H1's behavioral prediction.
An effect only when last requires an interaction account.

**Use when:** competing accounts still predict the same observed result.

### 2. Investigate residuals

**New task:** the leading account leaves 10% unexplained in a larger investigation.

**Picture and change:** allocate reserved calls to exact residual reruns and matched residual-feature contrasts.
Inspect severity and production reachability, not only frequency.

**Check:** residuals appearing only after a model-version change require separate version analysis.
Do not dismiss the tail as noise or spend the same reserve twice.

**Use when:** reproducible residuals could change the decision.

## Failure and transfer

A hundred successful paraphrases from one family establish neither production prevalence nor an internal risk classifier.

What if paraphrasing stops the conflict but source swapping does not?

<details>
<summary>Check your answer</summary>

H2's wording account gains support.
Test new paraphrases and source-by-phrase interactions before claiming a mechanism.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Keep an experiment ledger

For each probe, record:

| Field | Purpose |
| --- | --- |
| Hypothesis | Which explanation does this test? |
| Manipulation | What changed from control? |
| Predicted outcomes | What does each hypothesis expect? |
| Observation | Exact decision, score, or output |
| Update | Which beliefs moved, and how? |
| Next probe | Highest-information follow-up |

Separate exact scoring from model-graded judgments; an unavailable probability is not a license to invent confidence.

<span id="what-an-l4-answer-sounds-like"></span>

## An unsupported generalization

A prompt dump without a denominator, control or boundary cannot support a broad prevalence claim.

<span id="what-an-l5-answer-adds"></span>

## A five-minute readout

Present claim, experiment matrix, decision-changing result, uncertainty and next test.
Show the evidence needed for the conclusion rather than every prompt.

<span id="what-an-l6-answer-adds"></span>

## Decision and evaluation limits

Connect findings to a product surface, threat model or evaluation gap without treating the toy fixture as launch evidence.
A model grader can share the target's blind spot.
Safety changes can regress usefulness; static prompts can be overfit.
Nondeterminism can mimic interactions, and version changes can invalidate comparisons.

<span id="tells-that-get-you-a-strong-hire-vote"></span>

## Inspectable evidence

Bound claims, preserve negative results and choose controls before interactions.
No presentation format guarantees a hiring outcome.

<span id="tells-that-get-you-down-leveled"></span>

## Avoid overclaiming

Do not select only successes, conflate correlated prompts with independent tasks, or treat a model judge as ground truth.

## Common follow-up

"Your leading hypothesis explains 90% of observations. What do you do with the other 10%?"

Check noise, labels and missing conditions before averaging away the residual.

Use the [black-box behavior lab](/prep/labs/black-box/) for an executable attempt.

*Related: [design an ablation study](/questions/design-ablation-study/), [critique an ML paper](/questions/critique-ml-paper/), and [chain-of-thought monitorability](/concepts/chain-of-thought-monitorability/).*

</div>
