---
title: "What's the most over-rated technique in ML right now?"
description: "Critique a specific technical use with task evidence, cost, uncertainty and a useful boundary; state what would change the recommendation."
date: "2025-12-07"
draft: false
tags: ["questions"]
category: "questions"
---


Evaluate a claimed improvement from an ML technique by comparing its evidence, costs, and appropriate scope of use.

## Problem

**Context.** An interviewer asks which technique is overrated; the invented ledger below offers one specific use to examine.

**Task.** Explain the technique's mechanism, assess whether the observed improvement justifies its cost, and propose a deciding follow-up comparison.
Describe a useful boundary and evidence that could reverse your recommendation.

**Constraints.** Critique the claimed use rather than dismissing every use of the technique.
Aggregate correct counts alone do not establish paired statistical significance, causal product benefit, or a universal ranking.

This fictional ledger uses 1,000 matched held-out tasks:

| Setting | Observed correct | Counted tokens/request |
| --- | ---: | ---: |
| Baseline | 900 | 100 |
| Explicit-reasoning setting | 902 | 400 |

Does the extra cost earn its place? What evidence is missing before recommending deployment?

## Picture: Replace a Verdict With a Testable Claim

<!-- visual:overrated-technique-claim-stress-test -->
<figure class="learning-figure" aria-labelledby="overrated-technique-stress-test-title">
	<p class="visual-kicker">Bounded technical claim</p>
	<p class="visual-title" id="overrated-technique-stress-test-title">What turns a hot take into a defensible technical judgment?</p>
	<div class="visual-grid--two" role="group" aria-label="Side-by-side comparison of an unsupported hot take and a bounded, testable critique">
		<section class="visual-panel" aria-labelledby="hot-take-title">
			<h4 id="hot-take-title">Hot take: “CoT is overrated.”</h4>
			<p>A verdict without a scope or mechanism gives the interviewer nothing technical to test.</p>
			<table class="cm-grid" aria-label="Missing parts of the unsupported claim">
				<tbody>
					<tr><th scope="row">Target</th><td>Which use?</td></tr>
					<tr><th scope="row">Mechanism</th><td>Why?</td></tr>
					<tr><th scope="row">Boundary</th><td>When useful?</td></tr>
					<tr><th scope="row">Evidence</th><td>What changed?</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">Result: contrarian preference</p>
		</section>
		<section class="visual-panel" aria-labelledby="bounded-critique-title">
			<h4 id="bounded-critique-title">Bounded critique: generic CoT use</h4>
			<p>Make the claim narrower and testable. Preserve the case where the technique earns its cost.</p>
			<table class="cm-grid" aria-label="Four parts of a defensible critique">
				<tbody>
					<tr><th scope="row">Target</th><td class="cm-selected">Generic use on every task</td></tr>
					<tr><th scope="row">Mechanism</th><td>Hypothesis: longer generation may leave missing source facts unresolved</td></tr>
					<tr><th scope="row">Boundary</th><td>Test where intermediate computation is a demonstrated bottleneck</td></tr>
					<tr><th scope="row">Evidence</th><td>Pin task/model/settings; compare matched quality, latency and billed cost</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">Result: a claim experience can update</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> name the target, proposed mechanism, useful boundary and reversing evidence. The mechanism is a hypothesis to test, not a universal result about reasoning.</figcaption>
</figure>

## Work the Trade-off

Added counted tokens are $(400-100)\times1000=300{,}000$; observed net gain is $902-900=2$ correct answers.
The rates are 90.0% and 90.2%, a 0.2 percentage-point difference.

Those same totals permit either two or 198 discordant pairs.
You need paired outcomes and repeated model variability to assess uncertainty, not just aggregate counts.

Pin task, model, data, settings, quality target and resource budget; compare untouched cases and critical slices.
Visible tokens need not include hidden compute or all billed work. Prompt wording alone does not pin reasoning effort.
Do not request private hidden reasoning; public concise explanations and outcome measures suffice.

## Change 1: The Other Deployment Benefits

**New task.** The technique reliably helps another constrained task.

**Picture and change.** Inspect that quality/cost envelope and acknowledge where your critique does not apply.

**Check.** Updating is not losing the argument.

**Use when.** Evidence differs across deployments.

## Change 2: You Have Not Tested the Technique

**New task.** Personal evidence is limited.

**Picture and change.** Choose a better-supported topic or state a tentative hypothesis and a discriminating test.

**Check.** Unsupported certainty is not technical judgment. Do not invent personal experiments.

**Use when.** A fashionable topic exceeds your knowledge.

## Failure: Dismissal Without an Operating Point

Neither parameter counts nor a universal hybrid-search claim establishes a task-specific advantage.

## Transfer Check

Do two extra correct answers prove the expensive setting should ship?

### Reasoned Answer

No. Assess paired uncertainty, repetition, requirements, actual cost/latency and slice effects before deciding.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-sounds-like"></span>
<span id="what-an-l6-answer-adds"></span>
## Three Qualified Critiques

- **Reasoning allocation:** more generation may not address the actual retrieval/computation bottleneck. Test exact model/settings and task outcomes.
- **Distillation:** student architecture, objective and data can limit retained behavior. Parameter counts alone do not prove a task needs 70B parameters.
- **Synthetic data:** generator biases, selection and recursive replacement can cause failures. Stronger teachers/judges are not universal necessary conditions for usefulness.

Published examples have specific task/data/model conditions; they do not establish present deployment prevalence or universal boundaries.

<span id="tells-that-get-you-a-strong-hire-vote"></span>
## Claim Record

Record target/version, mechanism hypothesis, useful boundary, evidence provenance, uncertainty and reversal condition.
Separate your experiment, a paper's result and an untested hypothesis.

<span id="tells-that-get-you-down-leveled"></span>
## Missing Evidence

Qualify the unsupported claim instead of manufacturing confidence or an employer-specific expectation.

## Mirror question to be ready for

What is underrated? Possible topics, not universally winning answers:

- Evaluation engineering with demonstrated decision value.
- BM25/lexical retrieval when exact terms or hybrid comparisons justify it.
- Calibration when probability-based actions need reliable risk estimates.
- Distillation against a fixed product quality/cost target.

## Common follow-up

If the interviewer uses the technique, ask about their task, metric and constraints.
Revise the bounded claim when their evidence applies; do not turn the comparison into a contest of positions.

## Sources

- [Wei et al., chain-of-thought](https://arxiv.org/abs/2201.11903).
- [Hinton et al., distillation](https://arxiv.org/abs/1503.02531).
- [Shumailov et al., generated data](https://arxiv.org/abs/2305.17493).

---

*Related: [The 5 things every applied scientist interview is testing for](/guides/five-things-as-interview-tests/), [senior through senior-principal ML scope](/guides/l5-vs-l6-faang-ml/).*

</div>
