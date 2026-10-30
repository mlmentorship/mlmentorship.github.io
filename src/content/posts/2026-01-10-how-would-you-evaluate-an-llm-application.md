---
title: "How would you evaluate an LLM application you've built?"
description: "Define task success, combine hard checks with supported-answer review, and make an evidence-limited release decision."
date: "2026-01-10"
draft: false
tags: ["questions"]
category: "questions"
---


Define an LLM application's release decision using task success, unacceptable failures, cost, and representative evidence.

## Problem

**Context.** A document-answering application is assessed on held-out questions for valid citations and substantive source support.
The invented counts and a predeclared joint-quality floor appear below.

**Task.** Report each quality dimension, their intersection, cost, uncertainty, and the gate decision.
Describe follow-up evaluation and online evidence needed before claiming product improvement.

**Constraints.** Both quality requirements must hold on the same answer.
Do not substitute citation validity for factual support or treat a small observed pass rate as a population guarantee.

For 20 hypothetical held-out document questions, 18 answers cite valid IDs and 14 have independently human-reviewed substantive support.
Thirteen satisfy both requirements; total cost is $0.40.
The predeclared observed joint-quality floor is 70%.
What should the report and release decision say?

## Picture

Citation syntax and semantic support are different checks; an answer must pass both.
Five answers pass citation validation alone, one passes support alone, and one passes neither.

<p class="visual-kicker">Separate checks, then combine requirements</p>
<p class="visual-title">Eighteen valid citation IDs and 14 supported answers yield 13 jointly valid answers.</p>

<!-- visual:llm-eval-calibration-loop -->
<figure class="learning-figure" aria-labelledby="joint-eval-visual-title">
	<h3 class="visual-title" id="joint-eval-visual-title">Thirteen joint passes out of twenty miss the 70% release floor</h3><div class="visual-argument" data-visual-argument="llm-eval-calibration-loop" data-argument-kind="population"><p class="visual-argument-premise">Thirteen joint passes do not clear a fourteen-pass floor</p><svg viewBox="0 0 360 204" role="img" aria-label="Thirteen joint passes do not clear a fourteen-pass floor. Thirteen of twenty trials pass both requirements, or 65%. A 70% floor requires at least fourteen joint passes."><title>Thirteen joint passes do not clear a fourteen-pass floor</title><desc>Thirteen of twenty trials pass both requirements, or 65%. A 70% floor requires at least fourteen joint passes.</desc><text x="20" y="25" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">One mark = one evaluation trial</tspan></text><g data-proof-population="joint passes"><circle cx="24" cy="53" r="2" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-proof-population="joint passes"><circle cx="32" cy="53" r="2" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-proof-population="joint passes"><circle cx="40" cy="53" r="2" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-proof-population="joint passes"><circle cx="48" cy="53" r="2" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-proof-population="joint passes"><circle cx="56" cy="53" r="2" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-proof-population="joint passes"><circle cx="64" cy="53" r="2" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-proof-population="joint passes"><circle cx="72" cy="53" r="2" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-proof-population="joint passes"><circle cx="80" cy="53" r="2" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-proof-population="joint passes"><circle cx="88" cy="53" r="2" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-proof-population="joint passes"><circle cx="96" cy="53" r="2" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-proof-population="joint passes"><circle cx="104" cy="53" r="2" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-proof-population="joint passes"><circle cx="112" cy="53" r="2" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-proof-population="joint passes"><circle cx="120" cy="53" r="2" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-proof-population="joint failures"><path d="M126 51l4 4m-4 0l4 -4" style="stroke:var(--viz-warning-stroke);stroke-width:1.2"></path></g><g data-proof-population="joint failures"><path d="M134 51l4 4m-4 0l4 -4" style="stroke:var(--viz-warning-stroke);stroke-width:1.2"></path></g><g data-proof-population="joint failures"><path d="M142 51l4 4m-4 0l4 -4" style="stroke:var(--viz-warning-stroke);stroke-width:1.2"></path></g><g data-proof-population="joint failures"><path d="M150 51l4 4m-4 0l4 -4" style="stroke:var(--viz-warning-stroke);stroke-width:1.2"></path></g><g data-proof-population="joint failures"><path d="M158 51l4 4m-4 0l4 -4" style="stroke:var(--viz-warning-stroke);stroke-width:1.2"></path></g><g data-proof-population="joint failures"><path d="M166 51l4 4m-4 0l4 -4" style="stroke:var(--viz-warning-stroke);stroke-width:1.2"></path></g><g data-proof-population="joint failures"><path d="M174 51l4 4m-4 0l4 -4" style="stroke:var(--viz-warning-stroke);stroke-width:1.2"></path></g><text x="20" y="96" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">13 joint passes</tspan></text><text x="20" y="131" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">7 joint failures</tspan></text><text x="20" y="172" style="font-size:22px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">13 / 20 = 65.0%</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 600 540" role="img" aria-labelledby="joint-eval-chart-title joint-eval-chart-desc" style="width: 100%; min-width: 0; height: auto;">
		<title id="joint-eval-chart-title">Thirteen joint passes fail the seventy-percent release floor</title>
		<desc id="joint-eval-chart-desc">Each of twenty tiles represents one hypothetical answer, grouped in one possible ordering. Thirteen tiles marked VS pass citation validity and support. Five marked V pass citation validity only, one marked S passes support only, and one dash passes neither. Thus eighteen pass validity, fourteen pass support, but only thirteen of twenty, or 65%, pass both. The joint-rate bar stops before the required 70% marker. Hold this candidate; the existing production version remains unchanged. Repair on development cases and use fresh independent confirmation.</desc>
		<g fill="var(--c-text)" font-family="var(--font-ui)" font-size="24">
			<text x="40" y="30" font-weight="650">20 hypothetical answers; one tile each</text>
			<g data-eval-status="both" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)">
				<rect x="40" y="68" width="44" height="42"/><rect x="92" y="68" width="44" height="42"/><rect x="144" y="68" width="44" height="42"/><rect x="196" y="68" width="44" height="42"/><rect x="248" y="68" width="44" height="42"/>
				<rect x="300" y="68" width="44" height="42"/><rect x="352" y="68" width="44" height="42"/><rect x="404" y="68" width="44" height="42"/><rect x="456" y="68" width="44" height="42"/><rect x="508" y="68" width="44" height="42"/>
				<rect x="40" y="122" width="44" height="42"/><rect x="92" y="122" width="44" height="42"/><rect x="144" y="122" width="44" height="42"/>
			</g>
			<g data-eval-status="citation-only" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)"><rect x="196" y="122" width="44" height="42"/><rect x="248" y="122" width="44" height="42"/><rect x="300" y="122" width="44" height="42"/><rect x="352" y="122" width="44" height="42"/><rect x="404" y="122" width="44" height="42"/></g>
			<g data-eval-status="support-only" fill="var(--viz-state-bg)" stroke="var(--viz-state-stroke)"><rect x="456" y="122" width="44" height="42"/></g>
			<g data-eval-status="neither" fill="var(--viz-warning-bg)" stroke="var(--viz-warning-stroke)"><rect x="508" y="122" width="44" height="42"/></g>
			<g font-size="20" text-anchor="middle">
				<text x="62" y="95">VS</text><text x="114" y="95">VS</text><text x="166" y="95">VS</text><text x="218" y="95">VS</text><text x="270" y="95">VS</text><text x="322" y="95">VS</text><text x="374" y="95">VS</text><text x="426" y="95">VS</text><text x="478" y="95">VS</text><text x="530" y="95">VS</text>
				<text x="62" y="149">VS</text><text x="114" y="149">VS</text><text x="166" y="149">VS</text><text x="218" y="149">V</text><text x="270" y="149">V</text><text x="322" y="149">V</text><text x="374" y="149">V</text><text x="426" y="149">V</text><text x="478" y="149">S</text><text x="530" y="149">-</text>
			</g>
			<text x="40" y="207" font-size="22">VS: 13 pass both</text><text x="320" y="207" font-size="22">V: 5 citation only</text>
			<text x="40" y="241" font-size="22">S: 1 support only</text><text x="320" y="241" font-size="22">-: 1 passes neither</text>
			<text x="40" y="282" font-weight="650">Joint pass rate</text>
			<rect x="40" y="304" width="520" height="32" fill="var(--viz-neutral-bg)" stroke="var(--viz-edge)"/>
			<rect data-eval-passes="13" data-eval-total="20" x="40" y="304" width="338" height="32" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)"/>
			<path data-eval-floor="0.7" d="M404 292V352" stroke="var(--viz-warning-stroke)" stroke-width="3"/>
			<text x="378" y="374" text-anchor="end">65% observed</text><text x="404" y="402" font-size="22">70% floor</text>
			<text data-eval-decision="hold" x="40" y="446" font-size="28" font-weight="650">HOLD this candidate</text>
			<text x="40" y="482" font-size="22">Existing production stays unchanged.</text>
			<text x="40" y="518" font-size="22">Repair, then confirm on fresh cases.</text>
		</g>
	</svg>
	<div class="visual-derivation-caption">count answers passing both requirements, not their average or union. Thirteen of twenty fails the 70% observed floor. Monitoring the existing production version does not authorize releasing this candidate.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="llm-eval-calibration-loop"><strong>Check:</strong> Does the illustrated candidate pass the stated gate? Which arrow shows what happens after it fails?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="llm-eval-calibration-loop">Thirteen of twenty trials pass both requirements, or 65%. A 70% floor requires at least fourteen joint passes.</span></figcaption>
</figure>
<p class="diagram-source">References: <a href="https://arxiv.org/abs/2306.05685">LLM-as-judge study</a>, <a href="https://developers.openai.com/api/docs/guides/evaluation-best-practices">OpenAI evaluation guidance</a>, and <a href="https://doi.org/10.6028/NIST.AI.100-1">NIST AI RMF</a>.</p>

## Report the joint outcome

| | Supported | Unsupported | Total |
| --- | --- | --- | --- |
| Valid citation | $13$ | $5$ | $18$ |
| Invalid citation | $1$ | $1$ | $2$ |
| Total | $14$ | $6$ | $20$ |

Citation validity is 90%; support is 70%; joint validity is $13/20=65\%$.
Their arithmetic mean, 80%, is not the required success rate.
The union is $18+14-13=19$, which also answers a different question.
Cost per jointly valid answer is $0.40/13\approx0.03077$ dollars.

The observed joint floor fails.
Report counts, sampling unit and an uncertainty method separately from this descriptive release rule.
Twenty cases do not establish population quality; passing this floor would not prove improved user task completion.

## Keep a case-level record

Record `case_id`, source/evidence version, slice, rubric, hard-check results, semantic labels, judge/model/prompt versions, latency and cost.
Assign dataset ownership, review disagreements and preserve labels.
Use deterministic checks for exact requirements and qualified humans to validate open-ended scoring.

## Change the output or severity

### 1. Require a structured record

**New task:** the answer must satisfy an exact schema and field constraints.

**Picture and change:** validate parseability, types and constraints deterministically before semantic grading.

**Check:** malformed output fails its hard gate despite supported prose.

**Use when:** the consumer depends on an exact output contract.

### 2. Increase failure severity

**New task:** answers concern medication instructions.

**Picture and change:** add expert review, harmful-error slices, abstention and a governed release process.

**Check:** routine successes cannot compensate for a prohibited harmful answer.

**Use when:** errors can create high-impact harm; this example provides no domain validation.

## Failure and transfer

After tuning prompts on these final cases, later scores become development evidence, not independent confirmation.
Keep fresh confirmation data separate from rubric repair.

Is easy-question judge agreement enough when unsupported legal claims are missed?

<details>
<summary>Check your answer</summary>

No. Validate the consequential slice and route unreliable cases to qualified humans or abstention.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

<span id="what-an-l4-answer-sounds-like"></span>

## A metric list is incomplete

BLEU, ROUGE, METEOR and perplexity measure particular properties, not every product's task success.
Public benchmarks can supplement product-specific evidence, not replace it.

<span id="what-an-l5-answer-sounds-like"></span>

## Cases and scorer validation

Stratify representative traffic and known failure slices.
Size the set for coverage and uncertainty needs, not a universal 50–200-example rule.
Check scorer mistakes by slice, not agreement alone.
Chance-corrected agreement can complement, not replace, consequential-error analysis.

<span id="what-an-l6-answer-sounds-like"></span>

## Lifecycle and tradeoffs

Test judge sensitivity to position, verbosity, model family and domain gaps.
Pairwise comparisons can help; counterbalance order and validate their reliability.
Bradley–Terry ranking introduces assumptions, not guaranteed agreement.

Version scorer changes and retest against humans.
Measure quality, latency and cost jointly; inspect nondominated choices rather than optimizing a score alone.
Track user task completion, corrections or feedback with suitable denominators.
No universal offline-online correlation cutoff certifies an evaluation.

<span id="the-tells-that-get-you-a-strong-hire-vote"></span>

## Inspectable reasoning

State task, costly failures, evidence limits and the decision each measurement supports.
Use only truthful experience; no answer guarantees a hiring outcome.

<span id="the-tells-that-get-you-down-leveled"></span>

## Avoid unsupported certainty

A judge prompt is not calibration, and aggregate quality can hide consequential regressions.

<span id="what-the-interviewer-is-actually-checking"></span>

## Scope the answer

Explain what the system must do, how errors are checked and who owns evaluation changes.

*Read the long-form essay this distills: ["LLM Evals: The hardest part of shipping LLMs"](/guides/llm-evals-the-hardest-part/).*

</div>
