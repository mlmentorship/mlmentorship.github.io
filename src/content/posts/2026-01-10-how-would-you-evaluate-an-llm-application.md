---
title: "How would you evaluate an LLM application you've built?"
description: "Define task success, combine hard checks with supported-answer review, and make an evidence-limited release decision."
date: "2026-01-10"
draft: false
tags: ["questions"]
category: "questions"
---


Evaluate the user's task and unacceptable failures before choosing scorers, then connect offline evidence to product outcomes.

## Problem

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
```mermaid
flowchart TB
	accTitle: LLM application evaluation as a human-grounded calibration loop
	accDescr: The hypothetical fixture contains 20 held-out questions. Eighteen answers have valid citation IDs, fourteen have human-reviewed semantic support and thirteen pass both requirements. The joint observed score is 65 percent, below its predeclared 70 percent floor. Humans validate the rubric and scalable scorers. Cost, latency, uncertainty and consequential slices remain separate report fields. Production task outcomes test whether offline results predict useful behavior; failures trigger new development cases and independent confirmation, not silent reuse of the same final set.
	A["1 · Product contract<br/>valid citation AND supported answer"] --> B["2 · Version held-out cases<br/>20 questions + known slices"]
	B --> C["3 · Humans apply an explicit rubric<br/>resolve disagreement · preserve labels"]
	C --> D["4 · Validate scalable scorers<br/>closed tasks: deterministic checks<br/>open tasks: bias-tested LLM judge"]
	D --> E["5 · Joint gate<br/>13/20 = 65%; required 70%<br/>cost + latency + uncertainty"]
	E --> F["6 · Observe production<br/>task outcomes · guardrails · drift"]
	F --> G{"Do offline trends<br/>predict production?"}
	G ==>|"yes"| H["Continue monitoring<br/>keep the eval versioned"]
	G -. "no · drift or new failure" .-> I["Refresh cases + rubric<br/>recalibrate scorers"]
	I -. "rerun" .-> B
	class A,B viz-input
	class C,D viz-focus
	class E viz-state
	class F,H viz-output
	class G,I viz-warning
	class A viz-tall
```
<p class="diagram-caption"><strong>Read it this way:</strong> keep hard-check validity and human-reviewed support separate before applying their joint gate. Offline success still needs confirmation against the product outcome.</p>
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
