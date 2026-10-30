---
title: "Evaluation validity and benchmark contamination"
description: "Check that an evaluation measures the claimed ability, uses independent data, and still predicts performance on the target task."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

An evaluation is valid when its score supports the decision or claim being made. Benchmark contamination occurs when test content, answers, or close variants influence training or model selection.

## Why AI labs care

A model can improve on a benchmark without improving the intended ability.

Possible causes include:

- test examples appeared in training data;
- the team tuned prompts or checkpoints on the test set;
- a format change made the test easier;
- the grader rewards style instead of correctness;
- the benchmark covers only a narrow part of the task;
- the score no longer predicts production or research outcomes.

A high score is useful only when the evaluation supports the target claim.

## Start from the claim

Write the claim before choosing the metric.

Examples:

- "The model can solve new algebra problems."
- "The coding agent can repair unfamiliar repositories."
- "The assistant refuses unsafe requests without blocking normal help."

Each claim implies a data distribution, allowed tools, failure cost, and level of generalization.

A test of memorized textbook questions cannot support a broad claim about new problem solving.

## Main forms of validity

### Task validity

Does the test contain the work named in the claim? A code benchmark that measures only final unit tests may miss unsafe edits, weakened tests, or excessive patch scope.

### Population validity

Do test users, languages, domains, and difficulty levels match the target use? An English-only average does not support a multilingual claim.

### Grader validity

Does the scoring rule reward the intended behavior? A model judge may reward length or polished style. A unit test may miss an important invariant.

### Predictive validity

Does the offline score predict later human, production, or research outcomes? Track this relation over time.

## How contamination happens

### Training contamination

Exact test items, answers, source documents, or generated variants enter pretraining or fine-tuning data.

### Selection contamination

The team repeatedly checks the same test set while choosing data, prompts, checkpoints, and methods. The test set becomes part of development.

### Grader contamination

A model or policy learns details of the grader. It may exploit those details without learning the target ability.

### Source overlap

Train and test items come from the same repository, author, template, benchmark family, or generated source. Exact deduplication may miss this link.

**Learning objective:** distinguish an evaluation that provides independent confirmation from one that has become part of development through repeated feedback.

<!-- visual:independent-confirmation-vs-adaptive-reuse -->
<figure class="learning-figure editorial-illustration" aria-labelledby="independent-confirmation-vs-adaptive-reuse-edition-title"><span class="visual-fragment-anchor" id="evaluation-role-title" aria-hidden="true"></span><p class="visual-title" id="independent-confirmation-vs-adaptive-reuse-edition-title">Using benchmark feedback to choose a model turns it into development evidence</p><div class="visual-argument" data-visual-argument="independent-confirmation-vs-adaptive-reuse" data-argument-kind="timeline"><p class="visual-argument-premise">A benchmark becomes development evidence when it guides selection</p><svg viewBox="0 0 360 439" role="img" aria-label="A benchmark becomes development evidence when it guides selection. The first feedback influences the chosen model. Independent confirmation requires new evidence after that selection is locked."><title>A benchmark becomes development evidence when it guides selection</title><desc>The first feedback influences the chosen model. Independent confirmation requires new evidence after that selection is locked.</desc><text x="20" y="25" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Benchmark feedback observed</tspan></text><path d="M30 63L330 63" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="30" cy="63" r="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="30" y="92" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="30" dy="0">0</tspan></text><path d="M230 50L230 79" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="122" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Model chosen using that feedback</tspan></text><path d="M30 160L330 160" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="130" cy="160" r="5" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="130" y="189" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="130" dy="0">1</tspan></text><path d="M230 147L230 176" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="219" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Model selection locked</tspan></text><path d="M30 257L330 257" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="230" cy="257" r="5" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.75"></circle><text x="230" y="286" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-state-stroke)"><tspan x="230" dy="0">2</tspan></text><path d="M230 244L230 273" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="316" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Fresh confirmation evidence collected</tspan></text><path d="M30 354L330 354" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="330" cy="354" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="330" y="383" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="330" dy="0">3</tspan></text><path d="M230 341L230 370" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="180" y="413" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="180" dy="0">event order, not elapsed time</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 490" role="img" aria-labelledby="independent-confirmation-vs-adaptive-reuse-edition-svg-title independent-confirmation-vs-adaptive-reuse-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="independent-confirmation-vs-adaptive-reuse-edition-svg-title">A feedback arrow changes the benchmark's role</title><desc id="independent-confirmation-vs-adaptive-reuse-edition-svg-desc">Public benchmark B feeds model selection and therefore becomes development evidence. In a separate lane, M2 freezes before hidden fresh set H is opened once. There is deliberately no return arrow from H into that already-selected model. Later selection would require new confirmation.</desc><text x="90" y="24" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">DEVELOPMENT</tspan></text><text x="270" y="24" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">CONFIRMATION</tspan></text><g data-editorial-node><rect x="18" y="63" width="144" height="78" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="88" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">Public set B</tspan><tspan x="90" dy="19">score + errors</tspan><tspan x="90" dy="19">visible to team</tspan></text></g><path d="M90 141L90 183" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M86.64402122977057,176.8569220667674L90,183L93.35597877022943,176.8569220667674" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="18" y="187" width="144" height="78" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="90" y="212" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">Select revised</tspan><tspan x="90" dy="19">M2</tspan><tspan x="90" dy="19">using B feedback</tspan></text></g><path d="M18 226L7 226" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M7 226L7 102" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M7 102L18 102" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M11.85692206676739,105.35597877022943L18,102L11.85692206676739,98.64402122977057" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><text x="90" y="312" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">B now shapes M2</tspan></text><g data-editorial-node><rect x="198" y="63" width="144" height="78" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="270" y="88" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">Freeze selected</tspan><tspan x="270" dy="19">M2</tspan><tspan x="270" dy="19">before opening H</tspan></text></g><path d="M270 141L270 183" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M266.6440212297706,176.8569220667674L270,183L273.3559787702294,176.8569220667674" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="198" y="187" width="144" height="78" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="212" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">Fresh hidden H</tspan><tspan x="270" dy="19">score once</tspan><tspan x="270" dy="19">record scope</tspan></text></g><text x="270" y="312" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">No return arrow</tspan><tspan x="270" dy="19">into this selection</tspan></text><path d="M180 45L180 359" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><text x="180" y="401" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Revising M2 after H uses H as development</tspan><tspan x="180" dy="19">evidence.</tspan></text><text x="180" y="449" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Confirm the revised system on another</tspan><tspan x="180" dy="19">independent source.</tspan></text></svg><div class="visual-derivation-caption">The loop on B makes it development feedback. The one-way H path confirms only the frozen system and tested scope. Further selection needs another independent evaluation.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="independent-confirmation-vs-adaptive-reuse"><strong>Check:</strong> When does the benchmark become development data rather than independent confirmation?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="independent-confirmation-vs-adaptive-reuse">The first feedback influences the chosen model. Independent confirmation requires new evidence after that selection is locked.</span></figcaption></figure>

## Reduce contamination risk

1. Track source and transformation history for training and evaluation data.
2. Remove exact and near duplicates.
3. Split by source, time, repository, author, or task family where needed.
4. Keep a final set hidden from model and prompt development.
5. Use fresh examples after major tuning cycles.
6. Record every use of a test set.
7. Check suspicious gains on an independent evaluation.
8. Report known overlap instead of claiming perfect removal.

No decontamination method proves that a model has never seen related knowledge.

## Benchmark saturation

A benchmark is saturated when many systems score near its ceiling or when common development choices have been tuned to it.

Signs:

- small score changes depend on formatting;
- rankings change across prompt templates;
- most examples are easy for all models;
- errors cluster in ambiguous or broken items;
- benchmark gains do not move human or production outcomes.

Keep useful old tests for regression. Add harder and more representative tests for model selection.

## Small example

A coding agent improves pass rate on a public benchmark.

Before claiming better repository repair:

1. Check whether benchmark repositories appeared in training data.
2. Split evaluation by repository age and source.
3. Review whether the agent changed or removed tests.
4. Measure patch scope and regression failures.
5. Test private tasks from different repositories.
6. Ask humans to review a sample of passing patches.

If the gain disappears on new repositories, the broad claim is not supported.

## Adaptive evaluation

As models improve, evaluators add harder tasks. This keeps the test useful and introduces a risk: task writers may choose examples based on known model failures.

Keep separate sets for:

- stable regression tracking;
- current frontier comparison;
- final independent confirmation.

Version the evaluator and state which version supports each result.

## In an interview

Use this order:

1. State the exact claim and decision.
2. Define the target population and task.
3. Explain the grader and its failure modes.
4. Check training, selection, source, and grader contamination.
5. Use held-out task families or sources.
6. Confirm important gains on fresh data.
7. Track whether offline scores predict the real outcome.
8. Report limits on the claim.

## Common mistakes

- Treating a public benchmark as an independent test after repeated tuning.
- Using exact deduplication as proof of no contamination.
- Reporting one aggregate score.
- Ignoring broken or ambiguous benchmark items.
- Changing the prompt and calling the result a model gain.
- Letting the model under test grade itself without validation.
- Claiming general ability from one narrow task family.

*Related: [foundation-model data curation](/concepts/foundation-model-data-curation/), [LLM-as-judge evaluation](/concepts/llm-as-judge/), and [critique an ML paper](/questions/critique-ml-paper/).*