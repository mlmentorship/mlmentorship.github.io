---
title: "LLM-as-judge evaluation"
description: "Use a model to score open-ended outputs only after testing its rubric, bias, agreement, and resistance to manipulation."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

LLM-as-judge evaluation uses a language model to score, rank, or classify outputs from another model.

## Why AI labs care

Open-ended outputs often have no exact answer. Human review is useful and expensive. A model judge can score many examples quickly.

AI labs use model judges for:

- response quality;
- instruction following;
- factual support;
- safety behavior;
- agent trajectories;
- preference-data filtering;
- regression testing.

A judge is a learned measurement tool. It can be wrong in stable and predictable ways.

## Start with the easiest reliable grader

Use the most direct check available:

1. Executable tests for code.
2. Exact state checks for tool tasks.
3. Reference calculations for math.
4. Rule checks for required fields or forbidden actions.
5. Human or model judgment for meaning that cannot be checked directly.

Do not use a model judge for a condition that code can verify exactly.

**Learning objective:** distinguish an LLM judge's score from a validated evaluation decision by routing exact constraints to deterministic checks and uncertain semantic judgments through human calibration and review.

<!-- visual:judge-calibration-gates -->
<figure class="learning-figure editorial-illustration" aria-labelledby="judge-calibration-gates-edition-title"><span class="visual-fragment-anchor" id="judge-calibration-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="judge-exact-lane-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="judge-semantic-lane-title" aria-hidden="true"></span><p class="visual-title" id="judge-calibration-gates-edition-title">Hard failure vetoes quality; order-sensitive judgments need review</p><div class="visual-argument" data-visual-argument="judge-calibration-gates" data-argument-kind="matrix"><p class="visual-argument-premise">A hard failure vetoes quality; order sensitivity requires review</p><svg viewBox="0 0 360 145" role="img" aria-label="A hard failure vetoes quality; order sensitivity requires review. A high quality score cannot override a hard-rule failure. Reversing answer order and changing the judgment exposes a separate calibration problem."><title>A hard failure vetoes quality; order sensitivity requires review</title><desc>A high quality score cannot override a hard-rule failure. Reversing answer order and changing the judgment exposes a separate calibration problem.</desc><text x="140.71666666666667" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="140.71666666666667" dy="0">quality</tspan></text><text x="220.43" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="220.43" dy="0">hard rule</tspan></text><text x="300.1433333333333" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="300.1433333333333" dy="0">decision</tspan></text><text x="88.86" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="88.86" dy="0">violation</tspan></text><rect data-proof-cell="0,0" x="100.86" y="36" width="76.71333333333332" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="139.21666666666667" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="139.21666666666667" dy="0">.9</tspan></text><rect data-proof-cell="0,1" x="180.57333333333332" y="36" width="76.71333333333332" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="218.92999999999998" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="218.92999999999998" dy="0">FAIL</tspan></text><path d="M184.57333333333332 76L253.28666666666663 40" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><rect data-proof-cell="0,2" x="260.28666666666663" y="36" width="76.71333333333332" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="298.6433333333333" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="298.6433333333333" dy="0">hold</tspan></text><path d="M264.28666666666663 76L332.99999999999994 40" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="88.86" y="101" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="88.86" dy="0">order</tspan><tspan x="88.86" dy="18">flip</tspan></text><rect data-proof-cell="1,0" x="100.86" y="83" width="76.71333333333332" height="47" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="139.21666666666667" y="112" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="139.21666666666667" dy="0">.9</tspan></text><rect data-proof-cell="1,1" x="180.57333333333332" y="83" width="76.71333333333332" height="47" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="218.92999999999998" y="112" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="218.92999999999998" dy="0">pass</tspan></text><rect data-proof-cell="1,2" x="260.28666666666663" y="83" width="76.71333333333332" height="47" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="298.6433333333333" y="112" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="298.6433333333333" dy="0">review</tspan></text><path d="M264.28666666666663 126L332.99999999999994 87" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 491" role="img" aria-labelledby="judge-calibration-gates-edition-svg-title judge-calibration-gates-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="judge-calibration-gates-edition-svg-title">A hard failure and a position-sensitive judge remain separate failures</title><desc id="judge-calibration-gates-edition-svg-desc">Fictional response B lacks a required id while receiving quality 4.8 out of five. That score cannot override the deterministic failure. In an independent order-swap probe, the same response pair yields A when ordered AB and B when ordered BA, revealing first-position sensitivity and requiring semantic review.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional response pair A/B; fixed</tspan><tspan x="180" dy="19">independent rubric</tspan></text><text x="90" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">CHECK</tspan></text><text x="270" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">OBSERVED RESULT</tspan></text><path d="M15 91L345 91" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="111" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Deterministic output contract</tspan></text><g data-editorial-node><rect x="15" y="125" width="150" height="73" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="157" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">B: required id</tspan><tspan x="90" dy="19">absent</tspan></text></g><g data-editorial-node><rect x="195" y="125" width="150" height="73" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="147.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">HARD FAIL</tspan><tspan x="270" dy="19">quality 4.8</tspan><tspan x="270" dy="19">cannot veto</tspan></text></g><path d="M15 213L345 213" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="233" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Same pair; order swapped</tspan></text><g data-editorial-node><rect x="15" y="247" width="150" height="73" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="279" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">AB -&gt; prefers A</tspan><tspan x="90" dy="19">BA -&gt; prefers B</tspan></text></g><g data-editorial-node><rect x="195" y="247" width="150" height="73" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="269.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">first-position</tspan><tspan x="270" dy="19">bias</tspan><tspan x="270" dy="19">review the judge</tspan></text></g><path d="M15 335L345 335" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="355" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Independent human rubric</tspan></text><g data-editorial-node><rect x="15" y="369" width="150" height="92" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="420" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">human prefers A</tspan></text></g><g data-editorial-node><rect x="195" y="369" width="150" height="92" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="270" y="391.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">judge</tspan><tspan x="270" dy="19">disagreement</tspan><tspan x="270" dy="19">retain</tspan><tspan x="270" dy="19">separately</tspan></text></g></svg><div class="visual-derivation-caption">Neither high quality nor averaging repairs a missing required field. The swapped pair exposes a separate judge-instability problem. Preserve both outcomes and escalate the semantic disagreement.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="judge-calibration-gates"><strong>Check:</strong> Can a high judge quality score override the pictured hard failure? What observed instability sends the judgment to review?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="judge-calibration-gates">A high quality score cannot override a hard-rule failure. Reversing answer order and changing the judgment exposes a separate calibration problem.</span></figcaption></figure>

## Pointwise and pairwise judging

A pointwise judge scores one output on a scale. A pairwise judge chooses the better of two outputs.

Pairwise judging is often easier because the judge compares concrete alternatives. Pointwise scores are useful when a fixed quality threshold is needed.

For pairwise evaluation:

- randomize answer order;
- score both orderings on a sample;
- allow a tie when outputs are close;
- keep model identity hidden;
- fit a ranking only after checking judge consistency.

## Write a clear rubric

A rubric should define separate criteria. For a factual support task, it might ask:

- Does every claim answer the question?
- Is every claim supported by the given source?
- Does the output add facts that are not in the source?
- Does it follow the required format?

Avoid one broad request such as "score overall quality." The judge may replace the intended task with its own style preference.

Hard constraints should be separate from soft quality. An agent that completes a task through an unauthorized tool call should fail the permission check even if the final answer is correct.

## Common judge biases

### Position bias

The judge may prefer the first or second answer. Swap answer order and measure the change.

### Length and style bias

The judge may prefer longer, more formal, or more detailed text. Include concise high-quality examples in calibration data. Report response length beside the score.

### Model-family bias

A judge may prefer outputs that resemble its own style. Test outputs from several model families and compare with humans.

### Reference bias

A weak reference answer can make a better alternative look wrong. Treat references as evidence, not automatic truth.

### Shared blind spots

The model under test and the judge may fail on the same topic. Agreement between them does not prove correctness.

## Validate the judge

Before using the judge at scale:

1. Select a held-out sample from the target task.
2. Write the rubric before judging model variants.
3. Collect independent human labels.
4. Run the judge with fixed settings.
5. Measure agreement overall and by important slice.
6. Swap pair order and vary harmless formatting.
7. Inspect disagreements.
8. Set an abstention or human-review rule.
9. Repeat validation after changing the judge, rubric, or task.

No single agreement number is enough. A judge can agree on easy examples and fail on the cases that control a launch.

## Test resistance to manipulation

Outputs may contain text aimed at the judge. Agent traces may include untrusted web pages, tool output, or generated instructions.

Protect the grader:

- separate the rubric from the candidate output;
- mark candidate text as untrusted data;
- limit what context the judge can see;
- test prompt injection and score requests;
- compare with deterministic checks;
- use human review for high-impact failures.

Training against a fixed judge can teach the policy to exploit that judge. Re-test with independent graders and fresh human labels.

## Small example

A team wants to measure whether answers cite the provided document correctly.

Use two graders:

1. Code checks that every citation points to an existing passage.
2. A model judge checks whether each cited passage supports its claim.

Humans label a held-out set. The team measures judge errors by document type and claim difficulty. Unsupported high-impact claims go to human review.

The final score keeps citation validity and claim support separate. One cannot hide failure in the other.

## In an interview

Use this order:

1. Define the behavior being measured.
2. Use deterministic checks where possible.
3. Write a concrete rubric.
4. Choose pointwise or pairwise judging.
5. Calibrate against independent humans.
6. Test order, length, family, and injection bias.
7. Report slices and disagreements.
8. Revalidate after model or task changes.

## Common mistakes

- Using the strongest available model without validation.
- Reporting judge scores to several decimal places.
- Letting one overall score hide hard failures.
- Using the same model family for generation and judgment without a bias check.
- Tuning prompts on the final test set.
- Training against one fixed judge and testing with that same judge.
- Treating judge agreement as ground truth.

*Related: [LLM Evals](/guides/llm-evals-the-hardest-part/), [preference data and reward models](/concepts/preference-data-and-reward-models/), and [evaluate an agent](/questions/evaluate-an-agent/).*