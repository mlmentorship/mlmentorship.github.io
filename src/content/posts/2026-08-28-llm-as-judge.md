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