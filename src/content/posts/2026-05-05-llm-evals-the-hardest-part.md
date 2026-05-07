---
title: "LLM Evals: The hardest part of shipping LLMs, and why most teams get it wrong"
description: "Your model is only as good as your eval. Your eval is a product. Treat it like one. The patterns that separate teams that ship from teams that thrash."
date: "2026-05-05"
draft: false
tags: ["essays"]
category: "essays"
---


**Your model is only as good as your eval. Your eval is a product. Treat it like one.**

The eval pipeline is the single highest-leverage piece of engineering in any serious LLM application, and the piece most teams underinvest in (because it doesn't ship and isn't visible). The patterns below separate teams that ship reliably from teams that thrash.

## Why this is harder than classical ML eval

Classical ML eval: hold 10%, compute AUC, done. LLMs broke this in five ways:

1. **The output space is open-ended.** "Write a summary" has uncountably many correct answers and infinitely many wrong ones. There is no ground truth to compare against, only judgments.
2. **The task is fuzzy.** "Be helpful" doesn't decompose into a metric. You're evaluating a *behavior*, not a *prediction*.
3. **The model affects what you measure.** A summarization model's outputs become training data for the next iteration; the eval set drifts as users adapt.
4. **The task changes faster than the eval can keep up.** A new model release, a new prompt, a new tool integration, each invalidates parts of yesterday's eval.
5. **Cost and latency are first-class.** A 3% quality bump that costs 5&times; more is a regression. Classical ML eval rarely had to weigh dollars.

None of these are solved problems. They are, instead, *managed* problems. The teams that ship are the teams that manage them deliberately.

## The hierarchy of eval signals

Not all eval signals are equal. In rough order of trustworthiness (most → least):

1. **Production business metrics.** Task completion rate, retention, revenue. These are the ground truth, but they are slow, noisy, and expensive to move.
2. **Production behavioral metrics.** User edit-distance on outputs, copy rate, regenerate rate, thumbs. Fast, free, but proxy.
3. **Human ratings on production traffic samples.** Slow ($), but high-fidelity. The bridge between offline and online.
4. **Human ratings on a curated golden set.** Fast (small set), reproducible, but suffers distribution drift.
5. **LLM-as-judge on a curated golden set.** Cheap, scalable, but biased and needs calibration.
6. **Automated metrics (exact match, regex, programmatic checks).** Cheap and reliable, but only apply to closed-ended subtasks.
7. **Reference-based NLP metrics (BLEU, ROUGE, BERTScore).** Cheap, but correlate poorly with quality on instruction-following. Use only for sanity checks.
8. **Public benchmarks (MMLU, MT-Bench, etc.).** Useful for vendor comparison, near-useless for product decisions. They are *leaderboards*, not evals.
9. **Perplexity / training loss.** Useful during pretraining. Not an eval.

Most teams pick rows 5-6, automate, and stop. Good teams maintain signals at multiple levels and watch correlations between them.

## Building the eval, in order

Build these in this sequence. Don't skip steps.

### Step 1: Define the task in user terms

Write what success looks like *for the user* in 1-2 paragraphs: "The summary captures key actionable points," not "high ROUGE." If you can't write this, you lack a clear problem; go talk to users first.

### Step 2: Build the golden set (50-500 examples)

Hand-curated. Stratified across:
- **Real traffic samples** (so you measure the actual distribution)
- **Known failure modes** (so regressions are caught)
- **Edge cases** (so launches don't break)

Crucially, **the team builds this together**. The act of labeling 200 examples *is itself the most important eval activity*: it forces shared understanding of what good looks like. Outsourcing this step is the single most expensive shortcut you can take.

Version in git with maintainers and changelog. When adding examples, document *why*. Treat as production code.

### Step 3: Build a scoring rubric

For each example, decide *how* to score model output:

- **Exact match / programmatic**: for tasks with structured output (JSON, code, classification). Use these wherever possible. They're free, reliable, and unambiguous.
- **Reference-based**: when there's a canonical correct answer text. Rare for LLM tasks; mostly applies to translation and short factual QA.
- **Rubric-based human rating**: for everything else. The rubric should be a 3-5 point checklist with concrete examples of each point: "✓ Captures the conclusion. ✓ Cites the right entities. ✗ Adds unsupported claims." Vague rubrics produce noisy ratings; concrete rubrics produce signal.

Avoid 1-7 scales; inter-annotator agreement collapses. Use binary or pairwise.

### Step 4: Calibrate humans

Have two team members rate the same 50 examples. Compute Cohen's &kappa;. If it's below ~0.6, your rubric is too vague, go back to Step 3. Disagreement is not a labeling problem; it's a *spec problem*. Fix the spec.

Teams that skip this step wonder why numbers don't match reality.

### Step 5: Bring in LLM-as-judge (carefully)

LLM-as-judge is the only thing that scales. But it's biased and unreliable in known ways:

- **Position bias**: prefers whichever option appears first (or second, depending on model). Mitigation: randomize order, evaluate both orderings.
- **Length bias**: prefers longer responses. Mitigation: include a "concise" criterion in the rubric, or report length as a covariate.
- **Family bias**: GPT-4 prefers GPT-4-style outputs. Mitigation: use a different model family for judging than for generation, when possible.
- **Domain blindness**: the judge fails on tasks requiring knowledge it lacks (specialized medical, legal, code). Mitigation: don't use judges in those domains without human validation.

The protocol that works:

1. Take ~200 examples from your golden set.
2. Have humans rate them (with the rubric from Step 3).
3. Have the LLM judge rate the same examples.
4. Compute judge-vs-human agreement. **If &kappa; < 0.5, do not deploy this judge.** Iterate on the judge prompt, or pick a different judge.
5. Re-validate quarterly, and after any model upgrade.

When the judge agrees with humans, you can scale to 5K examples cheaply. When it doesn't, you've learned something about the task before you wasted a quarter.

### Step 6: Pairwise > Pointwise for open-ended

A surprising amount of eval pain disappears when you switch from "rate this output 1-5" to "which of these two is better." Inter-annotator agreement roughly doubles. LLM-as-judge agreement also rises sharply.

To get a single ranking from pairwise comparisons, fit a Bradley-Terry model. There are 50-line implementations in any language. You'll get ELO-like ratings for each model variant, with confidence intervals.

### Step 7: Online signals

Offline eval is necessary but never sufficient. Wire up at least:

- **Implicit signals**: regeneration rate, edit distance between LLM output and final user output, copy rate, time-to-action.
- **Explicit signals**: thumbs up/down (sparse, biased toward dissatisfaction).
- **Outcome signals**: whatever the business actually measures.

Then, the critical step, **measure the correlation between your offline eval and your online metric** every quarter. This number is the most important number on your team's dashboard. If it falls below ~0.6, your offline eval is lying to you, and any model "improvement" you ship based on it is a coin flip.

## Patterns in teams that ship

1. **They have a Friday eval review.** A standing 30-minute meeting where the team looks at 10 fresh examples together. Not metrics, examples. This is where shared taste forms. This is the most important meeting.

2. **They version the eval set.** Every change is a PR. Every PR has a justification. The eval set is a tracked artifact, not a Google Sheet.

3. **They report Pareto curves, not points.** Quality vs. cost vs. latency. Decisions are made on the curve, not on the headline number.

4. **They cut by slice.** The launch decision is gated on no-regression on every high-stakes slice (paying users, high-volume queries, regulated content). Aggregate improvements that hide slice regressions get caught.

5. **They have a story for every number that changes by more than ~2 standard errors.** Either it's real, or the eval is broken. Either way, you investigate.

6. **They rebuild the eval set when production drifts.** Quarterly at minimum. Models, users, and tasks all move; static eval sets become museum pieces.

## Patterns in teams that thrash

1. **One person owns eval.** When they leave, the institutional knowledge of "what counts as good" leaves with them.

2. **The eval is a Colab notebook.** Not in git, not versioned, not reproducible.

3. **They optimize the metric.** They beat the leaderboard, then ship, then watch online metrics fall, then can't explain why.

4. **They use LLM-as-judge without calibration.** They report numbers down to two decimals and trust them.

5. **They never look at outputs.** Spreadsheet of metrics; nobody has read 20 examples in a month. Taste atrophies. The team ships things that are technically better and humanly worse.

6. **They report aggregate metrics only.** Then a launch breaks something for 2% of users that turn out to be 40% of revenue.

## A reasonable starting stack in 2026

For a fresh LLM eval pipeline today:

- **Golden set**: ~300 examples in a versioned repo, JSONL format, with metadata for slicing.
- **Automated scoring**: programmatic checks for any closed-ended subtask. Mandatory.
- **LLM-as-judge**: Claude-class or GPT-class model with a structured rubric, validated against ~200 human ratings. Pairwise where possible.
- **Online wiring**: regeneration rate + edit distance + business metric, in that order of investment.
- **Dashboard**: Pareto curve (quality / cost / latency), offline-online correlation, per-slice table, time-series of all of the above.
- **Process**: Friday review, weekly metric report, quarterly eval-set rebuild and judge re-validation.
- **Tooling**: any of Inspect AI, Promptfoo, LangSmith, Braintrust, or, increasingly, just well-structured pytest. Do not write your own framework. Do not use a vendor that locks your data.

## The one thing that matters most

If you take one thing from this essay: **read the outputs**.

Not metrics. The outputs. Twenty a week, forever.

Senior practitioners share this habit. It's not glamorous or scalable, but it's the highest-leverage activity. Metrics show *what changed*; outputs show *what's happening*. The teams that ship reliably have senior people with taste from reading thousands of outputs. Eval systems scale that taste, not replace it.

---

*Related: ["How would you evaluate an LLM application you've built?"](/interviews/how-would-you-evaluate-an-llm-application/), an interview question with leveled answers covering this material.*
