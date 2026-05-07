---
title: "A/B testing for ML systems"
description: "The framework for proving a model change actually helps. Statistical power, novelty effects, network effects, all the things people get wrong."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---


## One-line definition

A/B testing for ML systems randomly assigns users to control and treatment models, then measures whether business outcome differences are statistically significant.

## Why it matters

Offline metrics are unreliable for ML systems. They suffer from distribution shift, label bias, observational confounds, and Goodhart's law. The only reliable way to know if a model change actually helps in production is to test it against the current model on real users.

It bridges offline experiments and production decisions. Standard for senior ML roles.

## The basics

Set up:
- Pick a primary metric tied to the business outcome (revenue, retention, task completion).
- Pick guardrail metrics (latency, cost, fairness slices) that should not regress.
- Randomize users (or sessions, or some unit) into control and treatment groups.
- Run long enough to achieve statistical power; analyze the difference.

Analysis:
- Compute the metric for each group.
- Compute the difference (effect size).
- Compute statistical significance (p-value, confidence interval).
- Decide: ship if effect is positive and significant, no guardrail regressions; otherwise hold or iterate.

## Statistical power

The most common A/B testing mistake: not running long enough. The minimum detectable effect (MDE) at given power and sample size is:

```
MDE ~ z * sigma / sqrt(N)
```

where `sigma` is the metric's standard deviation, `N` is the sample size per arm, and `z` depends on power (typically `z = 2.8` for 80% power at p=0.05).

For a small movement on a noisy metric, you need many users. If "no significant effect" appears, first check whether the test had power to detect your target effect.

## What an interviewer expects you to discuss

If asked about A/B testing for ML:

1. Define the basic setup (random assignment, primary + guardrails, statistical analysis).
2. Discuss statistical power and minimum detectable effect.
3. Mention multiple comparison correction (Bonferroni, BH) when looking at many metrics or slices.
4. Mention novelty / primacy effects (users react differently to new things initially).
5. Mention network / spillover effects (especially for recsys, social systems).
6. Mention SUTVA violations (when one user's treatment affects another's outcome).

For senior LLM-team interviews specifically:

7. Discuss how A/B testing for LLMs is harder (fewer users, slower iteration, harder to define metrics, longer time to outcome).

## Common pitfalls

### Multiple comparisons

Looking at 20 metrics and reporting "any significant" lets you find spurious wins. Fix: pick one primary metric in advance; treat the rest as guardrails (one-sided tests with stricter thresholds) or apply correction (Bonferroni, Holm, BH).

### P-hacking via early stopping

Looking at the test daily and stopping when significant inflates false positives massively. Fix: pre-commit to a sample size or use sequential testing methods (group sequential, mSPRT, always-valid p-values).

### Novelty effects

In the first days/weeks of a test, users react to "this is new" rather than to the underlying change. Fix: run the test long enough for the novelty to wear off (often 2-4 weeks for consumer products).

### Network / spillover effects

In social or marketplace systems, one user's treatment affects others' outcomes. Fix: cluster randomization at community level, or accept and document the bias.

### Heterogeneous treatment effects

Average effect can be positive while important segments regress. Fix: pre-specify slices to check; require no regression in critical slices even if average improves.

### Sample ratio mismatch (SRM)

If randomization breaks, the assignment ratio won't match expectations. Always check SRM; it almost always indicates a bug.

## Special cases for ML

### Recsys A/B testing

- Primary metric is usually long-term (next-day return, retention) but you only get short-term signal during the test. Use proxies + a longer hold-out for confirmation.
- Counterfactual evaluation (IPS, doubly robust) on logged data can pre-screen models before A/B test.
- Beware of feedback loops: your treatment changes user behavior which changes future training data.

### LLM A/B testing

- Per-request latency and cost are first-order metrics, not just quality.
- LLM quality is hard to measure online (no good aggregate metric for "is the answer good"). Use proxies (regeneration rate, edit distance, thumbs).
- Run shorter A/B tests with smaller traffic for cost; rely on offline eval to catch most issues before reaching A/B.

### Search ranking

- Click models / interleaving experiments can be more powerful than user-level A/B for ranker comparisons (less variance per query).

## When you can't A/B test

Sometimes A/B testing is infeasible:
- Brand-new products with no users.
- Regulatory constraints (some financial / medical settings).
- High-stakes decisions where exposing any user to a worse model is unacceptable.

Alternatives:
- **Shadow mode**: run the new model alongside the old one without exposing users. Compare on logged data using counterfactual estimators.
- **Quasi-experimental designs**: regression discontinuity, difference-in-differences, synthetic control.
- **Phased rollouts**: deploy to a small percentage, monitor, expand.

## Why interviewers ask

A/B testing questions test:
1. Whether you've actually deployed a model and validated it.
2. Whether you understand statistical reasoning beyond p-values.
3. Whether you know the practical pitfalls (multiple comparisons, novelty effects, network effects).
4. Whether you can advocate for or against shipping a change based on data.

A common follow-up: "What would make you not ship a model with a positive primary metric?" The senior answer: regressions in guardrails, regressions in important slices, low statistical power despite the positive estimate, novelty effects not having decayed yet, or strategic concerns the data doesn't capture.

---

*Related: [How would you evaluate an LLM application?](/interviews/how-would-you-evaluate-an-llm-application/), [Design YouTube's recommender](/interviews/design-youtube-recommender/), [Calibration](/reference/calibration/).*
