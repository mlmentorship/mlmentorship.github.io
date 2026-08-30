---
title: "A/B testing for ML systems"
description: "The framework for proving a model change actually helps. Statistical power, novelty effects, network effects, all the things people get wrong."
date: "2025-12-22"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

A/B testing for ML systems randomly assigns users to control and treatment models, then measures whether business outcome differences are statistically significant.

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

The most common A/B testing mistake: not running long enough. For two equal-sized arms and a continuous metric, the normal approximation for the minimum detectable effect (MDE) is:

```
MDE ≈ (z_(1-alpha/2) + z_(1-beta)) * sigma * sqrt(2/N)
```

where `sigma` is the metric's standard deviation and `N` is the sample size **per arm**. For a two-sided `alpha = 0.05` test at 80% power, the two z-scores are approximately 1.96 and 0.84. The exact calculation depends on the metric distribution and test.

<!-- visual:ab-test-mde-sample-size -->
<figure class="learning-figure plot-panel" aria-labelledby="ab-test-mde-visual-title">
	<p class="visual-kicker">Power intuition</p>
	<p class="visual-title" id="ab-test-mde-visual-title">Four times the users halves MDE; twice the noise doubles it.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 290" role="img" aria-labelledby="ab-test-mde-svg-title ab-test-mde-svg-desc">
			<title id="ab-test-mde-svg-title">Minimum detectable effect by per-arm sample size for two levels of metric noise</title>
			<desc id="ab-test-mde-svg-desc">An original line plot using a two-sided alpha of 0.05, 80 percent power, equal experiment arms, and the normal approximation. The horizontal axis gives 1,000, 4,000, and 16,000 observations per arm. For a metric standard deviation of one unit, shown as a solid line, minimum detectable effects are 0.125, 0.063, and 0.031 units. For a standard deviation of two units, shown as a dashed line, the values are 0.250, 0.125, and 0.063 units. Quadrupling sample size halves minimum detectable effect, while doubling standard deviation doubles it.</desc>
			<rect class="viz-plot-bg" x="58" y="28" width="272" height="210" rx="3"></rect>
			<path class="viz-gridline" d="M58 50H330 M58 144H330 M58 238H330 M70 28V238 M190 28V238 M310 28V238"></path>
			<path class="viz-axis" d="M58 28V238H330"></path>
			<path class="viz-pr-curve" style="stroke-dasharray: 7 5" d="M70 50 C110 88 150 128 190 144 C230 170 270 188 310 191"></path>
			<path class="viz-roc-curve" d="M70 144 C110 174 150 188 190 191 C230 207 270 214 310 215"></path>
			<circle class="viz-operating-point" cx="70" cy="50" r="4"></circle>
			<circle class="viz-operating-point" cx="190" cy="144" r="4"></circle>
			<circle class="viz-operating-point" cx="310" cy="191" r="4"></circle>
			<circle class="viz-operating-point" cx="70" cy="144" r="4"></circle>
			<circle class="viz-operating-point" cx="190" cy="191" r="4"></circle>
			<circle class="viz-operating-point" cx="310" cy="215" r="4"></circle>
			<text class="viz-callout" x="84" y="45">σ = 2 (dashed)</text>
			<text class="viz-callout" x="84" y="139">σ = 1 (solid)</text>
			<text class="viz-label" x="50" y="54" text-anchor="end">0.250</text>
			<text class="viz-label" x="50" y="148" text-anchor="end">0.125</text>
			<text class="viz-label" x="50" y="242" text-anchor="end">0</text>
			<text class="viz-label" x="70" y="256" text-anchor="middle">1k</text>
			<text class="viz-label" x="190" y="256" text-anchor="middle">4k</text>
			<text class="viz-label" x="310" y="256" text-anchor="middle">16k</text>
			<text class="viz-axis-label" x="194" y="281" text-anchor="middle">observations per arm (N)</text>
			<text class="viz-axis-label" transform="translate(14 190) rotate(-90)">MDE (metric units)</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> follow either line from 1k to 4k to 16k users per arm: each 4× increase halves the effect the test can detect. At any sample size, the dashed σ = 2 line is twice as high, so a metric with twice the noise needs 4× the users to detect the same effect.</figcaption>
</figure>

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

A common follow-up asks what could block launch despite a positive primary metric. A senior answer checks guardrail and slice regressions, statistical power, unresolved novelty effects, and strategic concerns that the experiment does not measure.

---

*Related: [causal inference for ML decisions](/concepts/causal-inference-for-ml-decisions/), [How would you evaluate an LLM application?](/questions/how-would-you-evaluate-an-llm-application/), [Design YouTube's recommender](/questions/design-youtube-recommender/), and [Calibration](/concepts/calibration/).*
