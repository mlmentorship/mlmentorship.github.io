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
	<p class="visual-title" id="ab-test-mde-visual-title">Randomize, commit the readout, and budget for noise and delayed effects</p><div class="visual-argument" data-visual-argument="ab-test-mde-sample-size" data-argument-kind="bars" data-argument-composition="experiment-and-power"><svg viewBox="0 0 360 1383" role="img" aria-label="Randomize, commit the readout, and budget for noise and delayed effects. Randomize comparable units and compare observed outcomes, not chosen winners. Keep assignment consistent and choose the readout before looking. Spillovers or delayed responses can invalidate a simple early comparison. With effect and error targets fixed, samples scale with variance. Doubled noise width covers four variance tiles, requiring four baseline sample budgets."><title>Randomize, commit the readout, and budget for noise and delayed effects</title><desc>Randomize comparable units and compare observed outcomes, not chosen winners. Keep assignment consistent and choose the readout before looking. Spillovers or delayed responses can invalidate a simple early comparison. With effect and error targets fixed, samples scale with variance. Doubled noise width covers four variance tiles, requiring four baseline sample budgets.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Randomize units; compare observed</tspan><tspan x="20" dy="18">outcomes</tspan></text><text x="20" y="80" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Eight comparable units; one fixed</tspan><tspan x="20" dy="18">assignment</tspan></text><g data-picture-experiment-source="0"><circle cx="36" cy="112" r="6" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-experiment-source="1"><circle cx="77" cy="112" r="6" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-experiment-source="2"><circle cx="118" cy="112" r="6" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-experiment-source="3"><circle cx="159" cy="112" r="6" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-experiment-source="4"><circle cx="200" cy="112" r="6" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-experiment-source="5"><circle cx="241" cy="112" r="6" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-experiment-source="6"><circle cx="282" cy="112" r="6" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-experiment-source="7"><circle cx="323" cy="112" r="6" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path d="M36 126L54 218M48.53833652947467 210.91690518666246L54 218L56.38947776835483 209.38081233557722" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-experiment-member="0" data-picture-experiment-group="0" data-picture-experiment-positive="1"><circle cx="54" cy="233" r="8" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path d="M159 126L79 218M81.23100763074427 209.33844096299282L79 218L87.26785180805231 214.5878706823911" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-experiment-member="3" data-picture-experiment-group="0" data-picture-experiment-positive="0"><circle cx="79" cy="233" r="8" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path d="M200 126L104 218M107.00828418798093 209.57680427365338L104 218L112.54352709386586 215.35270991457676" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-experiment-member="4" data-picture-experiment-group="0" data-picture-experiment-positive="0"><circle cx="104" cy="233" r="8" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path d="M323 126L129 218M134.51443776473826 210.95791393557076L129 218L137.94233151038634 218.18629857313303" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-experiment-member="7" data-picture-experiment-group="0" data-picture-experiment-positive="0"><circle cx="129" cy="233" r="8" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><text x="91" y="274" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="91" dy="0">A: 1 positive</tspan></text><path d="M77 126L234 218M225.07544109558745 217.4064948511844L234 218L229.12006877640493 210.50424978718058" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-experiment-member="1" data-picture-experiment-group="1" data-picture-experiment-positive="0"><circle cx="234" cy="233" r="8" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><path d="M118 126L259 218M250.1142648820116 216.9783780479318L259 218L254.4858564908615 210.27843873436836" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-experiment-member="2" data-picture-experiment-group="1" data-picture-experiment-positive="1"><circle cx="259" cy="233" r="8" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><path d="M241 126L284 218M276.9888786383355 212.4462465618276L284 218L284.23632993353925 209.05885084776497" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-experiment-member="5" data-picture-experiment-group="1" data-picture-experiment-positive="1"><circle cx="284" cy="233" r="8" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><path d="M282 126L309 218M302.90906146371503 211.4501551356388L309 218L310.58531276971803 209.19734225235533" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-experiment-member="6" data-picture-experiment-group="1" data-picture-experiment-positive="1"><circle cx="309" cy="233" r="8" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><text x="271" y="274" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="271" dy="0">B: 3 positive</tspan></text><text x="20" y="311" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Fill: a measured positive; not a</tspan><tspan x="20" dy="18">significance claim</tspan></text><path d="M100 375H260" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:5 4"></path><path d="M174 369L186 381" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M174 381L186 369" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="20" y="413" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Cross-group effects break isolation</tspan></text><text x="20" y="472" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Fix the readout before seeing results</tspan></text><path d="M35 548L325 548" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="128" cy="548" r="7" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="20" y="589" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">early response</tspan></text><path d="M35 668L325 668" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="298" cy="668" r="7" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="20" y="709" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">delayed response</tspan></text><path data-picture-experiment-readout="230" d="M230 520V688" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="230" y="754" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="230" dy="0">fixed readout</tspan></text><text x="20" y="809" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Do not move the finish line after</tspan><tspan x="20" dy="18">peeking</tspan></text><g transform="translate(0 895)"><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Same effect; same error targets</tspan></text><text x="20" y="61" style="font-size:16px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Noise is a width. Variance is</tspan><tspan x="20" dy="20">its square.</tspan></text><g data-picture-square="1" data-picture-variance="1"><rect x="36" y="188" width="72" height="72" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></rect><circle cx="72" cy="224" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path d="M36 272L108 272" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="72" y="299" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="72" dy="0">noise 1</tspan></text><text x="72" y="346" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="72" dy="0">1x samples</tspan></text><g data-picture-square="2" data-picture-variance="4"><rect x="184" y="188" width="72" height="72" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><circle cx="220" cy="224" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><rect x="256" y="188" width="72" height="72" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><circle cx="292" cy="224" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><rect x="184" y="116" width="72" height="72" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><circle cx="220" cy="152" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><rect x="256" y="116" width="72" height="72" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><circle cx="292" cy="152" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><path d="M184 272L328 272" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><text x="256" y="299" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="256" dy="0">noise 2</tspan></text><text x="256" y="346" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="256" dy="0">4x samples</tspan></text><text x="20" y="397" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Four equal tiles need four equal</tspan><tspan x="20" dy="18">budgets</tspan></text><text x="20" y="445" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Relative scaling, not a sample-size</tspan><tspan x="20" dy="18">prescription</tspan></text></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> follow either line from 1k to 4k to 16k users per arm: each 4× increase halves the effect the test can detect. At any sample size, the dashed σ = 2 line is twice as high, so a metric with twice the noise needs 4× the users to detect the same effect.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="ab-test-mde-sample-size"><strong>Check:</strong> How much must N increase to keep the same MDE when noise doubles?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="ab-test-mde-sample-size">Randomize comparable units and compare observed outcomes, not chosen winners. Keep assignment consistent and choose the readout before looking. Spillovers or delayed responses can invalidate a simple early comparison. With effect and error targets fixed, samples scale with variance. Doubled noise width covers four variance tiles, requiring four baseline sample budgets. <strong>Tripling noise requires how many baseline sample budgets under the same assumptions?</strong></span></figcaption>
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
