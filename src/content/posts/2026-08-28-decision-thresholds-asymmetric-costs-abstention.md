---
title: "Decision thresholds, asymmetric costs, and abstention"
description: "Choose actions from calibrated probabilities, error costs, and capacity constraints. Use separate thresholds for automatic action, human review, and abstention."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["classification threshold", "cost-sensitive classification", "reject option", "abstention", "human review threshold"]
roles: ["Applied Scientist", "Machine Learning Engineer", "Safety and Evals"]
rounds: ["Evaluation", "Product", "ML system design"]
difficulty: "Intermediate"
priority: "Core"
prerequisites: ["calibration", "precision-recall-f1"]
---

## Summary

A classifier score becomes a decision only after a threshold maps it to an action. The right threshold depends on the cost of false positives and false negatives, the probability calibration, review capacity, and constraints such as recall or safety limits.

Many systems need more than one threshold. High-confidence cases can receive an automatic action, uncertain cases can go to human review, and low-confidence cases can be allowed or rejected. Abstention is a valid action when the cost of deciding exceeds the cost of delay or escalation.

## Bayes decision rule

Suppose $p=P(Y=1\mid x)$ is calibrated. Let $C_{FP}$ be the cost of a false positive and $C_{FN}$ the cost of a false negative.

Predict positive when its expected error cost is lower:

$$
C_{FP}(1-p) < C_{FN}p.
$$

Solving for $p$ gives the threshold

$$
t^* = \frac{C_{FP}}{C_{FP}+C_{FN}}.
$$

If false negatives cost nine times as much as false positives, then $t^*=0.1$. The system should act at a lower probability because missing a positive is expensive.

This formula assumes the probabilities and costs are correct for the deployment population. Real systems often add policy constraints and operational limits.

## Scores are not probabilities

A ranking score can order examples well without representing $P(Y=1\mid x)$. The Bayes threshold formula does not apply directly to an arbitrary score.

Options include:

- calibrate the score on representative held-out data;
- choose a threshold from a precision-recall or ROC curve;
- optimize the measured business or safety objective directly;
- treat the threshold as a policy parameter and validate it online.

Recalibrate after a distribution shift, model update, or label-definition change.

## Constraints instead of costs

Teams often cannot assign credible dollar values to every error. Use an explicit constraint:

- maximize precision subject to recall at least 95%;
- minimize false negatives subject to 1,000 reviews per day;
- maximize accepted traffic subject to a safety-violation limit;
- minimize latency subject to a quality floor.

This turns threshold selection into an operating-point choice. Report the tradeoff curve and the chosen constraint.

## Review capacity

A review queue creates a hard capacity limit. If reviewers can inspect $K$ cases per day, choose the review region from the highest-value uncertain cases rather than a fixed score band by habit.

One policy may be:

```text
p >= 0.98          automatic block
0.60 <= p < 0.98  rank by expected review value
p < 0.60           allow
```

The queue can prioritize expected harm, model uncertainty, user impact, or information gain. Monitor backlog age and reviewer disagreement because the policy fails when the queue saturates.

<!-- visual:decision-threshold-action-regions -->
<figure class="learning-figure plot-panel" aria-labelledby="decision-policy-visual-title">
	<p class="visual-kicker">Spatial intuition</p>
	<p class="visual-title" id="decision-policy-visual-title">Two thresholds turn one calibrated risk score into three actions.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 292" role="img" aria-labelledby="decision-policy-svg-title decision-policy-svg-desc">
			<title id="decision-policy-svg-title">Allow, review, and block regions on a calibrated fraud-risk score</title>
			<desc id="decision-policy-svg-desc">A horizontal probability scale from zero to one has a solid threshold at 0.60 and another at 0.98. Scores below 0.60 are allowed, scores from 0.60 to 0.98 abstain to human review, and scores at or above 0.98 are blocked. Distinct dot, diagonal, and crosshatch patterns identify the regions. Below the scale, the worked example contrasts a 500 dollar missed-fraud cost with a 20 dollar false-block cost and notes that capacity and policy still determine the final review boundaries.</desc>
			<defs>
				<pattern id="allow-dots" width="8" height="8" patternUnits="userSpaceOnUse">
					<circle cx="2" cy="2" r="1.2" fill="var(--viz-input-stroke)"></circle>
				</pattern>
				<pattern id="review-hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(35)">
					<path class="viz-gridline" d="M0 0V8"></path>
				</pattern>
				<pattern id="block-crosshatch" width="6" height="6" patternUnits="userSpaceOnUse">
					<path class="viz-gridline" d="M0 0L6 6M6 0L0 6"></path>
				</pattern>
			</defs>
			<text class="viz-axis-label" x="24" y="22">CALIBRATED FRAUD RISK p</text>
			<text class="viz-callout" x="202" y="43" text-anchor="middle">t₁ = 0.60</text>
			<text class="viz-callout" x="306" y="43" text-anchor="end">t₂ = 0.98</text>
			<rect class="viz-node viz-node--input" x="24" y="58" width="169" height="68" rx="3"></rect>
			<rect x="24" y="58" width="169" height="68" rx="3" fill="url(#allow-dots)"></rect>
			<rect class="viz-node viz-node--focus" x="193" y="58" width="107" height="68"></rect>
			<rect x="193" y="58" width="107" height="68" fill="url(#review-hatch)"></rect>
			<rect class="viz-node viz-node--output" x="300" y="58" width="6" height="68"></rect>
			<rect x="300" y="58" width="6" height="68" fill="url(#block-crosshatch)"></rect>
			<path class="viz-axis" d="M193 48V138M300 48V138M24 138H306"></path>
			<text class="viz-callout" x="108" y="86" text-anchor="middle">ALLOW</text>
			<text class="viz-label" x="108" y="103" text-anchor="middle">model decides</text>
			<text class="viz-callout" x="246" y="82" text-anchor="middle">REVIEW</text>
			<text class="viz-label" x="246" y="99" text-anchor="middle">abstain</text>
			<text class="viz-label" x="246" y="114" text-anchor="middle">human decides</text>
			<path class="viz-operating-guide" d="M304 76H326V63"></path>
			<text class="viz-callout" x="332" y="58" text-anchor="middle">BLOCK</text>
			<text class="viz-label" x="24" y="154" text-anchor="middle">0</text>
			<text class="viz-label" x="193" y="154" text-anchor="middle">0.60</text>
			<text class="viz-label" x="300" y="154" text-anchor="middle">0.98</text>
			<text class="viz-label" x="306" y="169" text-anchor="middle">1</text>
			<rect class="viz-plot-bg" x="24" y="188" width="312" height="78" rx="3"></rect>
			<text class="viz-axis-label" x="36" y="207">ASYMMETRIC ERROR COSTS · WORKED EXAMPLE</text>
			<text class="viz-callout" x="36" y="229">missed fraud C_FN = $500</text>
			<text class="viz-callout" x="324" y="229" text-anchor="end">false block C_FP = $20</text>
			<path class="viz-axis" d="M36 240H324"></path>
			<circle class="viz-operating-point" cx="46" cy="240" r="6"></circle>
			<circle class="viz-operating-point" cx="314" cy="240" r="3"></circle>
			<text class="viz-label" x="180" y="258" text-anchor="middle">costs + capacity + policy set final thresholds</text>
			<text class="viz-label" x="180" y="282" text-anchor="middle">dollar values do not derive t₁ or t₂</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> below 0.60 the system allows, from 0.60 to 0.98 it abstains to human review, and at 0.98 it blocks. The much larger missed-fraud cost favors earlier intervention, while review capacity and policy determine where the two operational thresholds finally sit.</figcaption>
</figure>

## Abstention and selective prediction

A model with a reject option predicts only when confidence or estimated risk passes a rule. Two useful quantities are:

- **coverage:** fraction of examples that receive a model decision;
- **selective risk:** error rate on the decided examples.

Lower coverage can reduce selective risk if uncertainty estimates are useful. The system still needs a safe fallback for abstained examples.

Abstention can mean human review, a slower model, a rules system, a request for more information, or no action. Its cost belongs in the decision rule.

## Per-group thresholds

Different groups may have different base rates, error costs, or measurement quality. One global threshold can create unequal error rates.

Changing thresholds by group can improve one fairness criterion and harm another. It may also face legal or policy limits. State the target criterion, measurement uncertainty, and governance process. Do not present threshold adjustment as a complete fairness solution.

## Distribution shift

A threshold chosen on yesterday's data can fail when prevalence changes. Even with stable class-conditional score distributions, precision changes with the base rate.

Monitor:

- score and prevalence distributions;
- calibration by time and important slice;
- precision and recall at the operating point;
- action volume and review backlog;
- downstream harm and appeal outcomes;
- fallback use.

A stable AUC does not guarantee a stable operating point.

## Worked example

A fraud model estimates a 4% fraud probability. A false block costs $20$, while a missed fraud case costs $500$.

The cost-based threshold is

$$
t^*=\frac{20}{20+500}\approx 0.038.
$$

The expected-cost rule would block at 4%. If automatic blocking has legal or customer constraints, the same score may instead enter review. The final action depends on both expected cost and policy.

## In an interview

Use this order:

1. Define the available actions and error costs.
2. Ask whether the score is calibrated.
3. Derive or choose an operating point.
4. Add review capacity and abstention.
5. Check important slices and policy constraints.
6. Monitor calibration, action volume, and outcomes after launch.

## Common mistakes

- Using 0.5 as a universal threshold.
- Optimizing AUC without choosing an operating point.
- Applying probability formulas to uncalibrated scores.
- Ignoring review capacity and queue delay.
- Treating abstention as failure instead of an action.
- Selecting a threshold once and never monitoring it.

## Practice next

Use this framework in [calibration](/concepts/calibration/), [class imbalance](/questions/class-imbalance/), [fraud-system design](/questions/design-fraud-detection/), and [content moderation](/questions/content-moderation/).
