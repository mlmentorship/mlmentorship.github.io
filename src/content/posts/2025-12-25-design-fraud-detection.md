---
title: "Design fraud detection for a payment company"
description: "Fraud has the worst data of any ML problem: heavily imbalanced, biased labels, adversarial actors, and direct money on the line. The senior answer respects all four."
date: "2025-12-25"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: fintech and risk-team interviews.*

The L4 candidate proposes "train an XGBoost on transactions." The L6 candidate addresses label bias, adversarial drift, the precision-recall trade-off as a business decision, and the rules-vs-ML hybrid that defines the field.

## Why fraud is uniquely hard

1. **Heavy imbalance**: typical fraud rates are well under 1%.
2. **Biased labels**: you only see fraud you caught (or that customers reported). Missed fraud is a hidden majority.
3. **Adversarial drift**: fraudsters adapt. Today's good model is tomorrow's bypassed one.
4. **Asymmetric cost**: missed fraud costs money; false positives cost customer trust and revenue.
5. **Latency budget**: real-time decisions in tens of milliseconds.

Every architectural choice follows from one or more of these.

**Learning objective:** Trace how the deployed fraud policy determines which outcomes become labels, then identify why an independently reviewed sample must cover both allowed and declined transactions.

<!-- visual:fraud-policy-label-coverage -->
<figure class="learning-figure plot-panel" aria-labelledby="fraud-policy-label-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="fraud-policy-label-title">See how the deployed policy selects its own fraud labels.</p>
	<svg viewBox="0 0 360 530" role="img" aria-labelledby="fraud-policy-label-svg-title fraud-policy-label-svg-desc">
		<title id="fraud-policy-label-svg-title">Fraud actions create different label coverage</title>
		<desc id="fraud-policy-label-svg-desc">All attempted payments pass through current rules and a model. The allow branch enters an outcome window. A confirmed report becomes an observed fraud positive; a mature payment with no report is only a noisy negative; and an immature outcome remains pending, not negative. The decline or challenge branch has no realized payment outcome, so its counterfactual is missing and cannot be labelled legitimate. Dashed paths sample both action branches for independent analyst review, creating audit labels with coverage across actions. Policy-selected outcome labels and audit labels enter training and evaluation with their sources retained.</desc>
		<defs><marker id="fraud-label-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
		<rect class="viz-node viz-node--input" x="55" y="12" width="250" height="42" rx="4"></rect>
		<text class="viz-callout" x="180" y="30" text-anchor="middle">ALL ATTEMPTED PAYMENTS</text>
		<text class="viz-label" x="180" y="46" text-anchor="middle">same decision-time population</text>
		<path d="M180 56V74" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fraud-label-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="55" y="80" width="250" height="42" rx="4"></rect>
		<text class="viz-callout" x="180" y="98" text-anchor="middle">CURRENT RULES + MODEL</text>
		<text class="viz-label" x="180" y="114" text-anchor="middle">policy chooses the action</text>
		<path d="M180 124V140H91V154" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fraud-label-arrow)"></path>
		<path d="M180 140H269V154" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fraud-label-arrow)"></path>
		<rect class="viz-node" x="14" y="160" width="154" height="50" rx="4"></rect>
		<text class="viz-callout" x="91" y="180" text-anchor="middle">ALLOW</text>
		<text class="viz-label" x="91" y="198" text-anchor="middle">wait through outcome window</text>
		<rect class="viz-node" x="192" y="160" width="154" height="50" rx="4" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect>
		<text class="viz-callout" x="269" y="180" text-anchor="middle">DECLINE / CHALLENGE</text>
		<text class="viz-label" x="269" y="198" text-anchor="middle">no realized payment outcome</text>
		<path d="M91 212V230" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fraud-label-arrow)"></path>
		<rect class="viz-node" x="14" y="236" width="154" height="72" rx="4"></rect>
		<text class="viz-callout" x="91" y="254" text-anchor="middle">MATURE OUTCOMES</text>
		<text class="viz-label" x="26" y="273">report -> observed positive</text>
		<text class="viz-label" x="26" y="290">no report -> noisy negative</text>
		<text class="viz-label" x="26" y="303">pending -> not negative</text>
		<rect class="viz-node" x="192" y="236" width="154" height="72" rx="4" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect>
		<text class="viz-callout" x="269" y="254" text-anchor="middle">MISSING != LEGITIMATE</text>
		<text class="viz-label" x="269" y="275" text-anchor="middle">exclude from ordinary</text>
		<text class="viz-label" x="269" y="292" text-anchor="middle">outcome evaluation</text>
		<path d="M91 310V328" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fraud-label-arrow)"></path>
		<rect class="viz-node" x="14" y="334" width="154" height="48" rx="4" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect>
		<text class="viz-callout" x="91" y="353" text-anchor="middle">OUTCOME LABELS</text>
		<text class="viz-label" x="91" y="371" text-anchor="middle">selected by policy</text>
		<path d="M91 152H6V388Q6 404 22 414L53 431" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:5 4;marker-end:url(#fraud-label-arrow)"></path>
		<path d="M269 152H354V388Q354 404 338 414L307 431" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:5 4;marker-end:url(#fraud-label-arrow)"></path>
		<text class="viz-axis-label" x="180" y="398" text-anchor="middle">INDEPENDENT SAMPLE FROM BOTH ACTIONS</text>
		<rect class="viz-node viz-node--focus" x="55" y="410" width="250" height="48" rx="4"></rect>
		<text class="viz-callout" x="180" y="429" text-anchor="middle">ANALYST REVIEW / ADJUDICATION</text>
		<text class="viz-label" x="180" y="447" text-anchor="middle">audit labels cover both branches</text>
		<path d="M91 384H35V466Q35 474 43 478L118 481" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fraud-label-arrow)"></path>
		<path d="M180 460V480" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fraud-label-arrow)"></path>
		<rect class="viz-node viz-node--output" x="55" y="482" width="250" height="38" rx="4"></rect>
		<text class="viz-callout" x="180" y="498" text-anchor="middle">TRAIN + EVALUATE</text>
		<text class="viz-label" x="180" y="513" text-anchor="middle">retain label source and action</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> follow the solid operational path first. Only allowed payments can produce the ordinary chargeback outcome, and even a mature payment with no report is a noisy negative; a declined payment has no realized payment outcome at all. Then follow both dashed paths: independent review samples allowed and declined traffic, and its audit labels remain distinguishable from policy-selected outcome labels.</figcaption>
</figure>

## What an L5 answer sounds like

> "Two-layer architecture, common across the industry:
>
> **Layer 1: rules engine.** Hard-coded checks for known patterns (velocity limits, blacklists, country mismatches). Fast, explainable, easy to update when new patterns emerge. Catches the obvious.
>
> **Layer 2: ML model.** Gradient-boosted trees (XGBoost, LightGBM) or a neural network on engineered features: transaction amount, time, merchant category, geographic distance, device fingerprint, account age, recent transaction history. Outputs a fraud score; threshold determines action (allow, challenge, block).
>
> **Features**:
> - Transaction-level (amount, currency, merchant, time).
> - User-level aggregates (transactions per day/week, total spend, location entropy).
> - Network features (shared device, shared IP, shared payment instrument across accounts).
> - Sequence features (recent transaction velocity, deviation from typical pattern).
>
> **Eval**: precision-recall curve at multiple operating thresholds. Pick the threshold based on the business cost trade-off (cost per false positive vs cost per missed fraud), not on F1.
>
> **Online**: shadow mode for new models before they make decisions; A/B against the current model on a fraction of traffic with explicit business-metric tracking (chargeback rate, false-positive complaints)."

This is L5. Two-layer architecture, feature taxonomy, eval framework.

## What an L6 answer adds

> "...the things that make fraud hard in production:
>
> **Label bias is the dominant problem.** You only have labels for fraud you caught. Negatives in training data include actual frauds you missed. This biases the model toward the existing system's blindspots. Mitigations: semi-supervised methods (use unlabeled transactions in an auxiliary objective), active learning to invest expensive label investigation in uncertain cases, periodic 'random sample' investigations to estimate true fraud base rate.
>
> **Adversarial drift demands continuous retraining.** Frequency of retraining: weekly to monthly for traditional models. Plus an explicit detection-and-respond pipeline for sudden new attack patterns.
>
> **Concept drift detection**: monitor model score distributions, feature distributions, and per-segment performance over time. Alert on significant shifts.
>
> **Threshold tuning is a business decision, not an ML decision.** A threshold that catches 95% of fraud at 0.5% false positive rate may be the right answer; or 80% at 0.1% false positive rate may be. The risk team owns the decision; the ML team enables it with calibrated probabilities.
>
> **Calibration matters more than usual.** Downstream actions (auto-block at score > 0.95, manual review at 0.7-0.95, allow at < 0.7) require well-calibrated probabilities, not just a ranking. Apply Platt scaling or isotonic regression post-hoc.
>
> **Model explainability is a regulatory requirement** in some jurisdictions. SHAP values per decision, model documentation for regulators.
>
> **Network / graph features matter increasingly.** Fraudsters operate in connected rings (shared devices, shared addresses). Graph features (shortest path to known fraud accounts, community detection) catch ring fraud that single-account models miss."

## Tells that get you a strong-hire vote

- You name the **two-layer rules + ML** architecture explicitly.
- You bring up **label bias** as the dominant statistical problem.
- You discuss **threshold tuning as a business decision**.
- You insist on **calibration**.
- You name **graph / network features** for ring fraud.
- You have an **incident-response and continuous-retraining** plan.

## Tells that get you down-leveled

- ML alone with no rules engine.
- Reporting F1 instead of precision-recall curve.
- No mention of label bias.
- No drift detection.
- Treating fraud detection as a static problem.

## Common follow-up

"How would you debias the labels?"

The L6 answer:

> "No silver bullet, but two reasonable approaches. (1) Periodic random investigation: sample some declined and some allowed transactions; have analysts label them; use the labeled sample to estimate the true fraud rate and calibrate model scores. (2) Semi-supervised techniques: use unlabeled transactions in an auxiliary self-supervised task (e.g., masked-feature reconstruction); the model learns 'normal' transaction structure from unlabeled data, and labels supervise the discriminative head. The semi-supervised approach helps when labels are sparse and noisy; in fraud both apply."

---

*Related: [delayed and selective labels](/concepts/delayed-labels-selective-labels-feedback-loops/), [decision thresholds and abstention](/concepts/decision-thresholds-asymmetric-costs-abstention/), [class imbalance](/questions/class-imbalance/), and [calibration](/concepts/calibration/).*
