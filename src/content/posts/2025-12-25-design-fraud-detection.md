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

*Related: [How do you deal with class imbalance in 2026?](/questions/class-imbalance/), [Calibration](/concepts/calibration/), [A/B testing for ML systems](/concepts/ab-testing-for-ml/).*
