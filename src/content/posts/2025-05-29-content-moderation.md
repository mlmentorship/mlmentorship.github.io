---
title: "Design a content moderation system"
description: "Moderation is a multi-policy classification problem at scale, with appeals, human review, and adversarial users. The senior answer separates policy from model and treats human review as part of the system."
date: "2025-05-29"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: trust-and-safety, social-platform, and LLM-deployment interviews.*

Design a content-moderation system that applies explicit policies while accounting for uncertainty, review capacity, and user appeals.

## Problem

**Context.** A hypothetical platform must assess incoming content under several policies, with different error costs across categories and languages.
Automated scores are imperfect, and human review has limited capacity.

**Task.** Define policy labels, decision thresholds, automated actions, review and appeal paths, and a measurement and rollout plan.
Explain how policy changes and adversarial behavior affect operation.

**Constraints.** Separate policy authority from model prediction. Specify false-positive and false-negative costs rather than optimizing accuracy alone.
Protect user data, audit decisions, and provide a response when uncertainty exceeds the system's authority.

## What makes moderation hard

1. **Policy is fuzzy and contested.** "Hate speech" has no universal definition; the platform's policy is a specification.
2. **Users adapt.** What's flagged today is rephrased tomorrow.
3. **High-stakes both ways.** Over-moderation suppresses legitimate speech; under-moderation harms users.
4. **Scale.** Billions of items; can't human-review everything.
5. **Multi-policy.** Hate speech, sexual content, violence, spam, fraud, copyright are different tasks with different policies and costs.

## What an L5 answer sounds like

> "Architecture, in layers:
>
> **Layer 1: triage by score.** A multi-task classifier (or one model per policy) outputs scores for each policy violation. Most content is low-risk and gets allowed. High-risk content is flagged.
>
> **Layer 2: action determination.** Per policy, decide what to do based on the score:
> - Below low threshold: allow.
> - Between low and high thresholds: route to human review.
> - Above high threshold: auto-action (remove, reduce reach, age-gate, depending on policy).
>
> **Layer 3: human review.** Trained moderators review the routed content. Their decisions feed back as labels for retraining.
>
> **Layer 4: appeals.** Users can appeal automated actions; appeals route to senior reviewers; final decisions feed back as labels.
>
> **Eval**: per-policy precision and recall at multiple thresholds. Calibrate against trained moderator decisions on a held-out set. A/B-test policy threshold changes with explicit business and harm-floor metrics."

This is L5. Layered architecture with human-in-the-loop and feedback.

## What an L6 answer adds

> "...practical things:
>
> **Policy and model are separate concerns.** The platform decides policy; ML implements it. Conflating the two creates ML teams that look like they're making policy decisions (which they shouldn't be unilaterally). Document the policy in plain language; build the model to that spec; track 'model agreement with policy spec' as a metric, not 'model agreement with current model behavior.'
>
> **Per-policy thresholds, not a single 'bad' threshold.** Different policies have different cost structures. Hate speech: high cost of false negative, willing to tolerate more false positives. Spam: high volume, lower cost per error, optimize differently.
>
> **Adversarial robustness.** Users will test the system, find phrasings that bypass it, and share. Augment training data with known evasion patterns; track 'evasion rate' (flagged content that mutates to unflagged versions); maintain a fast-update path for new patterns.
>
> **Audit relevant slices.** Measure probability quality, false positives, false negatives, and policy consistency across languages and other appropriate slices. An accuracy gap is not itself a calibration diagnosis. Group-specific calibration or thresholds require a justified policy and an assessment of legal, privacy, fairness, and data-support constraints.
>
> **Route review by risk and uncertainty.** Near-threshold cases can be useful, but a confidently wrong extreme can also be consequential. Reserve some representative and risk-targeted audits outside the borderline queue; measure reviewer agreement and capacity.
>
> **Counterfactual eval for false negatives.** False positives are visible (users complain). False negatives are invisible by definition (the bad content stayed up). Track via periodic random audits and via reports from users / partners.
>
> **Appeals are valuable but selected evidence.** Appealers differ from users who cannot or do not appeal, and reviewers can disagree. Version the policy and adjudication, audit label quality, and keep representative sampling separate. As in the [fraud-design lesson](/questions/design-fraud-detection/), observed outcomes depend on the policy; appeal accuracy alone cannot estimate population error rates.
>
> **Explainability is a regulatory and trust issue.** When auto-actioning content, surface a reason ('flagged for hate speech against [protected category]'); helps user understanding and reduces disputes."

<!-- visual:content-moderation-governed-decision-loop -->
<figure class="learning-figure editorial-illustration" aria-labelledby="content-moderation-governed-decision-loop-edition-title"><p class="visual-title" id="content-moderation-governed-decision-loop-edition-title">A score selects an action only through a declared policy</p><svg viewBox="0 0 360 345" role="img" aria-labelledby="content-moderation-governed-decision-loop-edition-svg-title content-moderation-governed-decision-loop-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="content-moderation-governed-decision-loop-edition-svg-title">A score selects an action only through a declared policy</title><desc id="content-moderation-governed-decision-loop-edition-svg-desc">Under explicitly illustrative policy P7, score 0.62 falls between review thresholds 0.40 and 0.80 and therefore enters human review. The corrected decision retains policy version and reviewer provenance. Allowed-item audits and appeals also contribute corrected records.</desc><text x="180" y="23" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Illustrative policy P7; not universal</tspan><tspan x="180" dy="19">thresholds</tspan></text><text x="180" y="65" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">One item receives score 0.62</tspan></text><rect x="25" y="98" width="124" height="27" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><rect x="149" y="98" width="124" height="27" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><rect x="273" y="98" width="62" height="27" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="25" y="146" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="25" dy="0">0</tspan></text><text x="149" y="146" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="149" dy="0">0.40</tspan></text><text x="273" y="146" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="273" dy="0">0.80</tspan></text><text x="335" y="146" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="335" dy="0">1</tspan></text><path d="M217.2 70L217.2 98" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M213.84402122977056,91.8569220667674L217.2,98L220.55597877022942,91.8569220667674" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><text x="180" y="183" style="font-size:13px;font-weight:700;fill:var(--viz-focus-stroke);text-anchor:middle"><tspan x="180" dy="0">0.40 &lt;= 0.62 &lt; 0.80: HUMAN REVIEW</tspan></text><g data-editorial-node><rect x="30" y="211" width="300" height="63" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="180" y="238" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Decision + policy P7 + reviewer</tspan><tspan x="180" dy="19">provenance</tspan></text></g><text x="180" y="307" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Allowed-item audits and appeals also update</tspan><tspan x="180" dy="19">the record.</tspan></text></svg><figcaption><strong>Read it this way:</strong> In this illustrative policy, 0.62 enters human review, not automatic action. Audits and appeals preserve policy and reviewer provenance. These thresholds are not universal. <span class="visual-reader-check" data-reader-check="content-moderation-governed-decision-loop"><strong>Check:</strong> How can a missed violation on the allow path become correction evidence?</span></figcaption></figure>

## Tells that get you a strong-hire vote

- You **separate policy from model**.
- You name **per-policy thresholds** rather than one global score.
- You make **human review and appeals** part of the architecture.
- You bring up **adversarial robustness** with continuous retraining.
- You insist on **per-slice fairness eval**.
- You name **borderline routing** as the high-value review queue.

## Tells that get you down-leveled

- One classifier, one threshold.
- No human-in-the-loop or appeals process.
- No mention of adversarial drift.
- No fairness slicing.

## Common follow-up

"How do you handle policy changes? E.g., the company updates its definition of hate speech."

The L6 answer:

> "Several layers. First, the policy spec is a versioned document; changes go through a review process and have an effective date. Second, retrain the model on data labeled under the new policy (re-label a held-out set first; if old labels are now misclassified, retrain; otherwise just patch the rules layer). Third, track 'agreement with current policy' over time as a metric; if the model drifts from policy faster than retraining can keep up, that's a flag for accelerated retraining cadence. Fourth, communicate the change to moderators and users via release notes; appeals processed under the old policy may need re-review."

---

*Related: [How do you handle hallucinations in production?](/questions/handle-hallucinations-in-production/), [How do you deal with class imbalance in 2026?](/questions/class-imbalance/), [A/B testing for ML systems](/concepts/ab-testing-for-ml/).*
