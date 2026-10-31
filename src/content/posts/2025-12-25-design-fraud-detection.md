---
title: "Design fraud detection for a payment company"
description: "Design payment-risk decisions while preserving action, label maturity, audit provenance, point-in-time features and the costs of mistakes."
date: "2025-12-25"
draft: false
tags: ["questions"]
category: "questions"
---


Design payment-fraud decisions and evaluation when outcomes arrive late and the current policy determines which payments can produce labels.

## Problem

**Context.** A hypothetical payment policy allows some attempts and declines others; allowed outcomes mature only after a declared observation window.
The cohort below contains mature outcomes, pending outcomes, and declined attempts.

**Task.** Define labels, model and policy inputs, operating actions, and evaluation for the cohort.
Explain what can be estimated now and what evidence is needed to compare a changed decision policy.

**Constraints.** Missing, pending, and declined outcomes are not confirmed legitimate transactions.
Separate fraud risk from action costs, preserve decision-time feature availability, and account for selection and label delay.

Design payment-risk decisions for this hypothetical cohort of 1,000 attempts under policy P1.
P1 allows 900 and declines 100.
Of the allowed payments, 800 have completed the declared observation window; eight have confirmed fraud reports.
The other 100 allowed payments remain pending.

Define training labels, operating actions and evaluation without converting missing outcomes into legitimate transactions.

## Picture: Follow the Action Before Naming the Label

| Population | Count | What is observed |
| --- | --- | --- |
| Mature allowed | 800 | Eight confirmed reports; 792 without reports in the window |
| Pending allowed | 100 | Observation window incomplete; not negative |
| Declined | 100 | No realized outcome from allowing that payment |

The confirmed-report fraction is $8/800=1\%$ among mature allowed payments.
It is not population fraud prevalence.
The 792 no-report records are negative only under the declared, noisy reporting definition.
The ratio $8/1000=0.8\%$ describes currently confirmed reports per attempt, not complete fraud coverage.

Keep action, maturity, outcome source and uncertainty attached to every label.
Independent investigation can provide evidence about declined attempts.
It cannot observe their exact losses under an action that never occurred.

<!-- visual:fraud-policy-label-coverage -->
<figure class="learning-figure plot-panel" aria-labelledby="fraud-policy-label-title">
	<p class="visual-kicker">One cohort, three observation states</p>
	<p class="visual-title" id="fraud-policy-label-title">Declined and pending payments do not supply observed negative outcomes</p><div class="visual-argument" data-visual-argument="fraud-policy-label-coverage" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 240" role="img" aria-label="The policy changes which payment outcomes become observable. A declined payment has no observed repayment outcome. Pending and declined cases cannot be silently labeled as legitimate negatives."><title>The policy changes which payment outcomes become observable</title><desc>A declined payment has no observed repayment outcome. Pending and declined cases cannot be silently labeled as legitimate negatives.</desc><text x="154.24" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="154.24" dy="0">outcome</tspan></text><text x="278.08" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="278.08" dy="0">label</tspan></text><text x="80.32" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="80.32" dy="0">approved</tspan></text><g data-proof-status="present"><rect data-proof-cell="0,0" x="92.32" y="36" width="118.84" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><path d="M144.74 54l5 5 10-12" style="fill:none;stroke:var(--c-text);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path><text x="151.74" y="82" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="151.74" dy="0">yes</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="0,1" x="216.16" y="36" width="118.84" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="275.58" y="71" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="275.58" dy="0">0/1</tspan></text><path d="M221.16 89L330 89" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><text x="80.32" y="118" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="80.32" dy="0">pending</tspan></text><g data-proof-status="unknown"><rect data-proof-cell="1,0" x="92.32" y="100" width="118.84" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><circle cx="151.74" cy="118" r="10" style="fill:none;stroke:var(--c-text);stroke-width:1.75;stroke-dasharray:2 3"></circle><text x="151.74" y="146" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="151.74" dy="0">not yet</tspan></text></g><g data-proof-status="unknown"><rect data-proof-cell="1,1" x="216.16" y="100" width="118.84" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><circle cx="275.58" cy="118" r="10" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:2 3"></circle><text x="275.58" y="146" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="275.58" dy="0">?</tspan></text><path data-proof-warning="1,1" d="M220.16 108v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g><text x="80.32" y="182" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="80.32" dy="0">declined</tspan></text><g data-proof-status="absent"><rect data-proof-cell="2,0" x="92.32" y="164" width="118.84" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><circle cx="151.74" cy="182" r="9" style="fill:var(--viz-neutral-bg);stroke:var(--c-text);stroke-width:1.75"></circle><path d="M146.74 182L156.74 182" style="fill:none;stroke:var(--c-text);stroke-width:1.75"></path><text x="151.74" y="210" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="151.74" dy="0">no</tspan></text></g><g data-proof-status="unknown"><rect data-proof-cell="2,1" x="216.16" y="164" width="118.84" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><circle cx="275.58" cy="182" r="10" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:2 3"></circle><text x="275.58" y="210" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="275.58" dy="0">?</tspan></text><path data-proof-warning="2,1" d="M220.16 172v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 530" role="img" aria-labelledby="fraud-policy-label-svg-title fraud-policy-label-svg-desc">
		<title id="fraud-policy-label-svg-title">Fraud actions create different label coverage</title>
		<desc id="fraud-policy-label-svg-desc">Policy P1 allows 900 of 1,000 attempted payments and declines 100. Among allowed payments, 800 are mature: eight have confirmed reports and 792 have no report under the noisy reporting definition. Another 100 remain pending, not negative. Declined payments have no realized allowed-payment outcome. Dashed paths sample both actions for independent adjudication. Audit labels retain their source and do not reveal exact counterfactual payment losses.</desc>
		<defs><marker id="fraud-label-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
		<rect class="viz-node viz-node--input" x="55" y="12" width="250" height="42" rx="4"></rect>
		<text class="viz-callout" x="180" y="30" text-anchor="middle">1,000 ATTEMPTED PAYMENTS</text>
		<text class="viz-label" x="180" y="46" text-anchor="middle">same decision-time population</text>
		<path d="M180 56V74" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fraud-label-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="55" y="80" width="250" height="42" rx="4"></rect>
		<text class="viz-callout" x="180" y="98" text-anchor="middle">P1: RULES + MODEL</text>
		<text class="viz-label" x="180" y="114" text-anchor="middle">policy chooses the action</text>
		<path d="M180 124V140H91V154" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fraud-label-arrow)"></path>
		<path d="M180 140H269V154" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fraud-label-arrow)"></path>
		<rect class="viz-node" x="14" y="160" width="154" height="50" rx="4"></rect>
		<text class="viz-callout" x="91" y="180" text-anchor="middle">ALLOW: 900</text>
		<text class="viz-label" x="91" y="198" text-anchor="middle">800 mature + 100 pending</text>
		<rect class="viz-node" x="192" y="160" width="154" height="50" rx="4" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect>
		<text class="viz-callout" x="269" y="180" text-anchor="middle">DECLINE: 100</text>
		<text class="viz-label" x="269" y="198" text-anchor="middle">unobserved allowed outcome</text>
		<path d="M91 212V230" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#fraud-label-arrow)"></path>
		<rect class="viz-node" x="14" y="236" width="154" height="72" rx="4"></rect>
		<text class="viz-callout" x="91" y="254" text-anchor="middle">OUTCOME WINDOW</text>
		<text class="viz-label" x="26" y="273">8 confirmed reports</text>
		<text class="viz-label" x="26" y="290">792 no report: noisy</text>
		<text class="viz-label" x="26" y="303">100 pending: unknown</text>
		<rect class="viz-node" x="192" y="236" width="154" height="72" rx="4" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect>
		<text class="viz-callout" x="269" y="254" text-anchor="middle">NO OUTCOME</text>
		<text class="viz-label" x="269" y="275" text-anchor="middle">Do not label as</text>
		<text class="viz-label" x="269" y="292" text-anchor="middle">legitimate</text>
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
	<div class="visual-derivation-caption">follow the 1,000 attempts to their actions. Of 900 allowed payments, 800 matured and eight have confirmed reports. The 100 pending payments are not negative. The 100 declines have no realized allowed-payment outcome. Dashed paths sample both actions for independent review. Audit labels retain their source, not exact counterfactual losses.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="fraud-policy-label-coverage"><strong>Check:</strong> Why must the 100 declined outcomes and 100 pending outcomes not become negative labels?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="fraud-policy-label-coverage">A declined payment has no observed repayment outcome. Pending and declined cases cannot be silently labeled as legitimate negatives.</span></figcaption>
</figure>

<span id="what-an-l5-answer-sounds-like"></span>

## Define the Decision Path

Start with approved rules plus a measured model baseline, such as gradient-boosted trees on point-in-time features.
Use transaction amount/time, account aggregates, recent velocity and permitted device/network relationships.
Every feature must have been available before its decision deadline.

A challenge is intermediate, not necessarily a declined payment.
Record passed, failed or abandoned challenges and their final allow/decline actions.
Challenge failure alone does not prove fraud.

```text
Decision
	transaction_id, decision_id, event_time, decision_time
	policy_version, action, challenge_state, feature_version
	availability_cutoff, score, calibration_population
	latency, fallback_reason
Outcome
	transaction_id, event_time, available_time, observation_window
	maturity, report_status, label_source, uncertainty
Audit
	decision_id, sampling_stratum, selection_probability
	adjudication_task, evidence_refs, reviewer, available_time
```

## Choose Actions From Consequences

Suppose a separately validated probability $p$ and the following simplified loss model apply.

| Action | Expected loss in currency units | Required evidence |
| --- | --- | --- |
| Allow | $100p$ | Relevant probability of loss under allowance |
| Decline | $3(1-p)$ | Cost of rejecting a legitimate payment |
| Challenge | Friction/fee plus residual loss | Completion, abandonment, fraud and capacity estimates |

With only allow/decline, $p=0.02$ gives losses 2 versus 2.94; allow has lower modeled loss.
At $p=0.10$, losses are 10 versus 2.7; decline has lower modeled loss.
The cohort's observed report fraction is not automatically a valid $p$ for this decision.

The risk owner approves costs, constraints and capacity; ML supplies justified estimates and uncertainty.
Calibration must name its target, population and observation process.
PR-AUC describes ranking, not the action policy.

## Change 1: The New Policy Declines More Payments

**New task.** P2 declines more high-risk attempts than P1.

**Picture and change.** Recompute action coverage and compare equally mature cohorts.
Use appropriately sampled audits and validated causal designs where feasible before claiming better population detection.
Preserve propensity/support assumptions; audits do not supply every missing counterfactual.

**Check.** Blocking nearly everyone can reduce allowed losses while destroying legitimate approvals.
Track approval, challenge friction, loss, review capacity and customer impact together.

**Use when.** The policy changes which outcomes can become labels.

## Change 2: Shared-Device Activity Rises

**New task.** Related accounts begin sharing more devices.

**Picture and change.** Verify event timing, velocity, graph construction and source integrity.
Inspect legitimate shared-device cohorts, privacy constraints and segment effects before changing actions.
Use bounded approved rules or investigation while testing the suspected pattern.

**Check.** Shared infrastructure is evidence, not proof of fraud.

**Use when.** Adaptive or correlated behavior changes feature meaning.

<span id="tells-that-get-you-down-leveled"></span>

## Failure: Missing Becomes Legitimate

Combining declined, pending and mature no-report transactions as negatives hides different observation processes.
Semi-supervised reconstruction can learn structure; it does not identify missing allowed-payment outcomes by itself.

## Transfer Check

Does high PR-AUC identify the correct blocking threshold?

### Reasoned Answer

No. Use relevant calibrated evidence, consequence/friction costs, policy constraints and operating capacity.
Check the resulting policy, not only its score ranking.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

<span id="why-fraud-is-uniquely-hard"></span>
<span id="common-follow-up"></span>

## Audit the Observation Process

Outcome delay, reporting noise, selection and adaptive behavior require separate diagnostics.
Fraud prevalence and available latency vary by product; do not substitute generic industry rates.

Sample allowed and declined traffic using recorded strata and selection probabilities.
Define the adjudication target, independent evidence, reviewer uncertainty and any weighting assumptions.
Active-learning cases are not automatically representative samples.
An audit can estimate its defined target under justified assumptions, not every unobserved monetary loss.

Use time-based splits, decision-time feature snapshots and mature outcome windows.
Prevent future reports or later graph labels from leaking into earlier features.
Report precision/recall, calibration, losses and uncertainty for the population each dataset supports.

<span id="what-an-l6-answer-adds"></span>
<span id="tells-that-get-you-a-strong-hire-vote"></span>

## Operate a Changing Policy

Shadow scoring tests service behavior without changing actions; it does not reveal alternative payment outcomes.
Use approved staged experiments only where risk, customer protection and identification assumptions permit them.

Monitor feature/score changes, maturity, action mix, segment outcomes and service deadlines.
A distribution shift is a diagnostic signal, not proof that retraining solves the incident.
Choose update cadence from label readiness and validated operating needs.

Name risk-policy, fraud-operations, model and service owners.
Define temporary-rule expiry, incident escalation, fallback actions and model/policy rollback separately.
Revalidate after changing features, calibration, thresholds or label definitions.
Platt scaling or isotonic regression still requires suitable data and independent evaluation.

Explanation, retention and access requirements depend on jurisdiction, product and current rules.
Feature attributions can support investigation; they are not automatically a compliant explanation.

*Related: [delayed and selective labels](/concepts/delayed-labels-selective-labels-feedback-loops/), [decision thresholds and abstention](/concepts/decision-thresholds-asymmetric-costs-abstention/), [class imbalance](/questions/class-imbalance/), and [calibration](/concepts/calibration/).*

</div>
