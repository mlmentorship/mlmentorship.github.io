---
title: "Defend a values decision under ethical pressure"
description: "Explain a real values decision using prior evidence, competing concerns, obligations, ownership and observed outcomes without rewriting the record."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

Explain a values-based decision under pressure using the obligations, evidence, and alternatives available when the choice was made.

## Problem

**Context.** An interviewer asks about a real decision under ethical pressure; the fictional risk exercise below probes reasoning about outcomes.
Its two options are explicitly permitted and reversible.

**Task.** Explain the decision process, authority, escalation, and evidence, then assess what a single incident-free trial establishes.
Distinguish the quality of the original decision from the eventual observed outcome.

**Constraints.** Mandatory legal, safety, or privacy obligations are not interchangeable with a negotiable expected-risk calculation.
Use your own experience for personal claims; one outcome cannot validate a probability estimate or reveal the unchosen counterfactual.

This fictional exercise compares two permitted, reversible options:

| Option | Estimated severe-incident probability per trial |
| --- | ---: |
| A | 0.01 |
| B | 0.02 |

Only one option is tried once; no incident occurs.
Does that validate its estimate, establish the other option's outcome or prove the decision process was sound?

## Picture: Process and Outcome Are Different Axes

<!-- visual:ethical-decision-process-outcome -->
<figure class="learning-figure plot-panel" aria-labelledby="ethical-decision-title">
	<p class="visual-kicker">Process and outcome</p>
	<p class="visual-title" id="ethical-decision-title">How should an outcome change your judgment of a values decision?</p>
	<svg viewBox="0 0 360 430" role="img" aria-labelledby="ethical-decision-svg-title ethical-decision-svg-desc">
		<title id="ethical-decision-svg-title">Decision process and observed outcome are evaluated separately</title>
		<desc id="ethical-decision-svg-desc">A two-by-two matrix crosses the quality of the decision process, weak or sound based on evidence available at the time, with the observed outcome, favorable or adverse. A weak process with a favorable outcome is labelled lucky result and should not be rewarded. A sound process with a favorable outcome is labelled supported result. A weak process with an adverse outcome is labelled exposed flaw and requires correcting the process. A sound process with an adverse outcome is labelled adverse result and should not be condemned using hindsight. An arrow from all outcomes leads to a separate box: use the outcome as new evidence to update assumptions, controls, and the next decision, not to rewrite what was knowable before.</desc>
		<defs>
			<marker id="ethical-decision-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
		</defs>
		<rect class="viz-plot-bg" x="4" y="4" width="352" height="422" rx="5"></rect>
		<text class="viz-axis-label" x="206" y="22" text-anchor="middle">EVIDENCE AVAILABLE AT THE TIME</text>
		<text class="viz-callout" x="139" y="43" text-anchor="middle">WEAK PROCESS</text>
		<text class="viz-callout" x="275" y="43" text-anchor="middle">SOUND PROCESS</text>
		<text class="viz-axis-label" x="10" y="98">FAVORABLE</text>
		<text class="viz-axis-label" x="10" y="113">OUTCOME</text>
		<text class="viz-axis-label" x="10" y="230">ADVERSE</text>
		<text class="viz-axis-label" x="10" y="245">OUTCOME</text>
		<rect x="72" y="54" width="132" height="118" rx="4" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5;stroke-dasharray:6 4"></rect>
		<text class="viz-callout" x="138" y="82" text-anchor="middle">LUCKY RESULT</text>
		<text class="viz-label" x="138" y="108" text-anchor="middle">Weak reasoning</text>
		<text class="viz-label" x="138" y="125" text-anchor="middle">happened to work</text>
		<text class="viz-axis-label" x="138" y="151" text-anchor="middle">Do not reward luck</text>
		<rect class="viz-node viz-node--output" x="208" y="54" width="132" height="118" rx="4"></rect>
		<text class="viz-callout" x="274" y="82" text-anchor="middle">SUPPORTED RESULT</text>
		<text class="viz-label" x="274" y="108" text-anchor="middle">Sound reasoning</text>
		<text class="viz-label" x="274" y="125" text-anchor="middle">and a good result</text>
		<text class="viz-axis-label" x="274" y="151" text-anchor="middle">Recheck assumptions</text>
		<path d="M72 186 H204 V304 H72 Z" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></path>
		<text class="viz-callout" x="138" y="214" text-anchor="middle">EXPOSED FLAW</text>
		<text class="viz-label" x="138" y="240" text-anchor="middle">Weak reasoning</text>
		<text class="viz-label" x="138" y="257" text-anchor="middle">failed visibly</text>
		<text class="viz-axis-label" x="138" y="283" text-anchor="middle">Correct the process</text>
		<rect class="viz-node viz-node--focus" x="208" y="186" width="132" height="118" rx="4"></rect>
		<text class="viz-callout" x="274" y="214" text-anchor="middle">ADVERSE RESULT</text>
		<text class="viz-label" x="274" y="240" text-anchor="middle">Sound reasoning</text>
		<text class="viz-label" x="274" y="257" text-anchor="middle">met a bad result</text>
		<text class="viz-axis-label" x="274" y="283" text-anchor="middle">Do not use hindsight</text>
		<path class="viz-forward" style="marker-end:url(#ethical-decision-arrow)" d="M206 304 V334"></path>
		<rect class="viz-node viz-node--input" x="48" y="334" width="292" height="74" rx="5"></rect>
		<text class="viz-callout" x="194" y="358" text-anchor="middle">UPDATE THE NEXT DECISION</text>
		<text class="viz-label" x="194" y="381" text-anchor="middle">Update assumptions, controls, and thresholds</text>
		<text class="viz-axis-label" x="194" y="399" text-anchor="middle">Do not rewrite what was knowable before</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> judge process using evidence available then. Judge the observed result separately. Use outcomes to update controls, not to rewrite the record. <span class="visual-reader-check" data-reader-check="ethical-decision-process-outcome"><strong>Check:</strong> Which facts may judge the original decision? Which arrived too late to have informed it?</span></figcaption>
</figure>

## Work the Observation

Under the stated Bernoulli forecasts, one-trial no-incident probabilities are $1-0.01=0.99$ and $1-0.02=0.98$.
A favorable outcome is plausible under either forecast.
It proves neither estimate; the unchosen option remains unobserved.

Inspect the original evidence, alternatives, authority, threat model and mitigation before judging the process.
An unsupported forecast does not become justified afterward.

## Change 1: A Considered Choice Turns Out Badly

**New task.** An adverse result dominates the review.

**Picture and change.** Inspect hidden assumptions and controls; judge the prior process from prior evidence.

**Check.** A bad result alone does not prove recklessness. It can still reveal a repairable weakness.

**Use when.** Hindsight replaces the decision record.

## Change 2: Waiting Also Harms Users

**New task.** Delay has serious consequences.

**Picture and change.** Compare permitted alternatives, severity, exposure, uncertainty, monitoring delay and reversibility.

**Check.** Neither always waiting nor always moving fast is a complete rule.
Legal, privacy and integrity duties are not optional utility weights.

**Use when.** Both action and inaction have costs.

## Failure: A Mission Slogan Replaces Evidence

Do not portray yourself as the only ethical person or invent sacrifice, uncertainty, cost or later vindication.
State unavailable evidence honestly.

## Transfer Check

Can a favorable outcome validate an unsupported original probability?

### Reasoned Answer

No. It is new evidence whose strength depends on the data-generating process, not retroactive justification.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## What makes the story real

Use a real decision and check six properties:

1. Represent competing legitimate concerns fairly.
2. Identify the decision you actually influenced.
3. State evidence and uncertainty available then.
4. Explain who bore a real cost, or what cost evidence is unavailable.
5. Separate outcome from automatic vindication.
6. Describe later judgment/process changes, or acknowledge that none has yet occurred.

## A claim-first opening

**Fictional illustration, not a personal script:** a one-week launch hold delays a milestone after a severe slice regression.
The engineer owns the hold proposal, targeted evaluation and revised gate, while acknowledging late escalation damaged trust.
The process improvement is earlier slice ownership, not a claim that delay is always correct.

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-adds"></span>
<span id="what-an-l6-answer-adds"></span>
## Evidence Record

Record concerns, obligations, contemporaneous evidence, alternatives, choice/owner, personal action, real cost, result and changed behavior.
Use approved records where possible; distinguish personal/team work and unobservable counterfactual harm.
Inspect incentives, missing slice ownership and escalation pressure, not just individual motives.

<span id="tells-that-get-you-a-strong-hire-vote"></span>
## Check the Account

Name reversal evidence and a concrete later control when one exists. Alignment need not mean agreement with every premise.

<span id="tells-that-get-you-down-leveled"></span>
## Missing Evidence

Add the actual tension, uncertainty or action rather than a company mission quote or polished moral victory.

## Common follow-up

When should speed outweigh caution? Compare allowed options, affected users, severity, residual risk and delay cost under explicit authority.

Use the [values practice packet](/prep/values/) with an observer who will challenge the premise.

*Related: [quality or safety over speed](/questions/advocated-quality-over-speed/), [disagree with someone senior](/questions/disagreed-with-senior/), and [values practice](/prep/values/).*

</div>
