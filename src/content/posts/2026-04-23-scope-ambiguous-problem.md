---
title: "How do you scope an ambiguous problem?"
description: "Scoping is a core senior skill. The interview tests how you reduce ambiguity before choosing a model or system design."
date: "2026-04-23"
draft: false
tags: ["questions"]
category: "questions"
---


Turn a symptom into a bounded decision before committing to a model.
A prototype is useful when it resolves the deciding uncertainty.

## Problem

A requester wants a larger generator because answers are poor.
In this fictional triage, 50 selected complaint cases have disjoint assigned causes:

| Assigned cause | Cases |
| --- | ---: |
| Missing source information | 30 |
| Retrieval omission | 15 |
| Answer synthesis | 5 |

What should the first investigation change? Which conclusion cannot be drawn from these counts?

## Picture: Find the Decision-Changing Unknown

<!-- visual:ambiguous-problem-scope-loop -->
```mermaid
%%{init: {"themeVariables": {"fontSize": "19px"}, "flowchart": {"nodeSpacing": 12}}}%%
flowchart TB
	accTitle: A selected complaint sample guides a bounded scoping test
	accDescr: The fictional sample contains 30 missing-source, 15 retrieval and five synthesis cases. These are selected complaint categories, not population rates. Validate their assignment, identify the affected user outcome and define scope, constraints and non-goals. Check authoritative source availability before assuming a larger generator addresses the dominant assigned cause. A bounded test can support v1, rescope, select a non-ML path or stop; its owner records evidence and a decision date.
	A["50 selected complaints<br/>30 source · 15 retrieval<br/>5 synthesis"] --> B["Validate triage + sample<br/>users · costs · context"]
	B ==> C["Actual outcome<br/>what must change for whom?"]
	C --> D["Define the boundary<br/>success · guardrails · data<br/>in · out · deferred"]
	D --> E{"Model change justified<br/>by available evidence?"}
	E -->|"no"| F["Instrumentation / non-ML<br/>baseline or stop"]
	E ==>|"yes"| G["One-page decision<br/>scope · owner · timebox<br/>test · open questions"]
	G --> H["Key uncertainty<br/>required facts available?"]
	H ==> I["Bounded availability check<br/>decision rule set in advance"]
	I --> J{"What did the<br/>evidence change?"}
	J -->|"supports"| K["Proceed to v1<br/>with agreed scope"]
	J -. "changes assumptions" .-> D
	J -->|"invalidates"| L["Stop or choose<br/>a different approach"]
	class A viz-input
	class A viz-tall
	class C,E,H,J viz-focus
	class D,G viz-state
	class F,L viz-warning
	class I,K viz-output
```
<p class="diagram-caption"><strong>Read it this way:</strong> the request names a component, not yet a scoped problem. Define the decision and evidence before choosing v1, a non-ML baseline, further discovery or stop.</p>

## Work the Triage

$30+15+5=50$. Under this taxonomy, a generator-only change directly targets the five synthesis cases.
The other 45 motivate source/retrieval investigation, not an assumed larger-model fix.

First validate the categories and source availability against an authoritative version.
If required facts are absent but permitted to obtain, scope ingestion; if already available, revisit retrieval and triage.
If necessary inputs cannot lawfully be obtained, narrow the task or stop.

The sample estimates neither overall failure prevalence nor causal improvement.
Real failures can overlap; the disjoint categories are an exercise assumption.

## Make the Decision Artifact

Name the affected population, outcome/guardrails, inputs/outputs/labels, data quality, owner, timebox and deciding uncertainty.
Record in-scope work, explicit non-goals, deferred work and the result that means proceed, rescope or stop.
Agree a decision date and review authority; duration depends on risk and evidence cost, not a universal week.

## Change 1: Labels or Logging Are Missing

**New task.** Reliable outcome evidence does not exist.

**Picture and change.** Scope instrumentation, representative labeling or a smaller decision first.

**Check.** A convenient proxy does not automatically identify the intended outcome.

**Use when.** Evidence availability is the biggest unknown.

## Change 2: A Quality Promise Is Demanded

**New task.** Stakeholders request a guarantee before discovery.

**Picture and change.** Give conditional ranges, assumptions, a decision date and staged scope; escalate incompatible constraints honestly.

**Check.** Do not say yes externally while hiding an incompatible internal plan.

**Use when.** Commitment pressure conflicts with unresolved uncertainty.

## Failure: Architecture Without a Stop Rule

An elaborate design does not define non-goals, required evidence or when discovery should end.

## Transfer Check

When is coding a valid first scoping action?

### Reasoned Answer

When a bounded prototype targets a named uncertainty and its result changes scope or allocation.
It need not build the complete system.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

<span id="what-scoping-an-ambiguous-problem-means"></span>
## Six Scoping Steps

1. Ask requesters about actual users, outcomes and failure costs.
2. Define the decision, inputs, outputs, truth and evaluation; allow non-ML.
3. Inspect data availability, quality, labels and permission.
4. Compare the simplest reasonable rule/model baseline.
5. Identify shift, noise, regulation, cost, latency and other invalidating risks.
6. Agree a bounded one-page decision with the responsible stakeholders.

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-sounds-like"></span>
<span id="what-an-l6-answer-adds"></span>
## Use a Real Incident

State your actual negotiation, framing and test contribution, what changed and what remained uncertain.
The hypothetical triage is not a story to claim as personal experience.

<span id="tells-that-get-you-a-strong-hire-vote"></span>
## Check the Scope

Can another person reconstruct the boundary, baseline, biggest unknown, test, owner and end date?
Make v1, later work and exclusions explicit; document open questions instead of concealing them.

<span id="tells-that-get-you-down-leveled"></span>
## Missing Evidence

Add the missing measurement, risk, non-goal or decision rule before assuming model choice answers the request.

## Common follow-up

If asked for quality before discovery, state assumptions and the evidence needed to narrow the range.
Distinguish an estimate, a target and an accepted commitment.

## Sources

- [GOV.UK discovery](https://www.gov.uk/service-manual/agile-delivery/how-the-discovery-phase-works).
- [GOV.UK alpha](https://www.gov.uk/service-manual/agile-delivery/how-the-alpha-phase-works).
- [Google ML problem framing](https://developers.google.com/machine-learning/problem-framing/problem).

---

*Related: [The 5 things every applied scientist interview is testing for](/guides/five-things-as-interview-tests/), [senior through senior-principal ML scope](/guides/l5-vs-l6-faang-ml/), [How do you decide what to work on?](/questions/decide-what-to-work-on/).*

</div>
