---
title: "How do you scope an ambiguous problem?"
description: "Scoping is a core senior skill. The interview tests how you reduce ambiguity before choosing a model or system design."
date: "2026-04-23"
draft: false
tags: ["questions"]
category: "questions"
---


Turn an ambiguous request for a larger model into a bounded investigation with an explicit decision and success criterion.

## Problem

**Context.** A requester attributes poor generated answers to model size, while selected complaint cases have the invented cause assignments below.

**Task.** Clarify the desired outcome, identify the first deciding uncertainty, and propose a small investigation with predicted results.
State a success criterion, scope boundary, owner, and conditions for changing direction.

**Constraints.** The complaint sample is selected, not a representative estimate of all traffic.
Do not convert its counts into a causal claim that one intervention solves the whole product problem.

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
<figure class="learning-figure editorial-illustration" aria-labelledby="ambiguous-problem-scope-loop-edition-title"><p class="visual-title" id="ambiguous-problem-scope-loop-edition-title">Use a bounded check to decide scope, not the complaint sample alone</p><div class="visual-argument" data-visual-argument="ambiguous-problem-scope-loop" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 176" role="img" aria-label="A complaint sample proposes a scope; a bounded check tests it. The observed complaints concern one slice. A discriminating sample from other slices decides whether the proposed scope should widen."><title>A complaint sample proposes a scope; a bounded check tests it</title><desc>The observed complaints concern one slice. A discriminating sample from other slices decides whether the proposed scope should widen.</desc><text x="167.05" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="167.05" dy="0">evidence</tspan></text><text x="282.35" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="282.35" dy="0">affected</tspan></text><text x="97.39999999999999" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="97.39999999999999" dy="0">complaints</tspan></text><g data-proof-status="value"><rect data-proof-cell="0,0" x="109.39999999999999" y="36" width="110.30000000000001" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="164.55" y="71" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="164.55" dy="0">seen</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="0,1" x="224.7" y="36" width="110.30000000000001" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="279.85" y="71" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="279.85" dy="0">known</tspan></text><path d="M229.7 89L330 89" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><text x="97.39999999999999" y="118" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="97.39999999999999" dy="0">other</tspan></text><g data-proof-status="unknown"><rect data-proof-cell="1,0" x="109.39999999999999" y="100" width="110.30000000000001" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><circle cx="164.55" cy="118" r="10" style="fill:none;stroke:var(--c-text);stroke-width:1.75;stroke-dasharray:2 3"></circle><text x="164.55" y="146" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="164.55" dy="0">not yet</tspan></text></g><g data-proof-status="unknown"><rect data-proof-cell="1,1" x="224.7" y="100" width="110.30000000000001" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><circle cx="279.85" cy="118" r="10" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:2 3"></circle><text x="279.85" y="146" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="279.85" dy="0">?</tspan></text><path data-proof-warning="1,1" d="M228.7 108v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 428" role="img" aria-labelledby="ambiguous-problem-scope-loop-edition-svg-title ambiguous-problem-scope-loop-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="ambiguous-problem-scope-loop-edition-svg-title">A complaint sample suggests a question; a bounded check decides the next scope</title><desc id="ambiguous-problem-scope-loop-edition-svg-desc">A fictional selected complaint sample contains thirty source, fifteen retrieval, and five synthesis complaints. It is not a representative prevalence estimate. The named missing fact is whether permitted, current source evidence exists for ten sampled cases. A predeclared one-day availability check determines a bounded v1 source repair or a stop, rather than automatically proposing a model change.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional selected sample: 30 source / 15</tspan><tspan x="180" dy="19">retrieval / 5 synthesis</tspan></text><g transform="translate(0 19)"><g data-editorial-node><rect x="30" y="43" width="300" height="58" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="58" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Missing fact: permitted, current</tspan><tspan x="180" dy="19">evidence exists for 10 sampled</tspan><tspan x="180" dy="19">cases?</tspan></text></g><path d="M180 101L180 128" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,121.8569220667674L180,128L183.35597877022943,121.8569220667674" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="30" y="132" width="300" height="76" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="180" y="156" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">One-day availability check</tspan><tspan x="180" dy="19">predeclare evidence rule</tspan><tspan x="180" dy="19">selected sample != prevalence</tspan></text></g><path d="M180 208L180 229" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M90 229L270 229" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M90 229L90 266" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M86.64402122977057,259.85692206676737L90,266L93.35597877022943,259.85692206676737" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M270 229L270 266" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M266.6440212297706,259.85692206676737L270,266L273.3559787702294,259.85692206676737" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="90" y="253" style="font-size:12px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="90" dy="0">EVIDENCE FOUND</tspan></text><text x="270" y="253" style="font-size:12px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="270" dy="0">NOT AVAILABLE</tspan></text><g data-editorial-node><rect x="14" y="270" width="152" height="119" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="90" y="296.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">v1: source repair</tspan><tspan x="90" dy="19">model tuning</tspan><tspan x="90" dy="19">deferred</tspan><tspan x="90" dy="19">owner + review</tspan><tspan x="90" dy="19">date</tspan></text></g><g data-editorial-node><rect x="194" y="270" width="152" height="119" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="296.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">stop this</tspan><tspan x="270" dy="19">proposal</tspan><tspan x="270" dy="19">or change scope</tspan><tspan x="270" dy="19">record missing</tspan><tspan x="270" dy="19">fact</tspan></text></g></g></svg><div class="visual-derivation-caption">The selected complaints do not establish population prevalence. Resolve the named availability question within the agreed timebox. Its receipt changes scope or stops the proposal.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="ambiguous-problem-scope-loop"><strong>Check:</strong> What exact fact can change the scope? Which pre-agreed result would stop a model change?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="ambiguous-problem-scope-loop">The observed complaints concern one slice. A discriminating sample from other slices decides whether the proposed scope should widen.</span></figcaption></figure>

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
