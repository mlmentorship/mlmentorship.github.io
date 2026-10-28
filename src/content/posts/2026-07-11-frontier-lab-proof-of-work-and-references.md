---
title: "Proof of work: the bounded claim, artifacts and references"
description: "Make a personal contribution inspectable without claiming the team's work, exposing confidential evidence or scripting references."
date: "2026-07-11"
draft: false
tags: ["guides"]
category: "guides"
---

Choose one contribution you can support. Separate your decision from team work, observed results from estimates, and evidence from inference.

## Problem

Audit this fictional draft: "I built the training platform and improved launch quality".
The evidence instead attributes benchmark design and a launch-gate recommendation to the candidate; another team built the platform.
Rewrite the claim without inventing exclusive causal credit.

## Picture: One Claim, Different Checks

<!-- visual:proof-of-work-three-checks -->
<figure class="learning-figure editorial-illustration" aria-labelledby="proof-of-work-three-checks-edition-title"><span class="visual-fragment-anchor" id="proof-of-work-three-checks-title" aria-hidden="true"></span><p class="visual-title" id="proof-of-work-three-checks-edition-title">Evidence can support a bounded contribution without proving a broader outcome</p><svg viewBox="0 0 360 453" role="img" aria-labelledby="proof-of-work-three-checks-edition-svg-title proof-of-work-three-checks-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="proof-of-work-three-checks-edition-svg-title">Evidence can support a bounded contribution without proving a broader outcome</title><desc id="proof-of-work-three-checks-edition-svg-desc">Fictional contributor claims authorship of parser-boundary patch P17, not the team rollout. A shareable diff supports the implementation, a witness observed review but not productivity impact, and no controlled outcome comparison is available. Authorship is supported, the broad effect unknown, and a solo-deployment claim conflicts with team ownership.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional P17; shareable evidence only</tspan></text><text x="90" y="51" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">CLAIM</tspan></text><text x="270" y="51" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">ACTUAL SUPPORT</tspan></text><path d="M15 72L345 72" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="92" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Bounded implementation</tspan></text><g data-editorial-node><rect x="15" y="106" width="150" height="73" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="138" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">I wrote the</tspan><tspan x="90" dy="19">check</tspan></text></g><g data-editorial-node><rect x="195" y="106" width="150" height="73" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="128.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">P17 diff +</tspan><tspan x="270" dy="19">reviewer</tspan><tspan x="270" dy="19">SUPPORTED</tspan></text></g><path d="M15 194L345 194" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="214" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Productivity improvement</tspan></text><g data-editorial-node><rect x="15" y="228" width="150" height="73" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="260" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">the team became</tspan><tspan x="90" dy="19">faster</tspan></text></g><g data-editorial-node><rect x="195" y="228" width="150" height="73" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="270" y="250.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">no causal</tspan><tspan x="270" dy="19">comparison</tspan><tspan x="270" dy="19">UNKNOWN</tspan></text></g><path d="M15 316L345 316" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="336" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Deployment ownership</tspan></text><g data-editorial-node><rect x="15" y="350" width="150" height="73" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="382" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">I shipped it</tspan><tspan x="90" dy="19">alone</tspan></text></g><g data-editorial-node><rect x="195" y="350" width="150" height="73" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="372.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">team owned</tspan><tspan x="270" dy="19">rollout</tspan><tspan x="270" dy="19">CONFLICT</tspan></text></g></svg><figcaption><strong>Read it this way:</strong> The diff and witness support the narrow implementation claim, not every surrounding story. Preserve unknown outcomes, team ownership, and confidentiality. Consistency alone does not establish truth. <span class="visual-reader-check" data-reader-check="proof-of-work-three-checks"><strong>Check:</strong> Which part of the claimed contribution is actually supported? Which remains unknown or team-owned?</span></figcaption></figure>

## Work the Ownership Boundary

This replacement is also fictional, not an achievement to reuse as your own:

> I designed a slice benchmark and recommended a launch gate. Another team owned the training platform. The report records the observed slice recovery. Concurrent data cleanup limits causal attribution.

It has **29 words** under the rule "trim, then split on whitespace".
Use the destination form's actual counting rule and confirmed limit; 100 words is not a universal requirement.

## Build one evidence table

| Claim | Artifact | Your ownership | Team ownership | Direct witness | Caveat |
| --- | --- | --- | --- | --- | --- |
| Slice evaluation and gate | Permitted report or labeled reconstruction | Benchmark design; gate recommendation | Platform and rollout | Direct observer, if available | Concurrent cleanup limits causal credit |

Add **evidence kind**: observed/estimated/unavailable, and **disclosure**: approved/restricted. Record unresolved failures separately.

## Choose one contribution

Look for six properties: technical constraint, personal decision, available evidence, consequence, trade-off or learning, and role relevance.
Unavailable evidence is a limit to disclose, not a reason to invent an artifact, witness, failure or numeric result.

## The 100-word structure

Use five prompts within the confirmed limit: problem/constraint; personal ownership; technical mechanism; evidence; consequence or learning.

## Change 1: The Report Is Confidential

**New task.** Original evidence cannot be shared.

**Picture and change.** Use approved sanitized architecture, a synthetic fixture or a precise verbal reconstruction; label its provenance.

**Check.** An unrelated public repository does not prove this private contribution.

**Use when.** Disclosure is restricted.

## Change 2: The Reference Saw Only the Outcome

**New task.** A senior reference did not observe the decision.

**Picture and change.** Supplement with a direct observer if available; mark each person's knowledge boundary.

**Check.** Title does not establish firsthand knowledge.

**Use when.** Witness coverage differs from the claim.

## Common failure modes

Team work claimed personally, coordinated scripts and leaked evidence undermine verification rather than strengthen it.

## Transfer Check

Must all three sources repeat the same wording?

### Reasoned Answer

No. They should be factually compatible within their knowledge. Independent wording and honest limits are desirable.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Artifact selection

Choose the artifact relevant to the claim:

- Repository/PR: implementation.
- Paper/report/notebook: research.
- Design document: architecture/trade-offs.
- Benchmark/data card: evaluation/data.
- Incident review: debugging/operations.
- Talk/deck: communication.
- Product/demo: user value.

## Prepare the verification path

Ask seven questions: What evidence exists? Observed or estimated? Team ownership? Personal ownership? Direct observer? Confidential details? Unresolved failure?

## References are independent evidence

Choose direct observers and preserve independent judgment, not statistically independent measurements.
Possible perspectives: manager for scope/reliability; peer for technical quality; cross-functional partner for collaboration; mentor for research judgment.

Ask permission before contact. Share accurate role, resume and project context; do not script praise.
Keep names and contact details in private authorized storage, not public preparation artifacts.

## What references may expose

Reconcile factual errors about ownership, metric context, reliability, conflict, causal credit or later growth.
A changed recollection needs honest clarification, not synchronized accounts.

## Preparation checklist

1. Write within the confirmed statement limit.
2. Verify every personal ownership verb.
3. Attach an allowed artifact or labeled reconstruction, if available.
4. Prepare the same decisions for the confirmed presentation duration.
5. Build the claim/evidence table.
6. Ask references; privately confirm contact details.
7. Share which real work may arise.
8. Reconcile factual differences across resume, deck, artifact and reference context.

Absence of a witness or artifact remains an explicit corroboration limit, not proof that the claim is false.

*Related: [present a technical ML project](/questions/present-technical-ml-project/), [your most ambitious project](/questions/most-ambitious-project/), and [technical presentation practice](/prep/presentation/).*

</div>
