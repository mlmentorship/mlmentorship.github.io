---
title: "Applied Scientist vs MLE vs Research Engineer: what these roles actually do"
description: "Companies use the same ML titles for different work. Compare the modeling, research, engineering, and product mix before choosing a target role."
date: "2026-02-28"
draft: false
tags: ["guides"]
category: "guides"
---


Two teams can use the same title for different work.
Compare responsibilities, artifacts, decision rights and success criteria before choosing a role or preparation path.

## Problem

Consider these fictional teams. The candidate prefers systems debugging:

| Team/title | Actual responsibility | Artifact |
| --- | --- | --- |
| A / Research Engineer | Data, experiments and model evaluation | Experiment report |
| B / Applied Scientist | Distributed-training reliability and on-call | Incident diagnosis and reliability changes |

Which team deserves closer investigation? Can the title identify its interview format?

## Picture: Compare Work on Common Axes

The bars are **invented allocations**, not industry averages or measurements of teams A and B.
For this illustration, each activity gets one primary time category; real activities can overlap.

<!-- visual:ml-role-activity-profiles -->
<figure class="learning-figure" aria-labelledby="ml-role-profiles-title">
	<p class="visual-kicker">Fictional activity allocations</p>
	<p class="visual-title" id="ml-role-profiles-title">The same scale exposes different time budgets in these fictional roles</p><div class="visual-argument" data-visual-argument="ml-role-activity-profiles" data-argument-kind="matrix"><p class="visual-argument-premise">Shared activity units expose different fictional role budgets</p><svg viewBox="0 0 360 189" role="img" aria-label="Shared activity units expose different fictional role budgets. Rows are fictional percentages, not hiring benchmarks. RS emphasizes research; MLE emphasizes engineering; the lead example emphasizes coordination."><title>Shared activity units expose different fictional role budgets</title><desc>Rows are fictional percentages, not hiring benchmarks. RS emphasizes research; MLE emphasizes engineering; the lead example emphasizes coordination.</desc><text x="105.13333333333333" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="105.13333333333333" dy="0">research</tspan></text><text x="199.08" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="199.08" dy="0">build</tspan></text><text x="293.02666666666664" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="293.02666666666664" dy="0">coordinate</tspan></text><text x="46.16" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">RS</tspan></text><rect data-proof-cell="0,0" x="58.16" y="36" width="90.94666666666667" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="103.63333333333333" y="63.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="103.63333333333333" dy="0">60</tspan></text><rect data-proof-cell="0,1" x="152.10666666666668" y="36" width="90.94666666666667" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="197.58" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="197.58" dy="0">15</tspan></text><rect data-proof-cell="0,2" x="246.05333333333334" y="36" width="90.94666666666667" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="291.5266666666667" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="291.5266666666667" dy="0">25</tspan></text><text x="46.16" y="101" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">MLE</tspan></text><rect data-proof-cell="1,0" x="58.16" y="83" width="90.94666666666667" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="103.63333333333333" y="110.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="103.63333333333333" dy="0">10</tspan></text><rect data-proof-cell="1,1" x="152.10666666666668" y="83" width="90.94666666666667" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="197.58" y="110.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="197.58" dy="0">65</tspan></text><rect data-proof-cell="1,2" x="246.05333333333334" y="83" width="90.94666666666667" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="291.5266666666667" y="110.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="291.5266666666667" dy="0">25</tspan></text><text x="46.16" y="148" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">lead</tspan></text><rect data-proof-cell="2,0" x="58.16" y="130" width="90.94666666666667" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="103.63333333333333" y="157.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="103.63333333333333" dy="0">20</tspan></text><rect data-proof-cell="2,1" x="152.10666666666668" y="130" width="90.94666666666667" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="197.58" y="157.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="197.58" dy="0">20</tspan></text><rect data-proof-cell="2,2" x="246.05333333333334" y="130" width="90.94666666666667" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="291.5266666666667" y="157.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="291.5266666666667" dy="0">60</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-grid--two" role="group" aria-label="Three hypothetical profiles comparing research, modeling, engineering and product allocations">
		<section class="visual-panel plot-panel">
			<h4>Applied Scientist example</h4>
			<p>This invented allocation gives modeling the largest share.</p>
			<svg viewBox="0 0 300 138" role="img" aria-labelledby="as-profile-title as-profile-desc">
				<title id="as-profile-title">Applied Scientist activity profile</title>
				<desc id="as-profile-desc">An invented allocation, not an industry average. Common zero-to-one-hundred-percent scale: research 20, modeling 40, engineering 20 and product 20 percent.</desc>
				<text class="viz-label" x="0" y="20">Research</text><rect class="viz-plot-bg" x="82" y="9" width="180" height="14" rx="2"></rect><rect x="82" y="9" width="36" height="14" rx="2" style="fill:var(--viz-state-stroke)"></rect><text class="viz-axis-label" x="292" y="20" text-anchor="end">20%</text>
				<text class="viz-label" x="0" y="50">Modeling</text><rect class="viz-plot-bg" x="82" y="39" width="180" height="14" rx="2"></rect><rect x="82" y="39" width="72" height="14" rx="2" style="fill:var(--viz-focus-stroke)"></rect><text class="viz-axis-label" x="292" y="50" text-anchor="end">40%</text>
				<text class="viz-label" x="0" y="80">Engineering</text><rect class="viz-plot-bg" x="82" y="69" width="180" height="14" rx="2"></rect><rect x="82" y="69" width="36" height="14" rx="2" style="fill:var(--viz-input-stroke)"></rect><text class="viz-axis-label" x="292" y="80" text-anchor="end">20%</text>
				<text class="viz-label" x="0" y="110">Product</text><rect class="viz-plot-bg" x="82" y="99" width="180" height="14" rx="2"></rect><rect x="82" y="99" width="36" height="14" rx="2" style="fill:var(--viz-output-stroke)"></rect><text class="viz-axis-label" x="292" y="110" text-anchor="end">20%</text>
				<path class="viz-axis" d="M82 121H262"></path><text class="viz-label" x="82" y="130">0</text><text class="viz-label" x="262" y="130" text-anchor="end">100%</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<h4>Research Engineer example</h4>
			<p>This invented allocation gives engineering the largest share.</p>
			<svg viewBox="0 0 300 138" role="img" aria-labelledby="re-profile-title re-profile-desc">
				<title id="re-profile-title">Research Engineer activity profile</title>
				<desc id="re-profile-desc">An invented allocation, not an industry average. Common zero-to-one-hundred-percent scale: research 25, modeling 30, engineering 40 and product 5 percent.</desc>
				<text class="viz-label" x="0" y="20">Research</text><rect class="viz-plot-bg" x="82" y="9" width="180" height="14" rx="2"></rect><rect x="82" y="9" width="45" height="14" rx="2" style="fill:var(--viz-state-stroke)"></rect><text class="viz-axis-label" x="292" y="20" text-anchor="end">25%</text>
				<text class="viz-label" x="0" y="50">Modeling</text><rect class="viz-plot-bg" x="82" y="39" width="180" height="14" rx="2"></rect><rect x="82" y="39" width="54" height="14" rx="2" style="fill:var(--viz-focus-stroke)"></rect><text class="viz-axis-label" x="292" y="50" text-anchor="end">30%</text>
				<text class="viz-label" x="0" y="80">Engineering</text><rect class="viz-plot-bg" x="82" y="69" width="180" height="14" rx="2"></rect><rect x="82" y="69" width="72" height="14" rx="2" style="fill:var(--viz-input-stroke)"></rect><text class="viz-axis-label" x="292" y="80" text-anchor="end">40%</text>
				<text class="viz-label" x="0" y="110">Product</text><rect class="viz-plot-bg" x="82" y="99" width="180" height="14" rx="2"></rect><rect x="82" y="99" width="9" height="14" rx="2" style="fill:var(--viz-output-stroke)"></rect><text class="viz-axis-label" x="292" y="110" text-anchor="end">5%</text>
				<path class="viz-axis" d="M82 121H262"></path><text class="viz-label" x="82" y="130">0</text><text class="viz-label" x="262" y="130" text-anchor="end">100%</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<h4>ML Engineer example</h4>
			<p>This invented allocation gives engineering half the time.</p>
			<svg viewBox="0 0 300 138" role="img" aria-labelledby="mle-profile-title mle-profile-desc">
				<title id="mle-profile-title">ML Engineer activity profile</title>
				<desc id="mle-profile-desc">An invented allocation, not an industry average. Common zero-to-one-hundred-percent scale: research 5, modeling 30, engineering 50 and product 15 percent.</desc>
				<text class="viz-label" x="0" y="20">Research</text><rect class="viz-plot-bg" x="82" y="9" width="180" height="14" rx="2"></rect><rect x="82" y="9" width="9" height="14" rx="2" style="fill:var(--viz-state-stroke)"></rect><text class="viz-axis-label" x="292" y="20" text-anchor="end">5%</text>
				<text class="viz-label" x="0" y="50">Modeling</text><rect class="viz-plot-bg" x="82" y="39" width="180" height="14" rx="2"></rect><rect x="82" y="39" width="54" height="14" rx="2" style="fill:var(--viz-focus-stroke)"></rect><text class="viz-axis-label" x="292" y="50" text-anchor="end">30%</text>
				<text class="viz-label" x="0" y="80">Engineering</text><rect class="viz-plot-bg" x="82" y="69" width="180" height="14" rx="2"></rect><rect x="82" y="69" width="90" height="14" rx="2" style="fill:var(--viz-input-stroke)"></rect><text class="viz-axis-label" x="292" y="80" text-anchor="end">50%</text>
				<text class="viz-label" x="0" y="110">Product</text><rect class="viz-plot-bg" x="82" y="99" width="180" height="14" rx="2"></rect><rect x="82" y="99" width="27" height="14" rx="2" style="fill:var(--viz-output-stroke)"></rect><text class="viz-axis-label" x="292" y="110" text-anchor="end">15%</text>
				<path class="viz-axis" d="M82 121H262"></path><text class="viz-label" x="82" y="130">0</text><text class="viz-label" x="262" y="130" text-anchor="end">100%</text>
			</svg>
		</section>
	</div>
	<div class="visual-derivation-caption">compare the same activity across panels on the 0–100% scale. A longer bar means more allocated time only in this fictional schedule. Current team evidence, not these invented profiles, determines fit.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="ml-role-activity-profiles"><strong>Check:</strong> How much more engineering time does the fictional ML Engineer profile allocate than the Research Engineer profile?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="ml-role-activity-profiles">Rows are fictional percentages, not hiring benchmarks. RS emphasizes research; MLE emphasizes engineering; the lead example emphasizes coordination.</span></figcaption>
</figure>

## Work the Team Comparison

Team B better matches the stated systems interest **for further investigation**.
Ask about reliability ownership, on-call load, research expectations and learning support before deciding.
If B's current work instead becomes model evaluation, update the fit assessment.
Neither title establishes salary, degree requirements or interview format. Confirm the round constraints separately.

## The basic decomposition

- **Research:** hypotheses, experiments and scientific reports; success depends on supported findings.
- **Modeling:** data, training and evaluation; inspect quality against the task's requirements.
- **Engineering:** training/serving systems and operations; inspect reliability and resource constraints.
- **Product:** user needs and outcomes; inspect measurement, guardrails and adoption.

These activities overlap. Ask how the team assigns ownership and measures success, not just time percentages.

## The decision tree

Build an editable team profile:

| Field | Record |
| --- | --- |
| Task | Current recurring work |
| Artifact | Expected deliverable |
| Success criterion | Observable standard |
| Decision rights | What you own or must escalate |
| Collaborators | Who shares the work |
| Operating duties | Deployment, on-call or research obligations |
| Learning needs | Current gap and available support |
| Evidence | Source, scope, date and unknowns |

## Change 1: Responsibilities Contradict the Archetype

**New task.** Current manager/team evidence conflicts with the title's implied mix.

**Picture and change.** Replace the profile with current tasks, artifacts and operating duties.

**Check.** The title is not ground truth.

**Use when.** Role labels are ambiguous.

## Change 2: A Private-Lab Offer Looks Higher

**New task.** Compensation affects the choice.

**Picture and change.** Compare dated base, bonus, equity valuation/liquidity, location, level, risk and work fit; state unknowns.

**Check.** A posted base range does not establish a total-compensation multiple.

**Use when.** Offers differ in pay structure.

## Failure: A Prestige Ladder Instead of Team Evidence

Degree labels, invented averages and title equivalences do not establish the actual job.

## Transfer Check

What should overturn a title-based preparation plan?

### Reasoned Answer

Current tasks, artifacts, success criteria and ownership, confirmed with relevant people and dated sources. Then verify assessment constraints.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Role-by-role, in plain language

These are possible technical cores, not universal job descriptions or a prestige ranking.
Researchers can ship products; engineers can lead scientific experiments.

### Research Scientist

Possible core: question selection, research methods and supported findings. Ask about publication, deployment and collaboration obligations.

### Applied Scientist

Possible core: product-to-ML framing, modeling and evaluation. Ask who owns serving, launch and product outcomes.

### Research Engineer

Possible core: research implementations, experiments and training systems. Ask about scientific decisions versus infrastructure ownership.

### ML Engineer

Possible core: data/training pipelines, serving, testing and monitoring. Ask about modeling depth and operating duties.

### Software Engineer (ML adjacent)

Possible core: APIs, applications, integration and data systems. Ask which ML-specific behavior and failure modes require ownership.

## Fictional Allocation Reference

These invented schedules are not measured industry averages. Each row assigns 100% to disjoint primary activity bins.

| Example label | Research | Modeling | Engineering | Product |
| --- | ---: | ---: | ---: | ---: |
| RS | 60% | 30% | 5% | 5% |
| AS | 20% | 40% | 20% | 20% |
| RE | 25% | 30% | 40% | 5% |
| MLE | 5% | 30% | 50% | 15% |
| ML-adjacent SWE | 0% | 10% | 80% | 10% |

## Compensation patterns (US, mid-2026)

No verified pay comparison is established here. Use sourced offer terms; do not infer liquid equity value from a headline valuation.

## What changes between levels

Compare scope, uncertainty, authority, influence and time horizon using that employer's definitions. Level names are not fixed cross-company equivalents.

## How to interview for each

Use [confirmed work samples](/guides/frontier-lab-interview-processes-2026/) and the [role paths](/prep/role-paths/), not title-based round assumptions.

## A note on title inflation

Ask about shipped users, team track record, day-to-day mix and the work you would actually own.
Qualifications vary; verify the real requirements rather than impose a universal degree-to-role route.

---

*Related: [The 5 things every applied scientist interview is testing for](/guides/five-things-as-interview-tests/), [senior through senior-principal ML scope](/guides/l5-vs-l6-faang-ml/).*

</div>
