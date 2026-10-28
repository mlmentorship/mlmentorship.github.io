---
title: "Tell me about an ML project you killed or redirected"
description: "A staff-level behavioral and project question about sunk costs, evidence thresholds, influence, and responsible stopping."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> *Asked in: staff-level behavioral and project rounds.*

Describe a project you helped stop or materially redirect, including the evidence, alternatives, and consequences of that decision.

## Problem

**Context.** An interviewer asks about ending an ML effort despite prior investment or personal attachment.
The decision may involve technical feasibility, user value, operating cost, or a changed requirement.

**Task.** Explain the original goal, new evidence, options considered, your role, decision authority, and what happened to people or reusable work.
Describe what would have justified continuing instead.

**Constraints.** Use an actual experience and distinguish your recommendation from the final team decision.
Separate sunk costs from future alternatives, protect confidential information, and avoid claiming unmeasured counterfactual savings as established impact.

## What a strong answer covers

Explain the decision, not only the outcome. Cover:

1. Why the project was rational at the start.
2. Which evidence or failed assumption changed the case.
3. Why another iteration was a poor use of resources.
4. What experiment, proposal, and transition work you owned.
5. The cost avoided, effort redirected, users protected, or capability learned.
6. The earlier signal you would instrument next time.

<!-- visual:project-stop-forward-evidence-ledger -->
<figure class="learning-figure editorial-illustration" aria-labelledby="project-stop-forward-evidence-ledger-edition-title"><span class="visual-fragment-anchor" id="project-stop-ledger-title" aria-hidden="true"></span><p class="visual-title" id="project-stop-forward-evidence-ledger-edition-title">Past spend cannot fill a future-value column</p><svg viewBox="0 0 360 626" role="img" aria-labelledby="project-stop-forward-evidence-ledger-edition-svg-title project-stop-forward-evidence-ledger-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="project-stop-forward-evidence-ledger-edition-svg-title">Past spend cannot fill a future-value column</title><desc id="project-stop-forward-evidence-ledger-edition-svg-desc">Three fictional mutually exclusive options share a hundred future resource units. A bounded test spends ten, continued investment one hundred, and redirect one hundred. Past spend is excluded. The test stops on day seven and requires at least ninety-five of a hundred contract passes and zero leaks; passing triggers a new decision, not automatic renewal.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional options from today; not three</tspan><tspan x="180" dy="19">simultaneous spends</tspan></text><text x="180" y="72" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Past spend: recorded, excluded from this</tspan><tspan x="180" dy="19">choice</tspan></text><text x="18" y="126" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Bounded test</tspan></text><rect x="25" y="146" width="300" height="24" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3;stroke-dasharray:5 3"></rect><rect x="25" y="146" width="30" height="24" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="330" y="192" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:end"><tspan x="330" dy="0">10 / 100 future resource units</tspan></text><text x="18" y="217" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Continue</tspan></text><rect x="25" y="237" width="300" height="24" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3;stroke-dasharray:5 3"></rect><rect x="25" y="237" width="300" height="24" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="330" y="283" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:end"><tspan x="330" dy="0">100 / 100 future resource units</tspan></text><text x="18" y="308" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Redirect</tspan></text><rect x="25" y="328" width="300" height="24" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3;stroke-dasharray:5 3"></rect><rect x="25" y="328" width="300" height="24" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="330" y="374" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:end"><tspan x="330" dy="0">100 / 100 future resource units</tspan></text><g data-editorial-node><rect x="25" y="428" width="310" height="106" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="180" y="448" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Test owner: project lead; stop: day</tspan><tspan x="180" dy="19">7</tspan><tspan x="180" dy="19">need &gt;=95/100 contract passes AND</tspan><tspan x="180" dy="19">zero leaks</tspan><tspan x="180" dy="19">otherwise stop; no automatic renewal</tspan></text></g><text x="180" y="552" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Returns remain uncertain for all three</tspan><tspan x="180" dy="19">options.</tspan></text><text x="180" y="590" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">The test buys decision-changing evidence, not a</tspan><tspan x="180" dy="19">new quarter.</tspan></text></svg><figcaption><strong>Read it this way:</strong> Compare equal future budgets, not historical investment. The ten-unit test must change a decision by day seven. All returns remain uncertain; continuation displaces the redirect. <span class="visual-reader-check" data-reader-check="project-stop-forward-evidence-ledger"><strong>Check:</strong> What specific observation would justify another investment? Which better use of the same resources is displaced?</span></figcaption></figure>

## What an L4 answer sounds like

> "The model did not reach accuracy, so the team cancelled the project."

An outcome with no judgment or ownership in it.

## What an L5 answer adds

An L5 answer owns the evidence and recommendation inside a project area. It defines minimum viable quality before more spend, runs an experiment that separates a broken thesis from fixable execution, identifies reusable work, and transitions users, data, and infrastructure responsibly.

## What an L6 answer adds

An L6 answer manages the portfolio and organizational consequences: pushing past sunk-cost and prestige incentives, aligning several teams on a new direction, keeping the trust of sponsors who backed the original bet, and changing review criteria so similar weak bets fail faster. Keep the technical detail. "I reallocated the roadmap" with no evidence behind it sounds managerial and vague.

## Tells that get you a strong-hire vote

- The original decision was reasonable, not an obvious straw man.
- You name the assumption that failed.
- The stopping criteria were evidence-based.
- You handled the people and transition costs with care.
- You account for both avoided waste and lost opportunity.

## Tells that get you down-leveled

- The project died because another team cancelled it, not because of you.
- You take credit for hindsight.
- No one who disagreed is represented fairly.
- "We pivoted" hides the specific decision.
- Nothing you learned changed how you invest next time.

## Common follow-ups

- Why not iterate for another quarter?
- Who disagreed, and what evidence did they have?
- How much had already been invested?
- What parts of the work were reusable?
- Tell me about a time you stopped too early, or held on too long.

*Related: [decide what to work on](/questions/decide-what-to-work-on/), [scope an ambiguous problem](/questions/scope-ambiguous-problem/), and [senior through senior-principal ML scope](/guides/l5-vs-l6-faang-ml/).*
