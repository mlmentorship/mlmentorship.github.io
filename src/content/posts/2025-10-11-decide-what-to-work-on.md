---
title: "How do you decide what to work on?"
description: "Compare feasible portfolios against their next-best alternative, protect mandatory obligations and explain a real chosen-versus-declined decision."
date: "2025-10-11"
draft: false
tags: ["questions"]
category: "questions"
---


Choose a feasible work portfolio under a fixed capacity limit and explain what assumptions could change the priority.

## Problem

**Context.** Several optional projects compete for the same four engineer-weeks in the invented exercise below.

**Task.** Compare feasible portfolios, select one, and explain opportunity cost and evidence that could reverse the choice.
Connect the reasoning to how you prioritize real work without claiming invented values as measured outcomes.

**Constraints.** Projects are indivisible and independent, and their modeled benefits add on one common scale.
Respect capacity and mandatory obligations before optimizing optional value; dependencies or overlapping benefits change the calculation.

In this hypothetical exercise, four engineer-weeks are available:

| Project | Engineer-weeks | Modeled value units |
| --- | ---: | ---: |
| A | 4 | 100 |
| B | 2 | 60 |
| C | 2 | 55 |

All projects are optional, independently staffed and indivisible; benefits add without overlap on this invented common scale.
Which portfolio fits? What could reverse the choice?

## Picture: Choosing One Uses Capacity Elsewhere

<!-- visual:work-prioritization-counterfactual-portfolio -->
<figure class="learning-figure editorial-illustration" aria-labelledby="work-prioritization-counterfactual-portfolio-edition-title"><p class="visual-title" id="work-prioritization-counterfactual-portfolio-edition-title">Choosing A gives up the higher assumed value of B plus C</p><div class="visual-argument" data-visual-argument="work-prioritization-counterfactual-portfolio" data-argument-kind="bars"><p class="visual-argument-premise">The alternative portfolio is the opportunity cost</p><svg viewBox="0 0 360 277" role="img" aria-label="The alternative portfolio is the opportunity cost. In this illustrative fixed-capacity choice, A yields 8 while compatible B and C yield 11. Past effort on A does not change the forward comparison."><title>The alternative portfolio is the opportunity cost</title><desc>In this illustrative fixed-capacity choice, A yields 8 while compatible B and C yield 11. Past effort on A does not change the forward comparison.</desc><text x="18" y="23" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Choose A alone</tspan></text><path d="M28 64L332 64" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="8" data-proof-min="0" data-proof-max="11" data-proof-zero="28" x="28" y="46" width="221.0909090909091" height="35" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></rect><text x="28" y="106" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="28" dy="0">8 assumed value units</tspan></text><text x="18" y="140" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Choose B plus C</tspan></text><path d="M28 181L332 181" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="11" data-proof-min="0" data-proof-max="11" data-proof-zero="28" x="28" y="163" width="304" height="35" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><text x="28" y="223" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="28" dy="0">11 assumed value units</tspan></text><text x="28" y="257" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="28" dy="0">0</tspan></text><text x="332" y="257" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="332" dy="0">11</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 456" role="img" aria-labelledby="work-prioritization-counterfactual-portfolio-edition-svg-title work-prioritization-counterfactual-portfolio-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="work-prioritization-counterfactual-portfolio-edition-svg-title">Every attractive project spends the same remaining capacity</title><desc id="work-prioritization-counterfactual-portfolio-edition-svg-desc">A toy 100-unit capacity ledger reserves 20 mandatory units, leaving 80. Project A costs all 80 and has assumed value 100. Projects B and C cost 45 and 35 and have assumed combined value 115. The two portfolios occupy equal total capacity. Choosing B plus C displaces A, not mandatory work.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Toy planning ledger; values are assumptions</tspan></text><text x="20" y="61" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">100 capacity units</tspan></text><rect x="25" y="82" width="62" height="28" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><rect x="87" y="82" width="248" height="28" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="56" y="103" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="56" dy="0">20</tspan></text><text x="211" y="103" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="211" dy="0">80 available</tspan></text><text x="25" y="137" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="25" dy="0">Mandatory work is reserved first.</tspan></text><text x="20" y="184" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">A: capacity 80 / value 100</tspan></text><rect x="25" y="204" width="310" height="28" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="224" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">80</tspan></text><text x="20" y="284" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">B + C: capacity 80 / value 115</tspan></text><rect x="25" y="304" width="174.375" height="28" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="112.1875" y="324" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="112.1875" dy="0">45</tspan></text><rect x="199.375" y="304" width="135.625" height="28" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="267.1875" y="324" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="267.1875" dy="0">35</tspan></text><text x="180" y="378" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Choosing B + C displaces A, not the mandatory</tspan><tspan x="180" dy="19">work.</tspan></text><text x="180" y="420" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Revisit when value, risk, or capacity evidence</tspan><tspan x="180" dy="19">changes.</tspan></text></svg><div class="visual-derivation-caption">Reserve obligations before comparing portfolios. Under these toy assumptions, B + C beats A at equal capacity. Revisit when evidence changes the assumptions.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="work-prioritization-counterfactual-portfolio"><strong>Check:</strong> What other feasible work is forgone when project A is chosen?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="work-prioritization-counterfactual-portfolio">In this illustrative fixed-capacity choice, A yields 8 while compatible B and C yield 11. Past effort on A does not change the forward comparison.</span></figcaption></figure>

## Compare the Feasible Portfolios

A uses four weeks for 100 units.
B+C uses $2+2=4$ weeks for $60+55=115$ units: 15 more under the stated assumptions.
This is modeled value, not measured impact.

Do not add incompatible real-world metrics into a precise-looking score or double-count shared benefits.
Use ranges and explicit outcome/guardrail trade-offs instead.

## Change 1: B Needs an Unfunded Platform

**New task.** B requires two additional engineer-weeks before delivering value.

**Picture and change.** Include prerequisite cost, timing and shared benefits. B+C now needs six weeks and is infeasible.

**Check.** A's 100 beats feasible C's 55 here; B alone offers 60 within four weeks.

**Use when.** Projects share dependencies or scarce skills.

## Change 2: A Is Assigned Required Work

**New task.** A satisfies a mandatory obligation.

**Picture and change.** Represent its rationale fairly; agree authority, scope and capacity before proposing alternatives.

**Check.** Optional value cannot cancel an obligation. Saying no is not itself impact.

**Use when.** Priorities conflict with another owner.

## Failure: Criteria Without a Decision

Interests and strategy slogans do not identify a forgone alternative, evidence or actual outcome.

## Transfer Check

Why might a lower estimated-value project still be selected?

### Reasoned Answer

A duty, dependency, risk, skill constraint or uncertain forecast can change feasibility or utility.
Name the reason rather than hiding it inside a score.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-sounds-like"></span>
<span id="what-an-l6-answer-adds"></span>
## Evidence Worksheet

Use a real recent choice, not the fictional value exercise as a personal achievement.

| Field | Record |
| --- | --- |
| Candidate/outcome | User population, outcome and guardrails |
| Evidence | Known facts, uncertainty and modeled versus observed impact |
| Future cost | People, time and next-best alternative; do not justify from sunk effort |
| Leverage | Technical prerequisites and team multiplier without double counting |
| Fit | Skills, owner, delegation and sustained engagement |
| Decision | What you chose/declined, with whose authority |
| Revisit | What shipped, measured outcomes and reopening evidence |

Balance delivery and longer-term investment against strategy and obligations.
Two-thirds/one-third is an illustrative allocation, not a rule.
Engagement informs ownership feasibility; it does not override user duties.

<span id="tells-that-get-you-a-strong-hire-vote"></span>
## Check the Account

Can another person reconstruct the alternative, rationale, personal contribution and observed result?
Decision rights vary by team; independence is not a universal level test.

<span id="tells-that-get-you-down-leveled"></span>
## Missing Evidence

Add missing user value, strategic rationale, ownership or a concrete trade-off. Do not invent an example or outcome.

## Common follow-up

Describe a time you proposed not doing assigned work.
Explain the original rationale, alternative, assigning party's response, final decision, actual delivery and what you would change.

## Sources

- [NASA decision analysis](https://ntrs.nasa.gov/citations/20170001761).
- [OpenStax opportunity cost](https://openstax.org/books/principles-economics-3e/pages/2-1-how-individuals-make-choices-based-on-their-budget-constraint).
- [Project portfolio research](https://doi.org/10.1111/1467-9310.00225).

---

*Related: [The 5 things every applied scientist interview is testing for](/guides/five-things-as-interview-tests/), [senior through senior-principal ML scope](/guides/l5-vs-l6-faang-ml/).*

</div>
