---
title: "How do you decide what to work on?"
description: "Compare feasible portfolios against their next-best alternative, protect mandatory obligations and explain a real chosen-versus-declined decision."
date: "2025-10-11"
draft: false
tags: ["questions"]
category: "questions"
---


Choose work against its next-best feasible alternative, not against doing nothing.
Protect obligations before comparing optional projects.

## Problem

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
```mermaid
flowchart TB
	accTitle: A counterfactual process for choosing a portfolio of work
	accDescr: Protect mandatory obligations and deduct their capacity before comparing optional work. In the separate four-week exercise all projects are optional: A costs four weeks for 100 modeled units; B and C cost two weeks each for 60 and 55. Their feasible combined value is 115. Dependencies, ownership, uncertainty and strategy can change the choice. Record selected and declined alternatives, near and long-term capacity and revisit evidence without imposing a fixed allocation fraction.
	A["Candidate projects<br/>outcome, cost, evidence, owner"] --> Obligation{"MANDATORY<br/>obligation?"}
	Obligation -->|"yes"| Protect["Protect required work<br/>agree owner + capacity"]
	Protect --> D
	Obligation -->|"no"| B{"Advances user outcome<br/>and team strategy?"}
	B -->|"no"| C["Decline now<br/>record the reason"]
	B ==>|"yes"| D["Compare remaining capacity<br/>toy: A 100 vs B+C 115"]
	D --> E["Evidence + value<br/>effort + risk"]
	D --> F["Technical leverage<br/>team multiplier"]
	D --> G["Ownership + skills<br/>engagement"]
	E --> H{"Feasible portfolio<br/>under stated assumptions?"}
	F --> H
	G --> H
	H -->|"no"| C
	H ==>|"yes"| I["Select with manager<br/>and affected partners"]
	I --> J["Near + long-term capacity<br/>no fixed fraction"]
	I --> K["Record alternatives<br/>and revisit conditions"]
	C -. "new evidence or strategy" .-> A
	class A,D viz-input
	class Obligation,B,H viz-focus
	class C viz-warning
	class Protect,I,J,K viz-output
	class E,F,G viz-state
	class A viz-wide
```
<p class="diagram-caption"><strong>Read it this way:</strong> protect required work before optional scoring. Compare portfolios using the same remaining capacity, then record what would change the decision.</p>

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
