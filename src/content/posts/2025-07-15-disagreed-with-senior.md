---
title: "Tell me about a time you disagreed with someone senior"
description: "The standard behavioral question. The interviewer is checking whether you can hold technical positions, push back productively, and update on new information."
date: "2025-07-15"
draft: false
tags: ["questions"]
category: "questions"
---


Explain a technical disagreement with a senior colleague through both positions, the deciding evidence, and your role in the resolution.

## Problem

**Context.** An interviewer asks about a real disagreement; the fictional choice below provides a rehearsal of the decision mechanics.

**Task.** Explain the competing goals, gather deciding evidence, identify the decision owner, and describe the outcome and follow-through.
For the rehearsal, decide which option currently satisfies the requirements and what new evidence could reopen the choice.

**Constraints.** Mandatory processing-region rules cannot be traded away for speed or cost without authorized requirement changes.
Use your own experience in an interview, distinguish facts from assumptions, and represent the other position fairly.

Use this fictional rehearsal, not a personal project claim:

| Option | Latency | Cost | Approved processing region? |
| --- | ---: | ---: | --- |
| A | 80 ms | $100,000 | Yes |
| B | 60 ms | $70,000 | No |

The requirements are latency below 100 ms and an approved processing location.
Someone favors B's speed and cost; you favor A's compliance with the current boundary.
What evidence resolves the choice? What could reopen it?

## Picture: Evidence Before Commitment

<!-- visual:disagreement-evidence-commit-loop -->
<figure class="learning-figure editorial-illustration" aria-labelledby="disagreement-evidence-commit-loop-edition-title"><p class="visual-title" id="disagreement-evidence-commit-loop-edition-title">Change the decision when evidence changes the shared constraint</p><svg viewBox="0 0 360 409" role="img" aria-labelledby="disagreement-evidence-commit-loop-edition-svg-title disagreement-evidence-commit-loop-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="disagreement-evidence-commit-loop-edition-svg-title">Change the decision when evidence changes the shared constraint</title><desc id="disagreement-evidence-commit-loop-edition-svg-desc">A fictional technical disagreement compares a faster high-memory cache with a smaller cache under a shared eight-gigabyte limit. A nine-gigabyte measured toy footprint rules out the first alternative under that constraint. The decision owner chooses the feasible option, records a memory-based revisit trigger, and retains required safety review.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional cache discussion; shared limit 8</tspan><tspan x="180" dy="19">GB</tspan></text><g transform="translate(0 19)"><g data-editorial-node><rect x="30" y="43" width="300" height="58" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="67.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">A: faster cache, 9 GB</tspan><tspan x="180" dy="19">B: smaller cache, 6 GB</tspan></text></g><path d="M180 101L180 128" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,121.8569220667674L180,128L183.35597877022943,121.8569220667674" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="30" y="132" width="300" height="76" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="180" y="165.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Does the candidate meet the shared</tspan><tspan x="180" dy="19">memory constraint?</tspan></text></g><path d="M180 208L180 229" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M90 229L270 229" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M90 229L90 266" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M86.64402122977057,259.85692206676737L90,266L93.35597877022943,259.85692206676737" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M270 229L270 266" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M266.6440212297706,259.85692206676737L270,266L273.3559787702294,259.85692206676737" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="90" y="253" style="font-size:12px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="90" dy="0">B: 6 &lt;= 8</tspan></text><text x="270" y="253" style="font-size:12px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="270" dy="0">A: 9 &gt; 8</tspan></text><g data-editorial-node><rect x="14" y="270" width="152" height="100" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="90" y="296.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">owner selects B</tspan><tspan x="90" dy="19">revisit if A fits</tspan><tspan x="90" dy="19">required reviews</tspan><tspan x="90" dy="19">stay</tspan></text></g><g data-editorial-node><rect x="194" y="270" width="152" height="100" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="306" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">reject A for now</tspan><tspan x="270" dy="19">not a judgment</tspan><tspan x="270" dy="19">of its advocate</tspan></text></g></g></svg><figcaption><strong>Read it this way:</strong> The shared constraint, not seniority, selects the feasible option in this fictional case. Record what evidence would reopen the choice. Required reviews remain. <span class="visual-reader-check" data-reader-check="disagreement-evidence-commit-loop"><strong>Check:</strong> Which observation would change the preferred option, and who owns the final bounded decision?</span></figcaption></figure>

## Apply the Constraints Independently

Both 80 and 60 ms are below 100 ms.
B's lower cost cannot compensate for its failed region requirement.
The owner can select A or investigate an approved B configuration.
Record regional approval as a possible reopening condition; do not assume it will occur.

## Change 1: The Other Approach Is Selected

**New task.** The owner chooses another defensible, permitted approach.

**Picture and change.** Execute it responsibly; observe the distinguishing metrics and agreed revisit condition.

**Check.** Commitment does not require pretending your earlier uncertainty vanished or waiting to say you were right.

**Use when.** Your recommendation is not selected.

## Change 2: The Decision Crosses a Hard Boundary

**New task.** The chosen action violates a non-negotiable safety, privacy or integrity rule.

**Picture and change.** Use required review/escalation and keep appropriate records; do not treat ordinary commitment as authorization.

**Check.** An owner cannot waive every external duty.

**Use when.** The issue is a prohibited action, not an ordinary preference.

## Failure: A Scripted Victory or Conversion

A story fails to explain the decision when it omits credible opposition or invents a change of mind.

## Transfer Check

Must a disagreement end with a third option C?

### Reasoned Answer

No. Evidence can support retaining A or B. Synthesis is a possible result, not a required narrative.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## What the question actually tests

Describe technical conviction, productive challenge and willingness to update when warranted.
Do not fabricate disagreement or opposing competence to fit a story.

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-sounds-like"></span>
<span id="what-an-l6-answer-adds"></span>
## Evidence Worksheet

Use STAR(L) as optional prompts, not memorized filler:

| Prompt | Record |
| --- | --- |
| Situation/task | Shared goal, constraints, responsibility and decision rights |
| Positions | Strongest actual argument for each option |
| Action | Your discussion, document, test, collaboration or escalation |
| Evidence | Measured facts, assumptions and remaining uncertainty |
| Decision/result | Owner, selected option, what shipped and observed outcome |
| Learning | What changed or remained; revisit condition |

Separate personal/team contributions and observed/counterfactual impact.
When a decisive test is unavailable, state criteria and risks instead of inventing certainty.
Technical versus strategic topic labels do not determine scope or company level.

<span id="tells-that-get-you-a-strong-hire-vote"></span>
## Check the Account

Can another person probe the technical decision and reconstruct your actual contribution?
Neither changing your view nor creating C is mandatory.

<span id="tells-that-get-you-down-leveled"></span>
## Missing Evidence

Replace generic advocacy with the missing rationale, action, outcome limit or reopening rule.

## A trap to avoid

Respectful representation does not require pretending an unsafe proposal is acceptable.
Avoid caricaturing the other person; explain the disputed reasoning and constraint.

## Common follow-up

What if your recommendation lost? Explain allowed execution, distinguishing observations and agreed review, not a grudge.

## Sources

- [Amazon leadership principles](https://www.amazon.jobs/content/en/our-workplace/leadership-principles).
- [Psychological safety and team learning](https://doi.org/10.2307/2666999).
- [NASA decision analysis](https://ntrs.nasa.gov/citations/20170001761).

---

*Related: [The 5 things every applied scientist interview is testing for](/guides/five-things-as-interview-tests/), [senior through senior-principal ML scope](/guides/l5-vs-l6-faang-ml/).*

</div>
