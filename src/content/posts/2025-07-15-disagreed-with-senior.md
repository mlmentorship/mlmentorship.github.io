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
<figure class="learning-figure editorial-illustration" data-illustration-print-page aria-labelledby="disagreement-evidence-commit-loop-edition-title"><p class="visual-title" id="disagreement-evidence-commit-loop-edition-title">The shared memory limit rejects the faster but oversized cache</p><div class="visual-argument" data-visual-argument="disagreement-evidence-commit-loop" data-argument-kind="bars" data-argument-composition="endpoint"><svg viewBox="0 0 360 327" role="img" aria-label="The memory limit rejects an option, not its advocate. On one shared scale, faster cache A breaches 8 GB. Cache B fits; reopening A needs new evidence of feasibility."><title>The memory limit rejects an option, not its advocate</title><desc>On one shared scale, faster cache A breaches 8 GB. Cache B fits; reopening A needs new evidence of feasibility.</desc><text x="18" y="23" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">A: faster cache</tspan></text><path d="M28 64L332 64" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="9" data-proof-min="0" data-proof-max="9" data-proof-zero="28" x="28" y="46" width="304" height="35" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></rect><circle cx="332" cy="63.5" r="4" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><path data-proof-overflow="1" d="M298.22222222222223 54H332M298.22222222222223 63.5H332M298.22222222222223 73H332" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2"></path><path d="M298.22222222222223 40L298.22222222222223 89" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M332 69L332 86" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="297" y="106" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="297" dy="0">9 GB</tspan></text><text x="28" y="131" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="28" dy="0">limit 8 GB</tspan></text><text x="18" y="165" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">B: smaller cache</tspan></text><path d="M28 206L332 206" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="6" data-proof-min="0" data-proof-max="9" data-proof-zero="28" x="28" y="188" width="202.66666666666666" height="35" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><circle cx="230.66666666666666" cy="205.5" r="4" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><path d="M298.22222222222223 182L298.22222222222223 231" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M230.66666666666666 211L230.66666666666666 228" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><text x="230.66666666666666" y="248" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="230.66666666666666" dy="0">6 GB</tspan></text><text x="28" y="273" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="28" dy="0">limit 8 GB</tspan></text><text x="28" y="307" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="28" dy="0">0</tspan></text><text x="332" y="307" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="332" dy="0">9</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 584" role="img" aria-labelledby="disagreement-evidence-commit-loop-edition-svg-title disagreement-evidence-commit-loop-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="disagreement-evidence-commit-loop-edition-svg-title">Change the decision when evidence changes the shared constraint</title><desc id="disagreement-evidence-commit-loop-edition-svg-desc">Two zero-based bars share the same gigabyte scale. Fictional faster cache A occupies nine units and crosses the eight-gigabyte limit. Smaller cache B occupies six and fits. The decision owner chooses feasible B, retains required reviews, and can reopen A only when evidence shows it fits within the shared limit. The constraint rejects an option rather than its advocate.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">FICTIONAL COMPARISON; ONE SHARED GB SCALE</tspan></text><text x="30" y="68" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="30" dy="0">A: faster</tspan></text><rect data-cache-option="A" data-gigabytes="9" x="30" y="90" width="270" height="35" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="300" y="151" style="font-size:13px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="300" dy="0">9 GB</tspan></text><text x="30" y="173" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="30" dy="0">B: smaller</tspan></text><rect data-cache-option="B" data-gigabytes="6" x="30" y="195" width="180" height="35" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="210" y="256" style="font-size:13px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="210" dy="0">6 GB</tspan></text><path data-cache-limit="8" d="M270 72V264" style="stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:5 4"></path><text x="270" y="293" style="font-size:13px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="270" dy="0">8 GB limit</tspan></text><text x="180" y="340" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">A crosses the limit. B fits the same</tspan><tspan x="180" dy="19">constraint.</tspan></text><g data-editorial-node><rect x="20" y="377" width="320" height="77" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="180" y="411" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Owner selects feasible B.</tspan><tspan x="180" dy="19">Required reviews remain.</tspan></text></g><text x="180" y="493" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Reopen A only when evidence shows it fits</tspan><tspan x="180" dy="19">within 8 GB.</tspan></text><text x="180" y="548" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">The constraint rejects an option, not its</tspan><tspan x="180" dy="19">advocate.</tspan></text></svg><div class="visual-derivation-caption"><strong>Read it this way:</strong> The shared constraint, not seniority, selects the feasible option in this fictional case. Record what evidence would reopen the choice. Required reviews remain.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="disagreement-evidence-commit-loop"><strong>Check:</strong> Which observation would change the preferred option, and who owns the final bounded decision?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="disagreement-evidence-commit-loop">On one shared scale, faster cache A breaches 8 GB. Cache B fits; reopening A needs new evidence of feasibility.</span></figcaption></figure>

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
