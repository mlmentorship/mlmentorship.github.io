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
<figure class="learning-figure editorial-illustration" data-illustration-print-page aria-labelledby="disagreement-evidence-commit-loop-edition-title"><p class="visual-title" id="disagreement-evidence-commit-loop-edition-title">Compare evidence, respect hard constraints, then commit through the decision owner</p><div class="visual-argument" data-visual-argument="disagreement-evidence-commit-loop" data-argument-kind="bars" data-argument-composition="evidence-owner-commit-reopen"><svg viewBox="0 0 360 2350" role="img" aria-label="Compare evidence, respect hard constraints, then commit through the decision owner. Represent both positions against shared goals. Both options meet the latency limit; cheaper B fails the required region gate. The owner selects a permitted option. Follow it responsibly and observe agreed revisit conditions. New verified approval could reopen B; it is not presumed. Escalate prohibited actions rather than treating commitment as authorization. Neither victory nor a third option is required."><title>Compare evidence, respect hard constraints, then commit through the decision owner</title><desc>Represent both positions against shared goals. Both options meet the latency limit; cheaper B fails the required region gate. The owner selects a permitted option. Follow it responsibly and observe agreed revisit conditions. New verified approval could reopen B; it is not presumed. Escalate prohibited actions rather than treating commitment as authorization. Neither victory nor a third option is required.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Shared goals; credible competing</tspan><tspan x="20" dy="18">positions</tspan></text><text x="20" y="104" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">A favors approved processing. B</tspan><tspan x="20" dy="18">favors speed and lower cost.</tspan></text><text x="20" y="188" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">A: 80 ms; shared limit below 100 ms</tspan></text><rect x="25" y="239" width="248" height="35" data-picture-disagreement-latency="80" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect><path d="M335 231L335 284" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="337" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Cost: $100,000</tspan></text><text x="20" y="407" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Approved processing region</tspan></text><circle cx="308" cy="402" r="24" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="308" y="408" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="308" dy="0">yes</tspan></text><text x="20" y="494" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">B: 60 ms; shared limit below 100 ms</tspan></text><rect x="25" y="545" width="186" height="35" data-picture-disagreement-latency="60" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect><path d="M335 537L335 590" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="643" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Cost: $70,000</tspan></text><text x="20" y="713" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Approved processing region</tspan></text><circle cx="308" cy="708" r="24" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="308" y="714" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="308" dy="0">no</tspan></text><text x="20" y="794" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Cost cannot compensate for a failed</tspan><tspan x="20" dy="18">mandatory gate</tspan></text><circle cx="80" cy="915" r="28" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="80" y="921" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="80" dy="0">A</tspan></text><path d="M104 938L156 1060M149.1835194161061 1054.2090076455415L156 1060L156.5429055332305 1051.0722201202097" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:3;stroke-linejoin:round"></path><circle cx="285" cy="915" r="28" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="285" y="921" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="285" dy="0">B</tspan></text><path d="M285 951L285 1010" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M271 1010L299 1010" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="285" y="1046" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="285" dy="0">blocked</tspan></text><circle cx="180" cy="1093" r="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.75"></circle><text x="180" y="1099" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">Owner</tspan></text><text x="20" y="1173" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Record owner, evidence, decision and</tspan><tspan x="20" dy="18">revisit rule</tspan></text><circle cx="180" cy="1309" r="35" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="180" y="1315" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">Apply</tspan></text><text x="20" y="1298" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Permitted</tspan><tspan x="20" dy="18">choice</tspan></text><path d="M180 1352L180 1458M176 1450L180 1458L184 1450" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-linejoin:round"></path><circle cx="180" cy="1502" r="35" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="180" y="1508" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">Check</tspan></text><text x="20" y="1491" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Observed</tspan><tspan x="20" dy="18">outcome</tspan></text><text x="20" y="1584" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Follow an allowed decision</tspan><tspan x="20" dy="18">responsibly, even when your</tspan><tspan x="20" dy="18">recommendation loses.</tspan></text><text x="20" y="1703" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Conditional change: B gains verified</tspan><tspan x="20" dy="18">regional approval</tspan></text><circle cx="180" cy="1839" r="41" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.75"></circle><text x="180" y="1845" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">Evidence</tspan></text><path d="M180 1890L180 1984M176 1976L180 1984L184 1976" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:3;stroke-linejoin:round"></path><circle cx="180" cy="2029" r="40" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="180" y="2035" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">Reopen</tspan></text><text x="20" y="2125" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">B becomes eligible for</tspan><tspan x="20" dy="18">reconsideration, not automatically</tspan><tspan x="20" dy="18">selected. Approval is not presumed.</tspan></text><text x="20" y="2253" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Escalate prohibited actions. An owner</tspan><tspan x="20" dy="18">does not waive every external duty.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 584" role="img" aria-labelledby="disagreement-evidence-commit-loop-edition-svg-title disagreement-evidence-commit-loop-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="disagreement-evidence-commit-loop-edition-svg-title">Change the decision when evidence changes the shared constraint</title><desc id="disagreement-evidence-commit-loop-edition-svg-desc">Two zero-based bars share the same gigabyte scale. Fictional faster cache A occupies nine units and crosses the eight-gigabyte limit. Smaller cache B occupies six and fits. The decision owner chooses feasible B, retains required reviews, and can reopen A only when evidence shows it fits within the shared limit. The constraint rejects an option rather than its advocate.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">FICTIONAL COMPARISON; ONE SHARED GB SCALE</tspan></text><text x="30" y="68" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="30" dy="0">A: faster</tspan></text><rect data-cache-option="A" data-gigabytes="9" x="30" y="90" width="270" height="35" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="300" y="151" style="font-size:13px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="300" dy="0">9 GB</tspan></text><text x="30" y="173" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="30" dy="0">B: smaller</tspan></text><rect data-cache-option="B" data-gigabytes="6" x="30" y="195" width="180" height="35" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="210" y="256" style="font-size:13px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="210" dy="0">6 GB</tspan></text><path data-cache-limit="8" d="M270 72V264" style="stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:5 4"></path><text x="270" y="293" style="font-size:13px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="270" dy="0">8 GB limit</tspan></text><text x="180" y="340" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">A crosses the limit. B fits the same</tspan><tspan x="180" dy="19">constraint.</tspan></text><g data-editorial-node><rect x="20" y="377" width="320" height="77" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="180" y="411" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Owner selects feasible B.</tspan><tspan x="180" dy="19">Required reviews remain.</tspan></text></g><text x="180" y="493" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Reopen A only when evidence shows it fits</tspan><tspan x="180" dy="19">within 8 GB.</tspan></text><text x="180" y="548" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">The constraint rejects an option, not its</tspan><tspan x="180" dy="19">advocate.</tspan></text></svg><div class="visual-derivation-caption"><strong>Read it this way:</strong> The shared constraint, not seniority, selects the feasible option in this fictional case. Record what evidence would reopen the choice. Required reviews remain.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="disagreement-evidence-commit-loop"><strong>Check:</strong> Which observation would change the preferred option, and who owns the final bounded decision?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="disagreement-evidence-commit-loop">Represent both positions against shared goals. Both options meet the latency limit; cheaper B fails the required region gate. The owner selects a permitted option. Follow it responsibly and observe agreed revisit conditions. New verified approval could reopen B; it is not presumed. Escalate prohibited actions rather than treating commitment as authorization. Neither victory nor a third option is required. <strong>Does cheaper B override the region requirement?</strong></span></figcaption></figure>

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
