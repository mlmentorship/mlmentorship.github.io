---
title: "Five observable skills to practice across ML interview prompts"
description: "Use attempts and feedback to practice framing, relevant evidence, technical mechanisms, critique and collaboration without assuming a universal hiring rubric."
date: "2026-03-20"
draft: false
tags: ["guides"]
category: "guides"
---


One prompt can test framing, mechanisms, delivery and collaboration together.
Make an attempt, observe the gap, then select the next drill. These five lenses are a preparation heuristic, not every employer's rubric.

## Problem

A fictional learner says squared loss always gives small gradients, but cannot specify the model's output parameterization.
For a binary sigmoid model, let $p=\sigma(z)=0.01$ and target $y=1$.
Compare cross-entropy with $L_{\mathrm{sq}}=\tfrac12(p-1)^2$ under the **same logit** $z$.
Which practice gap does this expose?

## Picture: Observe Skills Across Prompt Types

<!-- visual:as-five-cross-cutting-signals -->
<figure class="learning-figure editorial-illustration" aria-labelledby="as-five-cross-cutting-signals-edition-title"><span class="visual-fragment-anchor" id="as-signals-title" aria-hidden="true"></span><p class="visual-title" id="as-five-cross-cutting-signals-edition-title">One follow-up can reveal several skills while leaving others unknown</p><div class="visual-argument" data-visual-argument="as-five-cross-cutting-signals" data-argument-kind="matrix"><p class="visual-argument-premise">One follow-up can reveal several skills and leave others unknown</p><svg viewBox="0 0 360 95" role="img" aria-label="One follow-up can reveal several skills and leave others unknown. A specific follow-up yields evidence about problem framing and experimentation here. Unknown signals remain unknown rather than being inferred from confidence."><title>One follow-up can reveal several skills and leave others unknown</title><desc>A specific follow-up yields evidence about problem framing and experimentation here. Unknown signals remain unknown rather than being inferred from confidence.</desc><text x="94.02999999999999" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="94.02999999999999" dy="0">frame</tspan></text><text x="148.69" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="148.69" dy="0">test</tspan></text><text x="203.35" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="203.35" dy="0">scale</tspan></text><text x="258.01" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="258.01" dy="0">lead</tspan></text><text x="312.67" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="312.67" dy="0">impact</tspan></text><text x="54.69999999999999" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="54.69999999999999" dy="0">probe</tspan></text><rect data-proof-cell="0,0" x="66.69999999999999" y="36" width="51.660000000000004" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="92.52999999999999" y="63.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="92.52999999999999" dy="0">seen</tspan></text><rect data-proof-cell="0,1" x="121.35999999999999" y="36" width="51.660000000000004" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="147.19" y="63.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="147.19" dy="0">seen</tspan></text><rect data-proof-cell="0,2" x="176.01999999999998" y="36" width="51.660000000000004" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="201.85" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="201.85" dy="0">?</tspan></text><path d="M180.01999999999998 76L223.67999999999998 40" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><rect data-proof-cell="0,3" x="230.68" y="36" width="51.660000000000004" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="256.51" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="256.51" dy="0">?</tspan></text><path d="M234.68 76L278.34000000000003 40" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><rect data-proof-cell="0,4" x="285.34000000000003" y="36" width="51.660000000000004" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="311.17" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="311.17" dy="0">?</tspan></text><path d="M289.34000000000003 76L333.00000000000006 40" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 595" role="img" aria-labelledby="as-five-cross-cutting-signals-edition-svg-title as-five-cross-cutting-signals-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="as-five-cross-cutting-signals-edition-svg-title">One exchange can expose several skills and leave others unknown</title><desc id="as-five-cross-cutting-signals-edition-svg-desc">A fictional mock asks whether an overall completion gain justifies shipping. The candidate asks about harmed users and stop criteria, then updates after a high-risk slice failure. Scoping, responsiveness, and explicit decision-making are observed; delivery ownership and scientific depth remain untested. The next drill targets uncertainty in the slice test.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional mock; observable evidence, not</tspan><tspan x="180" dy="19">personality scores</tspan></text><g data-editorial-node><rect x="25" y="62" width="310" height="62" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="79" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Interviewer: overall completion is</tspan><tspan x="180" dy="19">up.</tspan><tspan x="180" dy="19">Would you ship?</tspan></text></g><g data-editorial-node><rect x="25" y="151" width="310" height="62" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="180" y="177.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Candidate: which users lose?</tspan><tspan x="180" dy="19">What is the stop threshold?</tspan></text></g><g data-editorial-node><rect x="25" y="240" width="310" height="86" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="259.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Interviewer: the high-risk slice</tspan><tspan x="180" dy="19">fell.</tspan><tspan x="180" dy="19">Candidate: hold that path;</tspan><tspan x="180" dy="19">investigate.</tspan></text></g><text x="18" y="349" style="font-size:12px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:start"><tspan x="18" dy="0">OBSERVED</tspan></text><text x="18" y="378" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Scopes harm; updates under pushback; states</tspan><tspan x="18" dy="19">a decision.</tspan></text><text x="18" y="431" style="font-size:12px;font-weight:700;fill:var(--viz-state-stroke);text-anchor:start"><tspan x="18" dy="0">UNKNOWN</tspan></text><text x="18" y="460" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Delivery ownership and scientific depth</tspan><tspan x="18" dy="19">were not tested.</tspan></text><path d="M180 496L180 520" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,513.8569220667674L180,520L183.35597877022943,513.8569220667674" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><text x="180" y="553" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Next drill: explain a slice test and its</tspan><tspan x="180" dy="19">uncertainty.</tspan></text></svg><div class="visual-derivation-caption">The exchange provides overlapping evidence, not five separate rounds or personality scores. Unknown delivery and scientific evidence stays unknown. Choose the next drill from the observed gap.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="as-five-cross-cutting-signals"><strong>Check:</strong> How can one technical follow-up reveal both scientific reasoning and response to pushback?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="as-five-cross-cutting-signals">A specific follow-up yields evidence about problem framing and experimentation here. Unknown signals remain unknown rather than being inferred from confidence.</span></figcaption></figure>

## Work the Mechanism

Since $dp/dz=p(1-p)$, the gradients are:

$$\frac{dL_{\mathrm{CE}}}{dz}=p-1=-0.99,\qquad \frac{dL_{\mathrm{sq}}}{dz}=(p-1)p(1-p)=-0.009801.$$

The extra sigmoid derivative shrinks this particular confident-wrong squared-loss gradient.
This does not describe every MSE problem: direct probability parameters or different loss scaling change the comparison.
Reconstruct the chain rule, then change the parameterization; memorizing the numbers does not close the gap.

## How to use this in prep

Record **attempt → observed gap → feedback → changed condition → next check**.
Use confirmed work samples. Choose useful timing/story counts rather than impose a fixed scoping timer or mandatory failures.

## Change 1: A Researcher or Engineer Changes Role

**New task.** A candidate moves to a different work profile.

**Picture and change.** Diagnose all five dimensions instead of inferring science or shipping competence from their old label.

**Check.** Practice allocation follows observed gaps.

**Use when.** Transition advice risks stereotyping.

## Change 2: The Criticism Is Partly Wrong

**New task.** A mock observer challenges a valid part of the answer.

**Picture and change.** Clarify and test the objection; accept supported evidence and retain justified boundaries.

**Check.** Neither automatic agreement nor louder repetition establishes correctness.

**Use when.** Follow-ups introduce conflicting claims.

## Failure: Confidence Instead of Evidence

Do not invent operating experience, rank personality or assume all prior researchers understand the mechanism.

## Transfer Check

Does every AS loop use these five exact dimensions?

### Reasoned Answer

No. They are practice lenses. Confirmed employer/role instructions govern the actual assessment.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## 1. Can you scope an ambiguous problem?

Ask about user/harm, latency/scale, success criterion, data/labels and cost/hardware/regulatory constraints.
Stop clarifying when the next decision is sufficiently defined, not after a fixed number of minutes.

<span id="2-have-you-actually-shipped-this"></span>
## 2. Show Relevant Delivery or Research Evidence

Use actual implementation, experiments or operations with clear ownership and limits.
Specific failures and diagnoses help when real; a candidate need not invent a killed project or a launch.

## 3. Do you understand the science, or just the recipe?

Rebuild one small mechanism and test a changed assumption.
Warmup has several possible optimization mechanisms, not a universal first-1,000-step explanation.
Dropout and normalization depend on architecture/objective. RLHF does not universally outperform supervised fine-tuning.
See [loss choice](/questions/how-to-choose-loss-function/).

## 4. Can you communicate, especially under disagreement?

Clarify the objection, test its evidence and explain what changed or remains justified.
Use [disagreement practice](/questions/disagreed-with-senior/), not a compulsory conversion story.

<span id="5-are-you-the-kind-of-person-we-want-in-the-room"></span>
## 5. Make Collaboration Observable

Check shared understanding, name uncertainty and make decision ownership clear.
Use safe mocks, written explanations or appropriate accommodations; interviewing for unwanted jobs is not required practice.

## Next-Drill Ledger

| Lens | Failure signal | Targeted drill | Transfer check |
| --- | --- | --- | --- |
| Scope | Undefined outcome | Frame population and consequence | Change scale |
| Evidence | Unsupported ownership | Audit one real artifact | Restrict disclosure |
| Mechanism | Recipe without assumptions | Derive a small case | Change parameterization |
| Critique | Ignores valid constraint | Clarify and test objection | Add a partly wrong objection |
| Collaboration | Unmarked uncertainty | Summarize assumptions and owners | Change spoken/written format |

## A note on the AS-specific bar

Inspect the current team's work and rubric. Do not derive technical competence from research/engineering identity or title alone.

---

*Related: [senior through senior-principal ML scope](/guides/l5-vs-l6-faang-ml/) for level calibration. [Applied Scientist vs MLE vs Research Engineer](/guides/as-vs-mle-vs-re/) for the role taxonomy.*

</div>
