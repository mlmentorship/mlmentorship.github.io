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
<figure class="learning-figure editorial-illustration" aria-labelledby="as-five-cross-cutting-signals-edition-title"><span class="visual-fragment-anchor" id="as-signals-title" aria-hidden="true"></span><p class="visual-title" id="as-five-cross-cutting-signals-edition-title">Use an observed mechanism gap to choose a drill, then change the assumption</p><div class="visual-argument" data-visual-argument="as-five-cross-cutting-signals" data-argument-kind="matrix" data-argument-composition="practice-feedback-loop"><svg viewBox="0 0 360 2020" role="img" aria-label="Use an observed mechanism gap to choose a drill, then change the assumption. An observed error selects a drill, not a personality judgment. In this sigmoid case, squared loss has an extra derivative factor under the same logit. Reconstruct that chain, then change to direct probability parameters: the gradients change. Record the attempt, feedback and transfer check. Other skills remain unobserved until supported by work. These five practice lenses are not a universal hiring rubric; clarify and test objections rather than agreeing automatically."><title>Use an observed mechanism gap to choose a drill, then change the assumption</title><desc>An observed error selects a drill, not a personality judgment. In this sigmoid case, squared loss has an extra derivative factor under the same logit. Reconstruct that chain, then change to direct probability parameters: the gradients change. Record the attempt, feedback and transfer check. Other skills remain unobserved until supported by work. These five practice lenses are not a universal hiring rubric; clarify and test objections rather than agreeing automatically.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Attempt, observed gap, drill,</tspan><tspan x="20" dy="18">changed-condition check</tspan></text><text x="20" y="113" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Fictional attempt: squared loss</tspan><tspan x="20" dy="18">always gives small gradients. Missing</tspan><tspan x="20" dy="18">assumption: the output</tspan><tspan x="20" dy="18">parameterization.</tspan></text><text x="20" y="228" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Same logit; sigmoid probability 0.01;</tspan><tspan x="20" dy="18">target one. Squared loss uses half</tspan><tspan x="20" dy="18">the squared error.</tspan></text><circle cx="58" cy="363" r="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="58" y="369" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="58" dy="0">Logit</tspan></text><path d="M96 363L142 363M134 367L142 363L134 359" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="180" cy="363" r="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="180" y="369" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">Sigmoid</tspan></text><path d="M218 363L264 363M256 367L264 363L256 359" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="302" cy="363" r="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="302" y="369" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="302" dy="0">p 0.01</tspan></text><text x="20" y="490" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Cross-entropy logit gradient</tspan></text><rect x="20" y="522" width="297" height="25" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1"></rect><text x="20" y="591" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Signed value: -0.990000</tspan></text><text x="20" y="655" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Squared-loss logit gradient</tspan></text><rect x="20" y="687" width="2.9403" height="25" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1"></rect><text x="20" y="756" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Signed value: -0.009801</tspan></text><text x="20" y="845" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Extra sigmoid factor: 0.0099. It</tspan><tspan x="20" dy="18">shrinks this particular squared-loss</tspan><tspan x="20" dy="18">gradient. Same bar scale; not a claim</tspan><tspan x="20" dy="18">about every MSE problem.</tspan></text><text x="20" y="985" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Transfer: parameterize probability</tspan><tspan x="20" dy="18">directly. Cross-entropy gradient is</tspan><tspan x="20" dy="18">minus one hundred; squared-loss</tspan><tspan x="20" dy="18">gradient is minus 0.99. Removing the</tspan><tspan x="20" dy="18">sigmoid changes the comparison.</tspan></text><text x="20" y="1138" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Next drill: reconstruct the chain</tspan><tspan x="20" dy="18">rule, then explain the changed</tspan><tspan x="20" dy="18">parameterization. Memorizing two</tspan><tspan x="20" dy="18">numbers does not close the gap.</tspan></text><text x="20" y="1320" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Scope</tspan></text><text x="210" y="1320" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text-soft)"><tspan x="210" dy="0">Unobserved</tspan></text><text x="20" y="1426" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Evidence</tspan></text><text x="210" y="1426" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text-soft)"><tspan x="210" dy="0">Unobserved</tspan></text><text x="20" y="1532" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Mechanism</tspan></text><text x="210" y="1532" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="210" dy="0">Gap observed</tspan></text><text x="20" y="1638" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Critique</tspan></text><text x="210" y="1638" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text-soft)"><tspan x="210" dy="0">Unobserved</tspan></text><text x="20" y="1744" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Collaboration</tspan></text><text x="210" y="1744" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text-soft)"><tspan x="210" dy="0">Unobserved</tspan></text><text x="20" y="1895" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Clarify partly wrong criticism and</tspan><tspan x="20" dy="18">test its evidence. Other skills need</tspan><tspan x="20" dy="18">separate observations. Use current</tspan><tspan x="20" dy="18">role instructions, not a universal</tspan><tspan x="20" dy="18">five-part hiring rubric.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 595" role="img" aria-labelledby="as-five-cross-cutting-signals-edition-svg-title as-five-cross-cutting-signals-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="as-five-cross-cutting-signals-edition-svg-title">One exchange can expose several skills and leave others unknown</title><desc id="as-five-cross-cutting-signals-edition-svg-desc">A fictional mock asks whether an overall completion gain justifies shipping. The candidate asks about harmed users and stop criteria, then updates after a high-risk slice failure. Scoping, responsiveness, and explicit decision-making are observed; delivery ownership and scientific depth remain untested. The next drill targets uncertainty in the slice test.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional mock; observable evidence, not</tspan><tspan x="180" dy="19">personality scores</tspan></text><g data-editorial-node><rect x="25" y="62" width="310" height="62" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="79" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Interviewer: overall completion is</tspan><tspan x="180" dy="19">up.</tspan><tspan x="180" dy="19">Would you ship?</tspan></text></g><g data-editorial-node><rect x="25" y="151" width="310" height="62" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="180" y="177.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Candidate: which users lose?</tspan><tspan x="180" dy="19">What is the stop threshold?</tspan></text></g><g data-editorial-node><rect x="25" y="240" width="310" height="86" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="259.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Interviewer: the high-risk slice</tspan><tspan x="180" dy="19">fell.</tspan><tspan x="180" dy="19">Candidate: hold that path;</tspan><tspan x="180" dy="19">investigate.</tspan></text></g><text x="18" y="349" style="font-size:12px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:start"><tspan x="18" dy="0">OBSERVED</tspan></text><text x="18" y="378" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Scopes harm; updates under pushback; states</tspan><tspan x="18" dy="19">a decision.</tspan></text><text x="18" y="431" style="font-size:12px;font-weight:700;fill:var(--viz-state-stroke);text-anchor:start"><tspan x="18" dy="0">UNKNOWN</tspan></text><text x="18" y="460" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Delivery ownership and scientific depth</tspan><tspan x="18" dy="19">were not tested.</tspan></text><path d="M180 496L180 520" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,513.8569220667674L180,520L183.35597877022943,513.8569220667674" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><text x="180" y="553" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Next drill: explain a slice test and its</tspan><tspan x="180" dy="19">uncertainty.</tspan></text></svg><div class="visual-derivation-caption">The exchange provides overlapping evidence, not five separate rounds or personality scores. Unknown delivery and scientific evidence stays unknown. Choose the next drill from the observed gap.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="as-five-cross-cutting-signals"><strong>Check:</strong> How can one technical follow-up reveal both scientific reasoning and response to pushback?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="as-five-cross-cutting-signals">An observed error selects a drill, not a personality judgment. In this sigmoid case, squared loss has an extra derivative factor under the same logit. Reconstruct that chain, then change to direct probability parameters: the gradients change. Record the attempt, feedback and transfer check. Other skills remain unobserved until supported by work. These five practice lenses are not a universal hiring rubric; clarify and test objections rather than agreeing automatically. <strong>Does one corrected derivative establish delivery or collaboration skill?</strong></span></figcaption></figure>

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
