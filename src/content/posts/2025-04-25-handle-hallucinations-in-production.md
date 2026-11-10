---
title: "How do you handle hallucinations in production?"
description: "There is no single solution. The senior answer is a layered system that catches different hallucination types at different stages."
date: "2025-04-25"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: LLM-team interviews, especially in regulated domains (healthcare, fintech, legal-tech).*

Design a production response to unsupported model claims without assuming that fluent text or a citation makes an answer correct.

## Problem

**Context.** In a hypothetical document-answering service, some generated claims lack support in the authorized sources.
Sources can also be outdated, and the service sometimes lacks enough evidence to answer.

**Task.** Define failure categories, detection and prevention measures, fallback behavior, and an evaluation and incident-response plan.
Explain what happens when the system cannot substantiate a consequential claim.

**Constraints.** Preserve source access controls and distinguish factual support from formatting or citation presence.
Account for latency, abstention, and false alarms. Do not promise zero hallucinations or treat a detector score as proof.

## What an L4 answer sounds like

> "We can use RAG to ground the model in real documents, then the hallucination problem mostly goes away. We can also use a fact-checking model to verify outputs, and use temperature 0 to make it more deterministic."

Each of these is partially right and individually insufficient. RAG reduces hallucinations of facts the documents contain but does nothing for hallucinations of facts they don't. Fact-checking models hallucinate too. Temperature 0 affects determinism, not factuality.

you've read about hallucinations in tutorials but haven't fought them in production.

## What an L5 answer sounds like

> "Hallucinations come from a few different mechanisms and each needs a different mitigation:
>
> 1. **Knowledge gaps**: the model doesn't know the answer and confabulates. Fix: ground in retrieval (RAG), instruct to refuse if not in source.
> 2. **Reasoning errors**: the model has the facts but draws a wrong conclusion. Fix: chain-of-thought, multi-step decomposition, sometimes multiple sampled paths with self-consistency.
> 3. **Context confusion**: the model conflates facts from different parts of the input. Fix: shorter context, explicit citation requirements, structured prompts.
> 4. **Confident wrong outputs in long-tail**: rare cases where the model is just confident and wrong. Fix: human-in-the-loop for high-stakes decisions, confidence calibration, post-hoc verification.
>
> In production I'd build a layered system:
>
> - **Pre-generation**: ground the model in retrieved documents; restrict the prompt to ask for cited claims.
> - **Generation**: instruct 'if the answer is not in the provided sources, say so'; use a structured output format that forces citations.
> - **Post-generation**: a verification pass that checks each claim against its cited source. Flag ungrounded claims; either retry or surface to the user with a warning.
> - **Out-of-band**: a periodic eval set focused on hallucination, with metrics tracked release-over-release.
>
> Importantly: the goal isn't zero hallucinations, that's not achievable with current technology. The goal is *acceptable hallucination rate for the use case*, with detection and graceful handling for the rest."

<!-- visual:hallucination-claim-release-gates -->
<figure class="learning-figure" data-visual-id="hallucination-claim-release-gates" aria-labelledby="claim-release-heading">
<p class="visual-kicker">Learning objective</p>
<p class="visual-title" id="claim-release-heading">Require claim-level evidence and release permission rather than agreement or citation presence</p><div class="visual-argument" data-visual-argument="hallucination-claim-release-gates" data-argument-kind="matrix" data-argument-composition="claim-evidence-release"><svg viewBox="0 0 360 2400" role="img" aria-label="Require claim-level evidence and release permission rather than agreement or citation presence. Five answers can repeat one unsupported promise. Agreement, fluent reasoning and resolving citations do not prove claim support. Check authorized, current passages and separate disclosure permission. Unsupported claims need abstention, controlled correction or review; supported private claims still stop at release. Independently label verifier errors and confidence signals by slice. Track unsupported claims, false alarms, refusals, latency and incidents. No detector or reviewer guarantees zero hallucinations."><title>Require claim-level evidence and release permission rather than agreement or citation presence</title><desc>Five answers can repeat one unsupported promise. Agreement, fluent reasoning and resolving citations do not prove claim support. Check authorized, current passages and separate disclosure permission. Unsupported claims need abstention, controlled correction or review; supported private claims still stop at release. Independently label verifier errors and confidence signals by slice. Track unsupported claims, false alarms, refusals, latency and incidents. No detector or reviewer guarantees zero hallucinations.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Agreement is not evidence; support is</tspan><tspan x="20" dy="18">not permission</tspan></text><text x="20" y="117" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Five sampled answers repeat an</tspan><tspan x="20" dy="18">approval promise. The authorized</tspan><tspan x="20" dy="18">passage does not contain that</tspan><tspan x="20" dy="18">promise.</tspan></text><circle cx="38" cy="275" r="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="38" y="281" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="38" dy="0">1</tspan></text><path d="M38 302L180 392M171.10154145973206 391.0958785440369L180 392L175.3842220406097 384.3387602942078" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="109" cy="275" r="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="109" y="281" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="109" dy="0">2</tspan></text><path d="M109 302L180 392M171.9046846989946 388.19659755254486L180 392L178.18553277736086 383.2417062907226" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="180" cy="275" r="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="180" y="281" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">3</tspan></text><path d="M180 302L180 392M176 384L180 392L184 384" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="251" cy="275" r="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="251" y="281" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="251" dy="0">4</tspan></text><path d="M251 302L180 392M181.81446722263914 383.2417062907226L180 392L188.0953153010054 388.19659755254486" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="322" cy="275" r="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="322" y="281" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="322" dy="0">5</tspan></text><path d="M322 302L180 392M184.6157779593903 384.3387602942078L180 392L188.89845854026794 391.0958785440369" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="180" cy="435" r="34" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="180" y="441" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">No</tspan></text><text x="20" y="532" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Claim support is absent. Five copies</tspan><tspan x="20" dy="18">can share one wrong assumption; token</tspan><tspan x="20" dy="18">likelihood is not the chance the</tspan><tspan x="20" dy="18">claim is true.</tspan></text><text x="20" y="735" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Unsupported promise</tspan></text><circle cx="70" cy="821" r="25" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="70" y="827" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="70" dy="0">no</tspan></text><text x="70" y="880" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="70" dy="0">Support</tspan></text><circle cx="290" cy="821" r="25" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="290" y="827" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="290" dy="0">yes</tspan></text><text x="290" y="880" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="290" dy="0">Release</tspan></text><path d="M100 821L260 821" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="948" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Abstain, correct with evidence, or</tspan><tspan x="20" dy="18">route to review; do not release the</tspan><tspan x="20" dy="18">unsupported claim.</tspan></text><text x="20" y="1085" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Supported private claim</tspan></text><circle cx="70" cy="1171" r="25" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="70" y="1177" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="70" dy="0">yes</tspan></text><text x="70" y="1230" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="70" dy="0">Support</tspan></text><circle cx="290" cy="1171" r="25" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="290" y="1177" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="290" dy="0">no</tspan></text><text x="290" y="1230" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="290" dy="0">Release</tspan></text><path d="M100 1171L260 1171" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="1298" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Evidence supports it, but disclosure</tspan><tspan x="20" dy="18">is not authorized. Stop at the</tspan><tspan x="20" dy="18">separate release gate.</tspan></text><text x="20" y="1435" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Supported releasable claim</tspan></text><circle cx="70" cy="1521" r="25" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="70" y="1527" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="70" dy="0">yes</tspan></text><text x="70" y="1580" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="70" dy="0">Support</tspan></text><circle cx="290" cy="1521" r="25" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="290" y="1527" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="290" dy="0">yes</tspan></text><text x="290" y="1580" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="290" dy="0">Release</tspan></text><path d="M100 1521L260 1521" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="1648" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Both required gates pass in this</tspan><tspan x="20" dy="18">stipulated case; other task checks</tspan><tspan x="20" dy="18">still apply.</tspan></text><text x="20" y="1835" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">A citation can resolve without its</tspan><tspan x="20" dy="18">passage entailing the claim. Check</tspan><tspan x="20" dy="18">source access, freshness and every</tspan><tspan x="20" dy="18">consequential claim, not just answer</tspan><tspan x="20" dy="18">formatting.</tspan></text><text x="20" y="1980" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Knowledge gaps, reasoning errors and</tspan><tspan x="20" dy="18">context confusion need different</tspan><tspan x="20" dy="18">tests. RAG, retries, low temperature,</tspan><tspan x="20" dy="18">sample agreement and human review are</tspan><tspan x="20" dy="18">not proof of factuality.</tspan></text><text x="20" y="2130" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Independently label verifier false</tspan><tspan x="20" dy="18">accepts/rejects and confidence by</tspan><tspan x="20" dy="18">population/slice. Track unsupported</tspan><tspan x="20" dy="18">claims, refusals, false alarms,</tspan><tspan x="20" dy="18">latency and versioned incidents.</tspan></text><text x="20" y="2280" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Choose fallback and release</tspan><tspan x="20" dy="18">thresholds from consequence and</tspan><tspan x="20" dy="18">measured errors. Refusal can be</tspan><tspan x="20" dy="18">correct; no detector or reviewer</tspan><tspan x="20" dy="18">establishes zero hallucinations.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 760" role="img" aria-labelledby="claim-gates-title claim-gates-desc" style="width:100%; min-width:0; height:auto; font-family:var(--font-ui); font-size:24px; fill:var(--c-text)">
<title id="claim-gates-title">Claim support and release permission are separate gates</title>
<desc id="claim-gates-desc">An invented authorized source says applications are reviewed within five days and approval is not guaranteed. The claim that review takes at most five days is supported. The claim that the applicant will be approved within five days is not, despite citing the same passage. Unsupported claims are held for correction, abstention, or escalation, not released with a warning. A supported claim still needs current evidence and a separate policy, permission, and consequence check; consequential decisions require authorized review. Automated support checks remain fallible.</desc>
<rect x="40" y="18" width="520" height="136" rx="4" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)"/>
<text x="60" y="50" font-weight="700">Fictional authorized passage [1]</text>
<text x="60" y="90">Applications are reviewed within 5 days.</text>
<text x="60" y="124">Approval is not guaranteed.</text>
<g data-claim-support="supported">
<path d="M110 154V184" fill="none" stroke="var(--viz-edge)" stroke-width="2"/>
<rect x="40" y="184" width="520" height="122" rx="4" fill="var(--viz-state-bg)" stroke="var(--viz-state-stroke)"/>
<text x="60" y="216" font-weight="700">Claim A: support check passes</text>
<text x="60" y="254">Review takes at most 5 days. [1]</text>
<text x="60" y="286" font-size="22">Same event and bound as the passage.</text>
</g>
<g data-claim-support="unsupported" data-claim-decision="hold">
<path d="M530 154V332H510" fill="none" stroke="var(--viz-edge)" stroke-width="2" stroke-dasharray="6 4"/>
<rect x="40" y="332" width="470" height="142" rx="4" fill="var(--viz-warning-bg)" stroke="var(--viz-warning-stroke)"/>
<text x="60" y="364" font-weight="700">Claim B: HOLD, even with [1]</text>
<text x="60" y="402">You will be approved within 5 days.</text>
<text x="60" y="438" font-size="22">Review is not approval.</text>
<text x="60" y="464" font-size="22">Correct, abstain, or escalate.</text>
</g>
<path d="M110 306H24V494H140V514" fill="none" stroke="var(--viz-state-stroke)" stroke-width="3"/>
<rect x="40" y="514" width="520" height="154" rx="4" fill="var(--viz-focus-bg)" stroke="var(--viz-focus-stroke)"/>
<g data-claim-release-gate="required-for-supported">
<text x="60" y="548" font-weight="700">Claim A needs release permission</text>
<text x="60" y="584" font-size="22">Is the evidence current? Is this use allowed?</text>
<text x="60" y="616" font-size="22">Consequential decision: authorized review.</text>
<text x="60" y="648" font-size="22">No policy clearance -> HOLD or abstain.</text>
</g>
<text x="40" y="710" font-size="22">Support checks can err; audit real failures.</text>
<text x="40" y="742" font-size="22">Neither a citation nor a pass proves truth.</text>
</svg>
<div class="visual-derivation-caption">compare each claim with the same passage. The unsupported approval promise must stop, while the supported review-time claim reaches a separate release-policy gate, not automatic publication. The passage is an invented example. Automated entailment checks remain fallible. Original synthesis informed by <a href="https://arxiv.org/abs/2005.11401">Lewis et al. on retrieval-augmented generation</a>, <a href="https://aclanthology.org/2023.emnlp-main.398/">Gao et al. on citation correctness</a>, <a href="https://aclanthology.org/2023.emnlp-main.741/">Min et al. on atomic factual evaluation</a>, and the <a href="https://doi.org/10.6028/NIST.AI.600-1">NIST Generative AI Profile</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="hallucination-claim-release-gates"><strong>Check:</strong> Which exact passage supports the highlighted claim? What separate release gate still applies in a high-consequence use?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="hallucination-claim-release-gates">Five answers can repeat one unsupported promise. Agreement, fluent reasoning and resolving citations do not prove claim support. Check authorized, current passages and separate disclosure permission. Unsupported claims need abstention, controlled correction or review; supported private claims still stop at release. Independently label verifier errors and confidence signals by slice. Track unsupported claims, false alarms, refusals, latency and incidents. No detector or reviewer guarantees zero hallucinations. <strong>Do five agreeing samples justify releasing the unsupported approval promise?</strong></span></figcaption>
</figure>

This is L5. You've decomposed the problem, named specific mitigations per type, and acknowledged the operational reality.

## What an L6 answer sounds like

The L6 answer adds the things that come from running this in production for a couple of years:

> "...and a few more things I've learned the hard way:
>
> **Citation-checking is a separate task.** A resolving citation does not prove that its passage supports the claim. A verifier can score a claim/passage pair, but its false accepts and false rejects need independent, representative labels. A narrower task does not guarantee a lower error rate. The [RAG mechanism](/concepts/rag-overview/) separates authorized retrieval, claim support, and release permission.
>
> **Refusal is a quality, not a failure.** A model that says 'I don't know' on the questions it shouldn't answer is *better* than one that confidently makes things up. Train your team and your eval to reward refusals on out-of-scope questions.
>
> **Pure RAG isn't enough for adversarial inputs.** Users will ask things that look like the documents but aren't actually in them. The model will pattern-match and hallucinate. Mitigations: explicit instruction to refuse if no relevant passage found, retrieval thresholds (don't pass low-confidence retrievals to the model), distinct prompt branches for 'high confidence retrieval' vs 'low confidence retrieval'.
>
> **Self-consistency is a candidate method, not a verifier.** Multiple samples can improve some tasks, but correlated errors can make the majority confidently wrong. Compare accuracy and total resource cost with the same baseline budget, including any selection or verification step.
>
> **Agreement is not a calibrated probability.** Five answers can repeat the same unsupported claim. Token likelihood is also not the probability that a factual claim is true. Define the event, gather independent labels, and evaluate any proposed confidence signal by population and slice before setting abstention thresholds.
>
> **Plausible reasoning can still be unsupported.** Require evidence for consequential intermediate claims, not merely a fluent final answer. Depending on the task, deterministic checks, independent source review, restricted actions, abstention, and expert review can complement a failure-focused evaluation set. Human review is not an infallible final oracle."

This is L6. You've gone past the techniques into the *operational discipline* of managing hallucinations in a real product, with specific examples from your own experience.

**Predict before revealing:** five sampled answers all repeat an approval promise absent from the authorized passage. Does agreement justify releasing the promise?

<details><summary>Test support independently of repetition</summary>

No. The five answers may share one mistaken assumption.
The claim remains unsupported, and any separate release-authorization requirement still applies.
Agreement can be an evaluated feature, not a substitute for evidence.

</details>

## The tells that get you a strong-hire vote

- You **decompose the problem by hallucination type**: not as one undifferentiated thing.
- You acknowledge that **zero hallucinations is not the goal**; acceptable rate is.
- You bring up **citation-checking as a separate verification step**.
- You mention **calibration vs raw confidence** and that LLMs are uncalibrated.
- You distinguish **easy hallucinations** (factual errors) from **subtle ones** (plausible-but-wrong reasoning).

## The tells that get you down-leveled

- "Just use RAG", oversimplifies; doesn't address knowledge-not-in-source cases.
- "Use a fact-checking API", vague; the LLM-team interviewer wants to know how *you* would build this.
- "Temperature 0", affects determinism, not factuality.
- No mention of refusal as a valid output.
- No mention of A/B-testing or release-over-release tracking of a hallucination metric.
- Treating the question as a model-choice question instead of a system-design question.

## A common follow-up

"How would you measure your hallucination rate?"

The L6 answer:

> "Hallucination rate is hard to measure because the ground truth is judgment, not a label. I'd build it in layers:
>
> 1. **A golden set of ~200 question/answer pairs** with hand-labeled correct answers. Run the model, have humans (or an LLM-judge calibrated against humans) compare outputs to references. Compute a faithfulness rate.
> 2. **For RAG specifically: claim-level verification**: decompose each output into atomic claims, check each against the retrieved sources. Compute the fraction of unsupported claims.
> 3. **Targeted adversarial set**: questions designed to elicit hallucinations (out-of-scope queries, questions about non-existent things, questions with subtle factual traps). Track the refusal rate and the hallucination rate separately.
> 4. **Production sample audit**: randomly sample N production responses per week, have humans review.
>
> The numbers from these don't combine into a single 'hallucination rate' because they measure different things. I'd report all of them and tell the team which trends to watch."

If you can have this conversation fluently, you're at the senior bar.

---

*Related: [How would you evaluate an LLM application?](/questions/how-would-you-evaluate-an-llm-application/), [Designing a RAG system that actually works](/guides/designing-rag-that-works/), [LLM Evals essay](/guides/llm-evals-the-hardest-part/).*
