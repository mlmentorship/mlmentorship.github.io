---
title: "Where an LLM belongs in a recommender"
description: "Choose a changed task, budget optional inference, compare simpler baselines and validate generated recommendations against the current catalog."
date: "2025-12-12"
draft: false
tags: ["questions"]
category: "questions"
---


Decide where a language model could improve a recommender while respecting catalog validity, serving load, and request deadlines.

## Problem

**Context.** A fictional recommender optionally sends a fraction of requests to an LLM reranker, with the workload specified below.

**Task.** Estimate arrival rate, mean in-flight work, and billed input-token load, then assess the proposed deadline.
Choose a useful LLM task and define quality checks, catalog validation, fallback, and rollout evidence.

**Constraints.** Apply the stated stable-flow boundary and do not infer tail latency from a mean service time.
Account for permitted evidence, item eligibility, and failure behavior; generated item names are not proof of valid catalog IDs.

A fictional recommender handles 1,000 requests/second.
An optional reranker receives 10% of requests.
Its mean service time is 0.1 seconds; each call bills 2,000 input tokens.
Assume stable flow, no retries and this service boundary only.
Estimate its load, and assess a proposed 50-millisecond hard request deadline.

## Picture: Optional Work Still Consumes Capacity

<!-- visual:recsys-llm-bounded-augmentation -->
<figure class="learning-figure" data-visual-id="recsys-llm-bounded-augmentation" aria-labelledby="recsys-budget-heading">
<p class="visual-kicker">Choose placement, not a universal replacement</p>
<p class="visual-title" id="recsys-budget-heading">Deadline fallback and catalog checks bound optional generated recommendations</p><div class="visual-argument" data-visual-argument="recsys-llm-bounded-augmentation" data-argument-kind="matrix"><p class="visual-argument-premise">A generated recommendation must satisfy deadline and catalog gates</p><svg viewBox="0 0 360 198" role="img" aria-label="A generated recommendation must satisfy deadline and catalog gates. A timely suggestion for an unknown item fails the catalog gate. A valid but late suggestion also falls back to the bounded baseline."><title>A generated recommendation must satisfy deadline and catalog gates</title><desc>A timely suggestion for an unknown item fails the catalog gate. A valid but late suggestion also falls back to the bounded baseline.</desc><text x="105.13333333333333" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="105.13333333333333" dy="0">on time</tspan></text><text x="199.08" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="199.08" dy="0">catalog</tspan></text><text x="293.02666666666664" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="293.02666666666664" dy="0">result</tspan></text><text x="46.16" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">case</tspan><tspan x="46.16" dy="18">A</tspan></text><rect data-proof-cell="0,0" x="58.16" y="36" width="90.94666666666667" height="47" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="103.63333333333333" y="65" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="103.63333333333333" dy="0">yes</tspan></text><rect data-proof-cell="0,1" x="152.10666666666668" y="36" width="90.94666666666667" height="47" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="197.58" y="65" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="197.58" dy="0">NO</tspan></text><path d="M156.10666666666668 79L239.05333333333334 40" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><rect data-proof-cell="0,2" x="246.05333333333334" y="36" width="90.94666666666667" height="47" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="291.5266666666667" y="65" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="291.5266666666667" dy="0">fallback</tspan></text><text x="46.16" y="104" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">case</tspan><tspan x="46.16" dy="18">B</tspan></text><rect data-proof-cell="1,0" x="58.16" y="86" width="90.94666666666667" height="47" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="103.63333333333333" y="115" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="103.63333333333333" dy="0">NO</tspan></text><path d="M62.16 129L145.10666666666668 90" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><rect data-proof-cell="1,1" x="152.10666666666668" y="86" width="90.94666666666667" height="47" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="197.58" y="115" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="197.58" dy="0">yes</tspan></text><rect data-proof-cell="1,2" x="246.05333333333334" y="86" width="90.94666666666667" height="47" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="291.5266666666667" y="115" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="291.5266666666667" dy="0">fallback</tspan></text><text x="46.16" y="154" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">case</tspan><tspan x="46.16" dy="18">C</tspan></text><rect data-proof-cell="2,0" x="58.16" y="136" width="90.94666666666667" height="47" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="103.63333333333333" y="165" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="103.63333333333333" dy="0">yes</tspan></text><rect data-proof-cell="2,1" x="152.10666666666668" y="136" width="90.94666666666667" height="47" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="197.58" y="165" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="197.58" dy="0">yes</tspan></text><rect data-proof-cell="2,2" x="246.05333333333334" y="136" width="90.94666666666667" height="47" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="291.5266666666667" y="165" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="291.5266666666667" dy="0">eligible</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 850" role="img" aria-labelledby="recsys-budget-title recsys-budget-desc" style="width:100%; min-width:0; height:auto; font-family:var(--font-ui); font-size:24px; fill:var(--c-text)">
<title id="recsys-budget-title">A deadline fallback and a catalog check bound optional LLM work</title>
<desc id="recsys-budget-desc">One declared counterexample has fast candidates A and B ready at five milliseconds and an optional LLM call taking one hundred milliseconds. A shared time ruler places the hard fifty-millisecond deadline halfway through that call. Serve a valid fast fallback by the deadline; the late LLM result cannot replace the completed reply. This is a declared call, not a tail-latency estimate from a mean. Every displayed item must pass current catalog and permission checks. The stated workload implies one hundred routed calls per second, ten mean calls in flight, and two hundred thousand billed input tokens per second. Offline content features are computed before requests. A separately generated ID Z missing from the catalog is rejected rather than displayed.</desc>
<text x="40" y="30" font-size="22">One declared request, not a latency percentile</text>
<rect x="40" y="50" width="520" height="114" rx="4" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)"/>
<text x="60" y="82" font-weight="700">FAST RETRIEVE + RANK</text>
<text x="60" y="118">Ready at 5 ms: candidates [A, B]</text>
<text x="60" y="148" font-size="22">Keep these IDs for the deadline fallback.</text>
<text x="40" y="200">Optional call for this request: 100 ms</text>
<g data-recsys-timeline="100" data-deadline-ms="50">
<rect x="60" y="222" width="480" height="26" fill="var(--viz-focus-bg)" stroke="var(--viz-focus-stroke)"/>
<path d="M60 252V262M300 210V262M540 252V262" fill="none" stroke="var(--viz-edge)" stroke-width="2"/>
<text x="60" y="288" font-size="22">0 ms</text>
<text x="300" y="288" text-anchor="middle" font-size="22" font-weight="700">50 ms deadline</text>
<text x="540" y="288" text-anchor="end" font-size="22">100 ms</text>
</g>
<text x="40" y="326" font-size="22" data-recsys-decision="fallback">50 ms: fallback served; LLM still busy.</text>
<text x="40" y="360" font-size="22">100 ms: late result cannot replace this reply.</text>
<rect x="40" y="388" width="520" height="102" rx="4" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)"/>
<text x="60" y="424" font-weight="700">Before display: live catalog + policy</text>
<text x="60" y="462" font-size="22">Fast and reranked IDs pass the same gate.</text>
<text x="40" y="534" font-size="22">Route load: 100/s; mean in flight = 10.</text>
<text x="40" y="566" font-size="22">Input billing: 200,000 tokens/s.</text>
<path d="M40 588H560" fill="none" stroke="var(--c-rule)"/>
<text x="40" y="626">Offline content -> stored features</text>
<text x="40" y="658" font-size="22">Compute before requests; refresh by version.</text>
<text x="40" y="704" font-size="22">Generated IDs need the same live check.</text>
<rect x="40" y="726" width="180" height="94" rx="4" fill="var(--viz-state-bg)" stroke="var(--viz-state-stroke)"/>
<text x="130" y="762" text-anchor="middle" font-size="22">Generated ID</text>
<text x="130" y="800" text-anchor="middle" font-weight="700">Z</text>
<path d="M220 774H270M260 766L270 774L260 782" fill="none" stroke="var(--viz-edge)" stroke-width="2"/>
<g data-generated-id="Z" data-catalog-resolution="missing" data-catalog-decision="reject">
<rect x="270" y="726" width="290" height="94" rx="4" fill="var(--viz-warning-bg)" stroke="var(--viz-warning-stroke)"/>
<text x="415" y="762" text-anchor="middle" font-size="22">Catalog: Z is missing</text>
<text x="415" y="800" text-anchor="middle" font-weight="700">HOLD / REJECT</text>
</g>
</svg>
<div class="visual-derivation-caption">the declared 100 ms call cannot finish before the 50 ms request deadline. Serve a catalog-valid fallback within the total request budget; timeout does not make work free. This example does not infer a percentile from mean service time. Offline features avoid per-request encoding, while a generated ID still needs current catalog resolution and permission.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="recsys-llm-bounded-augmentation"><strong>Check:</strong> What serves the request when optional reranking misses its deadline? Why can a generated item ID not bypass the current catalog?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="recsys-llm-bounded-augmentation">A timely suggestion for an unknown item fails the catalog gate. A valid but late suggestion also falls back to the bounded baseline.</span></figcaption>
</figure>

## Calculate the Routed Load

| Quantity | Calculation | Meaning |
| --- | --- | --- |
| Reranker arrivals | 1,000 x 0.10 = 100/second | This route only |
| Mean calls in service | 100 x 0.1 = 10 | Little's law at the service boundary |
| Billed input rate | 100 x 2,000 = 200,000 tokens/second | Output, retries and other stages additional |

Ten is mean occupancy, not a safe capacity limit or p99 bound.
As a counterexample, let every routed call take 100 milliseconds and every bypass take five.
The service assumptions hold, but 10% of whole requests already exceed 50 milliseconds before other work.
Use a deadline-aware optional route with valid fallback; count timeout/cancellation costs and quality loss.

<span id="what-hasnt-changed"></span>

## Contract: Budget the Stage, Validate the Output

Specify route predicate, candidate/token caps, deadlines, model/prompt/catalog identity and fallback compatibility.
Keep current existence, permission, region, availability and composition checks across every path.
Support explanations with actual decision evidence, not fluent guesses.
Predictive quality does not settle causal product value or participant policy.

<span id="common-follow-up"></span>

## Change 1: Move Content Work Offline

**New task.** New items have text, but request latency is strict.

**Picture and change.** Compare stored LLM and simpler content features under the same publication-freshness and cold-item evaluation.

**Check.** Classical content models also represent items without engagement.

**Use when.** Representation, not dialogue, is the bottleneck.

## Change 2: Clarify Constraints Through Conversation

**New task.** Users need iterative preference clarification.

**Picture and change.** Retrieve/rank through tools, confirm constraints and revalidate catalog choices.

**Check.** Compare task completion and interaction friction, not just pleasant chat.

**Use when.** Dialogue solves a real expression problem; preserve the effective low-friction surface otherwise.

<span id="tells-that-get-you-down-leveled"></span>

## Failure: A Plausible ID Is Displayed Without Resolution

A generated sequence can be invalid, ambiguous, stale or unauthorized.
An explanation cannot repair a missing item or invent its selection reason.

## Transfer Check

Does an unseen semantic-ID sequence prove discovery of a valid new item?

### Reasoned Answer

No. Resolve it against the versioned catalog and check identity, eligibility and representation quality.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

<span id="what-an-l5-answer-sounds-like"></span>
<span id="what-an-l6-answer-adds"></span>

## Compare at the Changed Boundary

| Placement | Compare and control |
| --- | --- |
| Intent | Rewrite/disambiguation versus simpler parsing; preserve user constraints and timeout behavior |
| Stored content | Matched simpler encoder, freshness, versioned vectors, publication cost and cold/warm slices |
| Top-N reranking | Relevance/satisfaction, bounded work, tail latency, billing, fallback and evidence-backed reasons |
| Conversation | Completion, useful clarification, privacy and extra effort versus existing surfaces |
| Generated retrieval | Semantic-token mapping, collisions/invalid IDs, updates, current catalog checks and supported relevance |

Semantic IDs encode catalog items as token sequences, not unrestricted permission to invent products.
New-item support depends on mapping/publication, representation and retrieval evidence.

Compare automated judges with human-reviewed slices, failure types and bias.
Judge agreement is not ground truth or causal preference.
Use product experiments or other justified identification for incremental outcomes.
Keep calibration, support, feedback, cold-start and diversity contracts appropriate to the selected architecture.
Neither behavioral dominance nor a universal two-stage design follows from using an LLM.

<span id="tells-that-get-you-a-strong-hire-vote"></span>

## Sources

These papers document particular designs, not current adoption or universal performance.

- [YouTube retrieve-then-rank architecture, 2016](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/).
- [Language-based item representations, 2023](https://arxiv.org/abs/2305.13731).
- [LLM reranking, 2023](https://arxiv.org/abs/2305.08845).
- [Generative retrieval with semantic IDs, 2023](https://arxiv.org/abs/2305.05065).

---

*Related: [Design YouTube's recommender](/questions/design-youtube-recommender/), [Two-tower vs cross-encoder: when to use which?](/questions/two-tower-vs-cross-encoder/), [How would you do cold-start for a new user?](/questions/cold-start-new-user/), [Designing a RAG system that actually works](/guides/designing-rag-that-works/).*

</div>
