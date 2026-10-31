---
title: "How would you reduce LLM inference cost by 10x?"
description: "The cost-engineering question. The L6 answer doesn't pick a technique, it diagnoses where the cost is, then picks five."
date: "2026-02-05"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: LLM-team and platform interviews, especially at companies serving LLMs at scale.*

Investigate whether an LLM service can reduce inference cost tenfold without violating its quality, latency, or reliability requirements.

## Problem

**Context.** A hypothetical production service has a measured baseline workload and spend, but the dominant cost components are not yet identified.
The requested tenfold reduction is a target, not an established feasible outcome.

**Task.** Define cost per useful completed request, attribute the baseline, prioritize interventions, and propose controlled evaluations and rollout.
Estimate combined savings without double-counting and explain what would make the target infeasible.

**Constraints.** Hold the compared task mix and quality contract explicit.
Include retries, cache coverage, routing failures, output lengths, and operating costs; cheaper requests that fail more often may not save money.

<p class="visual-kicker">Learning objective</p>
<!-- visual:inference-cost-measure-test-compound-loop -->
<figure class="learning-figure editorial-illustration" aria-labelledby="inference-cost-measure-test-compound-loop-edition-title"><p class="visual-title" id="inference-cost-measure-test-compound-loop-edition-title">Accept savings only after quality gates, then update the cost baseline</p><div class="visual-argument" data-visual-argument="inference-cost-measure-test-compound-loop" data-argument-kind="bars" data-argument-composition="endpoint"><svg viewBox="0 0 360 394" role="img" aria-label="Rejected savings do not become the next baseline. The 8-unit change passes quality and replaces the 10-unit baseline. The 6-unit change fails tail quality, so the accepted baseline stays 8."><title>Rejected savings do not become the next baseline</title><desc>The 8-unit change passes quality and replaces the 10-unit baseline. The 6-unit change fails tail quality, so the accepted baseline stays 8.</desc><text x="18" y="23" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Original accepted baseline</tspan></text><path d="M28 64L332 64" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="10" data-proof-min="0" data-proof-max="10" data-proof-zero="28" x="28" y="46" width="304" height="35" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></rect><circle cx="332" cy="63.5" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M332 69L332 86" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="255.5" y="106" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="255.5" dy="0">10 toy cost units</tspan></text><text x="18" y="140" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Change 1: quality passes</tspan></text><path d="M28 181L332 181" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="8" data-proof-min="0" data-proof-max="10" data-proof-zero="28" x="28" y="163" width="243.20000000000005" height="35" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><circle cx="271.20000000000005" cy="180.5" r="4" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><path d="M271.20000000000005 186L271.20000000000005 203" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><text x="260" y="223" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="260" dy="0">8 toy cost units</tspan></text><text x="18" y="257" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Change 2: quality fails; reject</tspan></text><path d="M28 298L332 298" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="6" data-proof-min="0" data-proof-max="10" data-proof-zero="28" x="28" y="280" width="182.4" height="35" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></rect><circle cx="210.4" cy="297.5" r="4" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><path d="M210.4 303L210.4 320" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="210.4" y="340" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="210.4" dy="0">6 toy cost units</tspan></text><text x="28" y="374" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="28" dy="0">0</tspan></text><text x="332" y="374" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="332" dy="0">10</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 430" role="img" aria-labelledby="inference-cost-measure-test-compound-loop-edition-svg-title inference-cost-measure-test-compound-loop-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="inference-cost-measure-test-compound-loop-edition-svg-title">Count only savings that survive the same quality gates</title><desc id="inference-cost-measure-test-compound-loop-edition-svg-desc">Toy fixed-mix cost per successful request starts at ten units. Change one reduces it to eight and passes required quality slices, so eight becomes the new baseline. Change two reaches six but fails a tail-quality gate and is rejected. Accepted reduction is ten over eight, 1.25-fold, not an unsupported tenfold claim.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Toy cost units per successful request;</tspan><tspan x="180" dy="19">fixed task mix</tspan></text><text x="20" y="66" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">BASELINE: 10 units</tspan></text><rect x="25" y="86" width="310" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="20" y="162" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">CHANGE 1: 8 units; quality gates pass</tspan></text><rect x="25" y="182" width="248" height="29" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="20" y="258" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">CHANGE 2: 6 units; tail quality fails</tspan></text><rect x="25" y="278" width="186" height="29" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3;stroke-dasharray:5 3"></rect><text x="180" y="340" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Reject change 2. Current baseline stays 8, not</tspan><tspan x="180" dy="19">6.</tspan></text><text x="180" y="387" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">10 / 8 = 1.25x cost reduction; no 10x claim.</tspan></text></svg><div class="visual-derivation-caption">Keep change 1 and reject change 2. The accepted baseline is 8 units per successful request. Cost reduction is 1.25x, not a fictional 10x.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="inference-cost-measure-test-compound-loop"><strong>Check:</strong> Which proposed saving is rejected? Why must the second accepted saving be calculated from the updated baseline?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="inference-cost-measure-test-compound-loop">The 8-unit change passes quality and replaces the 10-unit baseline. The 6-unit change fails tail quality, so the accepted baseline stays 8.</span></figcaption></figure>

## What an L4 answer sounds like

> "We could use a smaller model, or quantize to INT8 or INT4. We could also use a cheaper API."

Three techniques, no diagnosis, no prioritization. Each technique is right but the answer isn't structured.

## What an L5 answer sounds like

> "Before picking techniques I'd want to know where the cost is actually coming from. There are usually five layers to LLM cost:
>
> 1. Per-request token cost (the model + tokens).
> 2. System prompt amortization (the prompt you pay on every request).
> 3. Retrieved context (RAG chunks, conversation history).
> 4. Multi-step / agentic overhead (chained LLM calls).
> 5. Tail cost (the 1-5% of requests that consume disproportionate compute).
>
> The mix differs by workload. Attribute each billed call and token category once; do not infer the dominant term from a chatbot or agent label. The [inference-cost ledger](/guides/llm-inference-cost/) separates repeated context, retries, model prices, and total request cost.
>
> Assuming a typical RAG-backed agent, my prioritized levers:
>
> - **Prompt caching:** price reads, writes, minimum lengths, hit rate, and expiry for the actual provider/version. A stable prefix alone does not establish a saving or an implementation schedule.
> - **Retrieval:** compare bounded context policies on required-evidence recall and answer quality. Fewer chunks can lose necessary evidence, and reranking itself has a cost.
> - **Tiered models:** measure routing accuracy, fallback frequency, and every attempted call. A cheaper first model can cost more overall if it creates extra retries or regressions.
> - **Output control:** test whether shorter outputs still satisfy the task. A token cap can truncate necessary work rather than remove waste.
> - **Cap iteration count for agent loops**: prevents runaway cost on hard requests; one of the easiest tail-cost reductions.
>
> Prioritize by measured cost contribution, implementation effort, and quality risk. Caching, retrieval changes, and model changes all need a valid comparison; no intervention is universally free or guaranteed to achieve the requested tenfold target."

This is L5. You've named the cost decomposition, prioritized correctly (cheap structural wins first), and acknowledged the quality risk of the deeper changes.

## What an L6 answer sounds like

The L6 answer adds the things that come from running cost optimization for several quarters:

> "...and a few practical considerations beyond the techniques:
>
> **You need a cost dashboard before you optimize.** Without per-request cost telemetry by feature, by user segment, and at p50 / p95 / p99, you can't know which changes actually moved the needle. I'd build that first; it usually surfaces the biggest wins automatically.
>
> **Quality regressions from cost changes are the failure mode.** Every cost reduction is a quality risk. Each change should be A/B tested against the baseline; cost savings without a quality eval are not real savings. We've seen 'cost-saving' optimizations get rolled back because they tanked a key metric.
>
> **Some cost optimizations are contagious in good ways.** Reducing context length helps cost, but it also reduces latency (less prefill), which improves agent step time, which lets you cap iterations earlier, which reduces total cost further. Optimizations compound.
>
> **Smaller-model-per-step is often the largest single win.** Most agentic systems use one model for everything. In reality, the planning step needs different capabilities than the synthesis step than the tool-calling step. Tiered serving (small model for routing, medium for planning, large only for synthesis) can substantially reduce cost with quality on par.
>
> **Distillation is a real option but expensive to do well.** If you have enough production data and a quality eval, you can distill a 70B model down to a 7B that handles 80% of your traffic and falls back to 70B for the remaining 20%. This is a months-long project but pays off significantly at scale.
>
> **Self-hosted vs API:** calculate a workload-specific crossover using utilization, accelerator and engineering costs, peak capacity, model quality, latency, and provider pricing. Monthly API spend alone does not determine the answer.
>
> **Evaluation turnaround:** reducing a run from four hours to twenty minutes is a 12x reduction in that runtime, not a guarantee of 12x experiment throughput. Human review, queues, and other stages remain. Preserve test coverage and failure sensitivity while shortening the loop."

This is L6. You've gone past the techniques into the *operational discipline* of cost reduction, with specific patterns from running this in production.

## The tells that get you a strong-hire vote

- You **decompose where the cost is** before naming techniques.
- You **prioritize cheap structural wins** (prompt caching, better retrieval) over deep technique changes (smaller model, quantization).
- You bring up **quality A/B testing** as a gate on cost changes.
- You mention **tiered serving** (different models for different steps), the largest underrated win.
- You discuss **monitoring and dashboards** as a precondition for optimization.

## The tells that get you down-leveled

- "Just use a smaller model" as the first answer.
- "Quantize to INT8" as the only technique mentioned.
- No mention of prompt caching (the cheapest single win in 2026).
- Treating cost as a model-choice question rather than a system-design question.
- No mention of quality measurement.

## A common follow-up

"What's the trade-off you're most worried about?"

The L6 answer:

> "The cost-vs-quality regression on the long tail. Most of my cost optimizations work great on the median request and then fail on the 5% of hard requests, the ones with sparse context, edge-case queries, or unusual user behavior. The aggregate cost goes down, the aggregate quality goes down slightly, and the tail quality goes down a lot. If those tail users are also the high-value users, you've made a bad trade you can't see in your average metrics.
>
> The mitigation is rigorous tail monitoring. Track quality at p50 and p95 separately. Cut by user segment and request type. Gate releases on no-regression in the high-stakes slices, even if the average is up. This is the same pattern as in any system optimization, the average is for the leadership deck; the per-slice table is what actually drives decisions."

If you can have this conversation, you're at the senior bar.

---

*Related: [How to think about LLM inference cost](/guides/llm-inference-cost/) for the long-form treatment, [Designing a RAG system that actually works](/guides/designing-rag-that-works/).*
