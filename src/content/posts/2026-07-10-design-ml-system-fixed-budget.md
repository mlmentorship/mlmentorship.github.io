---
title: "Design an ML system under a fixed serving budget"
description: "A cost-constrained system-design question where quality, latency, traffic, and annual spend must fit one defensible operating point."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> Design a personalized ranking system for 20 million daily users. The serving budget is $3 million per year and p95 latency must stay below 80 ms. What do you ship?

Design a personalized ranking service that fits an explicit annual serving budget and request-latency requirement.

## Problem

**Context.** The hypothetical product has 20 million daily users, a $3 million annual serving budget, and an 80 ms p95 limit.
Requests per user, peak traffic, catalog size, and infrastructure prices are not yet specified.

**Task.** State assumptions, estimate traffic and allowable unit cost, and choose retrieval, ranking, caching, and fallback stages.
Show a feasible budget and a measurement plan, or identify which requirement must change.

**Constraints.** Daily users are not requests per second; separate averages, peaks, and tail latency.
Include replicated capacity, data access, and operating overhead within the declared budget boundary rather than counting accelerator inference alone.

## Clarify before calculating

Ask how many requests each user makes, how traffic peaks and what the annual budget includes.
Also define quality, freshness, availability and fallback requirements.
These determine which operating points are allowed.

## Build an order-of-magnitude model

Assume five requests per daily user: 100 million requests per day.
Use 365 days and a fivefold peak factor for this example.

| Quantity | Calculation | Result |
| --- | --- | --- |
| Average requests/second | $100{,}000{,}000/86{,}400$ | About 1,157 |
| Assumed peak | $5\times1{,}157.4$ | About 5,787 requests/second |
| Annual requests | $100{,}000{,}000\times365$ | 36.5 billion |
| Daily budget | $3{,}000{,}000/365$ | About $8,219 |
| Unit ceiling | $3{,}000{,}000/36.5\text{ billion}\times1{,}000$ | About $0.08219 per thousand |

The unit ceiling covers the declared budget scope, not just model kernels.
Subtract included feature, logging, storage and operational costs before allocating inference spend.
Excluded costs still belong in a separately reported total.

<!-- visual:fixed-budget-ranking-funnel -->
<figure class="learning-figure editorial-illustration" aria-labelledby="fixed-budget-ranking-funnel-edition-title"><p class="visual-title" id="fixed-budget-ranking-funnel-edition-title">Shrink candidates before spending expensive scoring work</p><div class="visual-argument" data-visual-argument="fixed-budget-ranking-funnel" data-argument-kind="bars"><p class="visual-argument-premise">Spend expensive scoring only after narrowing</p><svg viewBox="0 0 360 394" role="img" aria-label="Spend expensive scoring only after narrowing. A true count scale makes the 10,000-to-100 reduction visible. Stage widths are not latency measurements."><title>Spend expensive scoring only after narrowing</title><desc>A true count scale makes the 10,000-to-100 reduction visible. Stage widths are not latency measurements.</desc><text x="18" y="23" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Retrieved candidates</tspan></text><path d="M28 64L332 64" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="10000" data-proof-min="0" data-proof-max="10000" data-proof-zero="28" x="28" y="46" width="304" height="35" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></rect><text x="28" y="106" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="28" dy="0">10000 candidates</tspan></text><text x="18" y="140" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Pre-ranked candidates</tspan></text><path d="M28 181L332 181" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="1000" data-proof-min="0" data-proof-max="10000" data-proof-zero="28" x="28" y="163" width="30.400000000000006" height="35" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.75"></rect><text x="28" y="223" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-state-stroke)"><tspan x="28" dy="0">1000 candidates</tspan></text><text x="18" y="257" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Final scoring candidates</tspan></text><path d="M28 298L332 298" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="100" data-proof-min="0" data-proof-max="10000" data-proof-zero="28" x="28" y="280" width="3.039999999999999" height="35" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><text x="28" y="340" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="28" dy="0">100 candidates</tspan></text><text x="28" y="374" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="28" dy="0">0</tspan></text><text x="332" y="374" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="332" dy="0">10000</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 473" role="img" aria-labelledby="fixed-budget-ranking-funnel-edition-svg-title fixed-budget-ranking-funnel-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="fixed-budget-ranking-funnel-edition-svg-title">Spend expensive scoring only after shrinking the candidate set</title><desc id="fixed-budget-ranking-funnel-edition-svg-desc">Explicitly illustrative candidate counts shrink from 10,000 to 1,000 to 100. Widths communicate ordered narrowing rather than a linear count scale. The separate request-level target remains p95 below 80 milliseconds and cost at most 0.08 dollars per 1,000 requests. Stage p95s are not added as if they were request p95.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Illustrative counts, not measured stage</tspan><tspan x="180" dy="19">latencies</tspan></text><g data-editorial-node><rect x="25" y="67" width="310" height="59" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="92" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">retrieve</tspan><tspan x="180" dy="19">10,000 candidates</tspan></text></g><path d="M180 126L180 150" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,143.8569220667674L180,150L183.35597877022943,143.8569220667674" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="65" y="154" width="230" height="59" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="180" y="179" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">pre-rank</tspan><tspan x="180" dy="19">1,000 candidates</tspan></text></g><path d="M180 213L180 237" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,230.8569220667674L180,237L183.35597877022943,230.8569220667674" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="105" y="241" width="150" height="59" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="180" y="266" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">final rank</tspan><tspan x="180" dy="19">100 candidates</tspan></text></g><text x="180" y="348" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Widths show narrowing, not a linear count</tspan><tspan x="180" dy="19">scale.</tspan></text><text x="180" y="391" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Measure request p95 &lt; 80 ms and cost &lt;= $0.08 /</tspan><tspan x="180" dy="19">1K.</tspan></text><text x="180" y="437" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Overload: fewer candidates or a bounded</tspan><tspan x="180" dy="19">fallback.</tspan></text></svg><div class="visual-derivation-caption"><strong>Read it this way:</strong> The counts are illustrative, and funnel widths are schematic. Measure latency and cost for the whole request. Under overload, reduce work or use fallback.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="fixed-budget-ranking-funnel"><strong>Check:</strong> Where should expensive work be spent? Which work can be removed before exceeding the shared request budget?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="fixed-budget-ranking-funnel">A true count scale makes the 10,000-to-100 reduction visible. Stage widths are not latency measurements.</span></figcaption></figure>

## A defensible architecture progression

Start with a heuristic, linear or tree baseline. Measure a quality, cost and latency point before adding stages.
For an illustrative funnel, retrieval keeps 200 of 10,000 candidates; the final ranker scores 20 of those 200.
Measure retrieval recall: no final ranker can recover a relevant item already discarded.

Precompute stable embeddings and reuse candidates when their age is acceptable.
Batch only within the latency limit. Test quantization or distillation against the measured bottleneck.
They can reduce work without improving latency if another stage dominates.

Measure end-to-end p95 under representative peak load, including queueing, features, network calls and fallbacks.
Adding stage p95 values does not produce the request's p95.
Track per-stage work and cost to explain the total.

## Change the operating point

### 1. Only 5% of requests benefit from expensive ranking

**New task:** concentrate extra work where it improves the outcome.

**Picture and change:** put a router before the expensive branch.
Measure missed opportunities and segment quality as well as the fraction routed.
Include router cost and shared work.

**Check:** 5% observed use does not prove that the other 95% never benefits.

**Use when:** measured marginal value differs across requests.

### 2. The cache meets cost but returns stale candidates

**New task:** retain useful results as items and users change.

**Picture and change:** include versioned keys, invalidation and an age limit.
Measure outcome quality by age, plus miss and fallback cost.

**Check:** a high hit rate can coexist with unacceptable freshness.

**Use when:** reuse trades computation against changing eligibility or relevance.

## Try doubled traffic

Traffic doubles without a budget increase. What happens to the unit ceiling?

<details>
<summary>Check your answer</summary>

It halves to about $0.04110 per thousand requests.
Peak capacity must also be retested. The old operating point no longer proves feasibility.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

<a id="what-an-l4-answer-sounds-like"></a>

## Check a hardware proposal

"Use a two-tower model and autoscale GPUs" does not supply a capacity or cost calculation.
Record workload shapes, current prices, utilization, batching limits and availability reserves.
Autoscaling cannot make an unaffordable peak fit the budget.

<a id="what-an-l5-answer-adds"></a>

## Choose a feasible point

Compare quality, full cost and end-to-end tail latency under the same traffic assumptions.
Include a measured degraded path, rollout limit and rollback trigger.
Use cost per useful outcome alongside cost per request.

<a id="what-an-l6-answer-adds"></a>

## Compare further investment

Test better candidate generation against a larger ranker.
Compare building infrastructure with buying capacity, including maintenance and price changes.
Stop adding work when its measured value no longer justifies its cost.

<a id="tells-that-get-you-a-strong-hire-vote"></a>

## Evidence for a recommendation

A claimed 1% conversion gain at triple cost needs a causal value estimate and actual cost accounting.
Check uncertainty, cheaper alternatives and the hard budget before recommending shipment.

<a id="tells-that-get-you-down-leveled"></a>

## Failure cases

A larger final model cannot repair retrieval that dropped every relevant item.
An inference-only spreadsheet cannot establish total serving cost.

## Common follow-ups

- Traffic doubles with no budget increase. What changes first?
- The largest model adds 1% conversion but triples cost. Do you ship it?
- How do prefill and decode change the cost model for an LLM ranker?
- What if only 5% of requests need personalization?
- How do you detect that caching is silently harming freshness?

*Related: [reduce LLM inference cost 10×](/questions/reduce-llm-inference-cost-10x/), [GPU memory hierarchy](/concepts/gpu-memory-hierarchy/), and [quantization](/concepts/quantization/).*

</div>
