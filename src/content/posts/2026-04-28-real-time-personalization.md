---
title: "Design real-time personalization"
description: "Real-time personalization fails most often at the data infrastructure, not the model. The senior answer designs the feature freshness and serving stack first."
date: "2026-04-28"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: recsys, search, and platform-ML interviews.*

Design personalization that responds to new user events within an explicitly defined freshness and request-latency budget.

## Problem

**Context.** A recommendation service should adapt to recent interactions, but ingestion, feature computation, retrieval, and ranking have different delays.
Events may arrive late, repeat, or be missing.

**Task.** Define what must update in real time, design the event-to-decision path, and specify measurements and fallback behavior.
Explain how training examples reproduce the information actually available at each historical decision.

**Constraints.** State freshness and latency objectives rather than treating real time as instantaneous.
Preserve privacy and eligibility, handle event ordering and duplicate processing, and distinguish feature freshness from model retraining frequency.

## Define "real-time" first

"Real-time" can mean any of:
- **Per-request**: features computed at request time from current state. Latency budget is tight (~100ms total).
- **Streaming-fresh**: features updated within seconds of an event (e.g., session-aware recommendations).
- **Near-real-time**: features updated within minutes (e.g., page-view aggregations).
- **Daily-batch**: stale by hours but still called "real-time" by some teams.

Each has very different infrastructure costs and use cases. Scope first.

<p class="visual-kicker">Learning objective</p>
<!-- visual:personalization-three-feature-clocks -->
<figure class="learning-figure editorial-illustration" aria-labelledby="personalization-three-feature-clocks-edition-title"><p class="visual-title" id="personalization-three-feature-clocks-edition-title">Low request latency does not guarantee fresh feature values</p><div class="visual-argument" data-visual-argument="personalization-three-feature-clocks" data-argument-kind="timeline"><p class="visual-argument-premise">A fast request can still use an old feature snapshot</p><svg viewBox="0 0 360 342" role="img" aria-label="A fast request can still use an old feature snapshot. The request completes quickly after reading a much earlier snapshot. Request latency and feature freshness use different timestamp differences."><title>A fast request can still use an old feature snapshot</title><desc>The request completes quickly after reading a much earlier snapshot. Request latency and feature freshness use different timestamp differences.</desc><text x="20" y="25" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Feature snapshot materialized</tspan></text><path d="M30 63L330 63" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="30" cy="63" r="5" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="30" y="92" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="30" dy="0">0</tspan></text><text x="20" y="122" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Request starts</tspan></text><path d="M30 160L330 160" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="327" cy="160" r="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="327" y="189" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="327" dy="0">9.9</tspan></text><text x="20" y="219" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Response returns</tspan></text><path d="M30 257L330 257" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="330" cy="257" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="330" y="286" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="330" dy="0">10</tspan></text><text x="180" y="316" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="180" dy="0">shared illustrative clock units</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 454" role="img" aria-labelledby="personalization-three-feature-clocks-edition-svg-title personalization-three-feature-clocks-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="personalization-three-feature-clocks-edition-svg-title">A fast request can still assemble stale features</title><desc id="personalization-three-feature-clocks-edition-svg-desc">A declared illustrative noon request uses an 08:00 batch snapshot with a 24-hour limit, a stream value whose event time is 11:57 and arrival 11:59 with a two-minute event-age limit, and current request context. The batch age is four hours and passes; stream event age is three minutes and fails despite one-minute arrival age. Request execution latency is separate.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Illustrative request at noon; declared age</tspan><tspan x="180" dy="19">contracts</tspan></text><text x="90" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">SOURCE CLOCK</tspan></text><text x="270" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">AGE / CONTRACT</tspan></text><path d="M15 91L345 91" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="111" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Batch snapshot</tspan></text><g data-editorial-node><rect x="15" y="125" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="163.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">08:00</tspan></text></g><g data-editorial-node><rect x="195" y="125" width="150" height="67" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="163.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">4 h &lt;= 24 h</tspan></text></g><path d="M15 207L345 207" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="227" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Stream event / arrival</tspan></text><g data-editorial-node><rect x="15" y="241" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="279.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">11:57 / 11:59</tspan></text></g><g data-editorial-node><rect x="195" y="241" width="150" height="67" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="279.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">3 min &gt; 2 min</tspan></text></g><path d="M15 323L345 323" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="343" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Live request context</tspan></text><g data-editorial-node><rect x="15" y="357" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="395.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">12:00</tspan></text></g><g data-editorial-node><rect x="195" y="357" width="150" height="67" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="395.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">current; age 0</tspan></text></g></svg><div class="visual-derivation-caption">The stream value arrived recently but describes an older event. Enforce each feature's declared clock contract. Request latency is not source freshness.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="personalization-three-feature-clocks"><strong>Check:</strong> At this request, which value is fresh enough by its contract? Which stale value should trigger fallback or an alert?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="personalization-three-feature-clocks">The request completes quickly after reading a much earlier snapshot. Request latency and feature freshness use different timestamp differences.</span></figcaption></figure>

## What an L5 answer sounds like

> "Architecture, in three pieces:
>
> **Feature pipeline.**
> - **Batch features** (computed daily or hourly): user history aggregates, item embeddings, graph features. Stored in a feature store.
> - **Streaming features** (computed seconds to minutes after event): session activity, recent-clicks features, real-time engagement counts. Computed in a stream processor (Flink, Spark Streaming, Kafka Streams) and written to the feature store.
> - **Request-time features**: context features (time, device, location), query embeddings. Computed at request time.
>
> **Online serving.**
> - Request comes in, fetches batch + streaming features from the feature store, computes request-time features, calls the model, returns predictions.
> - Latency budget allocation: feature fetch (10-30ms), model inference (20-50ms), application logic (10-30ms), network (10-20ms). Total p99 around 100-150ms.
>
> **Training pipeline.**
> - Joins logged production features with delayed labels (e.g., did the user click the recommended item).
> - Critical: training and serving must use *exactly* the same feature pipeline. Skew between them is the dominant production failure mode. Use the feature store for both, or generate from a shared library.
>
> **Eval**: offline (held-out logged data, counterfactual estimators), online (A/B test). Plus monitoring: feature staleness, feature null-rates, model score distribution drift."

This is L5. Three layers, latency budget allocated, training-serving skew called out.

## What an L6 answer adds

> "...practical things:

> **Training-serving skew is the dominant production failure mode.** Same feature computed differently in training and serving (different aggregation window, different null handling, different units) produces a model that works offline and fails online. The fix is mechanical: shared feature definitions, validation that feature distributions match between training and serving, alerting on divergence.
>
> **Feature freshness vs cost trade-off.** Streaming features (sub-second freshness) cost orders of magnitude more than batch features. Most features don't need streaming freshness; reserve it for features where freshness directly drives quality (session-aware ranking, just-clicked items).
>
> **Caching at multiple levels.** User feature vectors cached per request session. Model predictions cached for popular request signatures. Each cache adds staleness; tune the TTL based on the cost of staleness vs the cost of the request.
>
> **Embedding update strategies.** Item embeddings change as the model retrains. Online inference might use embedding version V1 in candidate generation and V2 in ranking; this breaks. Either pin the version per request or update everything atomically.
>
> **Cold-start in real-time.** New users / items don't have batch features ready; serving must handle missing features gracefully. Default values, fallback models, or a separate cold-start path.
>
> **Monitoring**: feature freshness lag, feature null-rate, model score distribution per-slice. The most common silent failure is a streaming pipeline falling behind; the model still serves but with stale features and quality degrades silently. Alert aggressively on staleness."

## Tells that get you a strong-hire vote

- You **scope 'real-time'** first.
- You name **training-serving skew** as the dominant failure mode.
- You allocate the **latency budget** explicitly.
- You discuss **caching** at multiple levels.
- You insist on **freshness monitoring**.
- You bring up **embedding-version atomicity**.

## Tells that get you down-leveled

- Model-first design.
- No latency budget.
- Treating "real-time" as one thing.
- No mention of training-serving skew.

## Common follow-up

"What if your streaming feature pipeline is down for an hour. What does the system do?"

The L6 answer:

> "Three patterns. (1) Serve stale features with monitoring: model still works, quality degrades by an amount you can quantify (depending on which features were stale). (2) Fall back to a model variant trained without those features (graceful degradation). (3) Worst-case fallback: serve cached predictions or simple heuristics (popularity-based recommendations). The right choice depends on which is least bad for the use case. Plan all three before launching; test each in chaos exercises before they're needed."

---

*Related: [delayed and selective labels](/concepts/delayed-labels-selective-labels-feedback-loops/), [point-in-time correctness](/concepts/data-leakage-point-in-time-correctness/), [YouTube recommendation design](/questions/design-youtube-recommender/), and [personalized search ranking](/guides/personalized-search-ranking/).*
