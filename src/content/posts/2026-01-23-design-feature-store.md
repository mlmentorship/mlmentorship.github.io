---
title: "Design a feature store from scratch"
description: "A feature store solves training-serving skew, feature reuse, and lineage. The senior answer explains why each property matters and what minimum viable looks like."
date: "2026-01-23"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML platform and infra interviews.*

Design shared feature infrastructure that reproduces training examples and serves consistent, sufficiently fresh features for online decisions.

## Problem

**Context.** Several models reuse features computed from event streams and historical tables.
Training needs past feature values, while serving needs low-latency current values with known freshness and ownership.

**Task.** Specify feature definitions, computation, historical storage, online serving, metadata, and point-in-time dataset construction.
Explain backfills, late events, validation, failure recovery, and when a feature store is unnecessary or should be purchased.

**Constraints.** Prevent future or unavailable information from entering historical examples.
State event-time and availability semantics, version contracts, access controls, freshness targets, and consistency choices rather than promising perfect synchronization.

## Three problems a feature store solves

1. **Training-serving skew**: features computed in training don't match features computed at serving. The dominant ML production failure mode.
2. **Feature reuse**: every team rebuilds the same features (user demographics, product attributes). Duplication, inconsistency, wasted compute.
3. **Lineage and reproducibility**: which features did this model use, with what definitions, computed from what data, at what time?

If you don't have these problems at scale, you don't need a feature store. Most teams do at scale.

## What an L5 answer sounds like

> "Components:
>
> **Feature definition layer.** Features are defined as named, typed transformations of source data. Definitions live in code (Python, often), versioned in git. The same definition is used in batch (training) and online (serving) computation.
>
> **Offline store.** A historical record of all feature values over time. Used for training: given an event timestamp, fetch the feature values *as they would have been* at that time (point-in-time correctness). Typically a columnar warehouse (BigQuery, Snowflake, Iceberg).
>
> **Online store.** Latest feature values for fast lookup at serving time. Typically a low-latency KV store (Redis, DynamoDB, Bigtable). Updated by the same pipeline that writes to the offline store.
>
> **Materialization pipeline.** Computes features from source data, writes to both offline and online stores. Batch (Spark, dbt) for slow-changing features; streaming (Flink, Kafka) for fresh features.
>
> **Serving API.** At inference, the application calls `get_features(entity_id, feature_names)` and the store returns latest values. SLA: single-digit milliseconds for a feature lookup of ~100 features.
>
> **Lineage / metadata.** Which model used which features, when, with what definitions. Backed by metadata DB.
>
> Reference implementations: Feast (open source), Tecton (commercial), Hopsworks, Vertex AI Feature Store, Databricks Feature Store."

<p class="visual-kicker">Learning objective</p>
<!-- visual:feature-store-two-time-contract -->
<figure class="learning-figure editorial-illustration" aria-labelledby="feature-store-two-time-contract-edition-title"><p class="visual-title" id="feature-store-two-time-contract-edition-title">The historical join and the live lookup answer different time questions</p><svg viewBox="0 0 360 454" role="img" aria-labelledby="feature-store-two-time-contract-edition-svg-title feature-store-two-time-contract-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="feature-store-two-time-contract-edition-svg-title">The historical join and the live lookup answer different time questions</title><desc id="feature-store-two-time-contract-edition-svg-desc">For illustrative entity E, value 3 occurs at 11:00 and is available at 11:05. Value 7 occurs at 11:50 but is available at 12:10. A noon historical query must return 3. A live 12:15 lookup can return 7. Event time alone is insufficient for the historical contract.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Illustrative entity E; historical query at</tspan><tspan x="180" dy="19">noon</tspan></text><text x="90" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">VERSION A</tspan></text><text x="270" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">VERSION B</tspan></text><path d="M15 91L345 91" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="111" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Value and event time</tspan></text><g data-editorial-node><rect x="15" y="125" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="163.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">3 at 11:00</tspan></text></g><g data-editorial-node><rect x="195" y="125" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="270" y="163.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">7 at 11:50</tspan></text></g><path d="M15 207L345 207" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="227" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Actually available</tspan></text><g data-editorial-node><rect x="15" y="241" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="279.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">11:05</tspan></text></g><g data-editorial-node><rect x="195" y="241" width="150" height="67" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="279.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">12:10</tspan></text></g><path d="M15 323L345 323" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="343" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Historical noon / live 12:15</tspan></text><g data-editorial-node><rect x="15" y="357" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="395.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">noon returns 3</tspan></text></g><g data-editorial-node><rect x="195" y="357" width="150" height="67" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="395.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">12:15 returns 7</tspan></text></g></svg><figcaption><strong>Read it this way:</strong> At noon, version B is not available, despite its earlier event time. The later live lookup legitimately returns another version. Values are illustrative. <span class="visual-reader-check" data-reader-check="feature-store-two-time-contract"><strong>Check:</strong> Why can a correct live lookup return a value that would leak into a historical training example?</span></figcaption></figure>

This is L5. Components named, point-in-time correctness explicit, build-or-buy options listed.

## What an L6 answer adds

> "...practical points:
>
> **Point-in-time correctness is the hardest part.** When generating training data, you need feature values *as they were at the time of the event*, not as they are now. Naive joins use current values and leak future information. Implementation options: time-travel queries on a versioned store, append-only event-log architectures, or careful materialized views. Get this wrong and your offline metrics overestimate; the model fails in production.
>
> **Feature freshness vs cost.** Streaming features cost orders of magnitude more than batch. Tag each feature with a freshness SLA; route to batch by default; promote to streaming only when the use case demands it.
>
> **Schema evolution is painful.** Adding fields is easy; renaming or changing types breaks downstream consumers. Schema registries, deprecation processes, and dual-writes during transitions are required at scale.
>
> **Build vs buy**: building a custom feature store is months of work and ongoing maintenance. For most teams, an off-the-shelf option (Feast / Tecton / cloud-provider equivalent) is the right call. Build only if you have very specific requirements (extreme scale, on-prem, niche compliance).
>
> **The feature store is not magic for ML quality.** It solves the *infrastructure* problem of features being available, fresh, and consistent. It doesn't fix the *modeling* problem of which features to compute. Teams sometimes treat the feature store as the project goal; it's an enabler for actual ML work.
>
> **Feature store as serverless interface**: modern systems push toward 'declarative' features (you specify what, the system figures out how to compute and serve). Reduces application boilerplate but adds a layer of abstraction that can be hard to debug."

## Tells that get you a strong-hire vote

- You name the **three problems** (training-serving skew, reuse, lineage) up front.
- You explain **point-in-time correctness** correctly.
- You separate **offline and online stores** as different SLA tiers.
- You discuss **build vs buy** explicitly.
- You acknowledge that **feature stores enable**, don't replace, modeling work.

## Tells that get you down-leveled

- "Use Feast" with no further detail.
- No mention of point-in-time correctness.
- Conflating offline and online stores.
- No discussion of feature freshness tiers.

## Common follow-up

"What's the simplest viable feature store for a small team?"

The L6 answer:

> "For a small team: don't build one. Start with a feature library (a Python module with shared feature-computing functions). Use the same module in both your training pipeline and your serving code. Cache computed features in your existing data warehouse for offline and a Redis instance for online. This gives you most of the benefit (consistency, reuse) without the operational overhead of a full feature-store platform. Promote to a real feature-store product when feature count grows past ~50 and you have multiple teams contributing."

---

*Related: [multi-team ML platform design](/questions/design-multi-team-ml-platform/), [ML data lineage and versioning](/concepts/ml-data-lineage-versioning/), [point-in-time correctness](/concepts/data-leakage-point-in-time-correctness/), and [personalized search ranking](/guides/personalized-search-ranking/).*
