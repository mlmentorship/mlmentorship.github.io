---
title: "Analyze an ML experiment in SQL"
description: "Compute exposure-aware experiment metrics without double-counting users, leaking post-treatment data, or hiding sample-ratio problems."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> You have assignment and event tables for an ML experiment. Write SQL to compute exposed users, conversion, average latency, and a sample-ratio check by variant.

Tables:

```text
assignments(user_id, variant, assigned_at)
events(user_id, event_time, event_name, latency_ms)
```

The interviewer is testing data grain and causal hygiene more than SQL syntax.

## Clarify the contract

- Is assignment unique and sticky per user?
- What event marks first valid exposure?
- What is the conversion window?
- Can events occur before assignment?
- Is conversion binary per user or counted per event?
- Are bots, employees, or corrupted clients excluded?

## A defensible query shape

```sql
WITH assigned AS (
  SELECT user_id, variant, MIN(assigned_at) AS assigned_at
  FROM assignments
  GROUP BY 1, 2
),
exposure AS (
  SELECT
    a.user_id,
    a.variant,
    MIN(e.event_time) AS exposed_at
  FROM assigned a
  JOIN events e
    ON e.user_id = a.user_id
   AND e.event_time >= a.assigned_at
   AND e.event_name = 'model_exposed'
  GROUP BY 1, 2
),
user_outcomes AS (
  SELECT
    x.user_id,
    x.variant,
    MAX(CASE WHEN e.event_name = 'conversion'
              AND e.event_time < x.exposed_at + INTERVAL '7 day'
             THEN 1 ELSE 0 END) AS converted,
    AVG(CASE WHEN e.event_name = 'model_response'
             THEN e.latency_ms END) AS avg_latency_ms
  FROM exposure x
  LEFT JOIN events e
    ON e.user_id = x.user_id
   AND e.event_time >= x.exposed_at
  GROUP BY 1, 2
)
SELECT
  variant,
  COUNT(*) AS exposed_users,
  AVG(converted * 1.0) AS conversion_rate,
  AVG(avg_latency_ms) AS avg_user_latency_ms
FROM user_outcomes
GROUP BY 1;
```

The exact dialect is secondary. The important choice is one row per randomized user before aggregation.

## Sample-ratio check

Compute assigned counts separately from exposed counts. A 50/50 assignment that becomes 60/40 exposure can signal treatment delivery failure. Do not “fix” it by dropping inconvenient users without understanding why.

## What an L4 answer sounds like

A correct join and group-by, but event rows are counted directly. Heavy users dominate conversion and duplicate assignment rows multiply outcomes.

## What an L5 answer adds

- Establishes user grain before calculating rates
- Filters post-assignment and post-exposure windows explicitly
- Separates intention-to-treat from treatment-on-treated views
- Checks duplicate or cross-variant assignments
- Computes assignment and exposure sample ratios
- Explains null latency and users with no outcome

## What an L6 answer adds

An L6 candidate asks whether the query can support the causal decision:

- Did treatment change whether exposure was logged?
- Is exclusion post-treatment and therefore biased?
- Is the unit really user, or should it be account or cluster?
- Are late events and backfills stable enough for decision time?
- Should the metric pipeline be versioned and independently reconciled?

## Common mistakes

- Joining all events before deduplicating users
- Counting conversions before assignment
- Using the assignment timestamp as exposure
- Averaging event latency so heavy users dominate
- Silently excluding unexposed assigned users
- Ignoring users assigned to multiple variants

## Likely follow-ups

- Write the duplicate-assignment diagnostic.
- Compute intention-to-treat conversion.
- Add a pre-experiment covariate without leaking treatment information.
- How would cluster randomization change the query?
- How would you detect a broken event logger in one variant?

*Related: [A/B testing for ML](/concepts/ab-testing-for-ml/) and [design an ML A/B test](/questions/design-ml-ab-test/).*
