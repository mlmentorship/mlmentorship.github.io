---
title: "Design an ML system under a fixed serving budget"
description: "A cost-constrained system-design question where quality, latency, traffic, and annual spend must fit one defensible operating point."
date: "2026-07-10"
draft: false
tags: ["questions"]
category: "questions"
---

> Design a personalized ranking system for 20 million daily users. The serving budget is $3 million per year and p95 latency must stay below 80 ms. What do you ship?

The constraint is the question. A candidate who proposes the best model without estimating cost has not designed the requested system.

## Clarify before calculating

- Requests per user per day and peak-to-average traffic
- Candidate set size and ranking stages
- Existing infrastructure and hardware prices
- Quality target and minimum worthwhile gain
- Batchability, cacheability, and freshness requirements
- Availability target and fallback behavior
- Whether the budget includes feature storage, logging, retraining, and on-call overhead

## Build an order-of-magnitude model

Suppose there are 100 million ranking requests per day:

- Average QPS: roughly 1,160
- At a 5× peak factor: roughly 5,800 peak QPS
- Annual serving budget: about $8,200 per day
- Cost ceiling: roughly $0.08 per thousand requests before non-serving costs

The exact estimates will change. Writing them down prevents an architecture whose cost is off by an order of magnitude.

## A strong architecture progression

1. Establish a cheap heuristic or linear/tree baseline.
2. Use a multi-stage system: retrieval, light pre-ranker, expensive ranker only on a small set.
3. Precompute stable embeddings and cache popular candidates.
4. Batch where latency allows and quantize only after measuring the bottleneck.
5. Distill or simplify the expensive model if quality per dollar is poor.
6. Add graceful fallback and cost/latency monitoring by stage.
7. Run an experiment that includes incremental value **and** incremental cost.

## What an L4 answer sounds like

> “Use a two-tower model and autoscale GPUs.”

This is plausible technology but does not prove the design fits traffic, latency, or budget.

## What an L5 answer adds

An L5 candidate estimates traffic and unit economics, proposes a baseline, allocates latency and cost across stages, and defines a quality-versus-cost experiment. They know which values must be measured before committing.

## What an L6 answer adds

An L6 candidate treats cost as a product portfolio decision:

- Which user segments justify expensive ranking?
- Can spend be dynamic by request value or uncertainty?
- Would better candidate generation beat a larger ranker?
- What organizational incentives prevent cost from drifting upward?
- What is the marginal value curve beyond the selected operating point?
- Should the team build infrastructure or buy capacity?

## Strong-hire signals

- Order-of-magnitude math before naming hardware.
- Cost per successful outcome, not cost per request alone.
- Quality/cost/latency frontier rather than one “best” model.
- Explicit fallback and overload behavior.
- Monitoring that attributes spend to stage, model, and traffic segment.

## Down-leveling tells

- Ignoring peak traffic.
- Assuming quantization always improves latency.
- Spending the entire budget on model inference while omitting features and logging.
- No baseline or staged rollout.
- Treating annual budget as someone else’s problem.

## Likely follow-ups

- Traffic doubles with no budget increase. What changes first?
- The largest model adds 1% conversion but triples cost. Do you ship it?
- How do prefill and decode change the cost model for an LLM ranker?
- What if only 5% of requests need personalization?
- How do you detect that caching is silently harming freshness?

*Related: [reduce LLM inference cost 10×](/questions/reduce-llm-inference-cost-10x/), [GPU memory hierarchy](/concepts/gpu-memory-hierarchy/), and [quantization](/concepts/quantization/).*
