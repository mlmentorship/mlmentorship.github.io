---
title: "Design a production LLM inference service"
description: "Turn prefill, decode, KV memory, admission, batching, fairness, overload, and cost into one explicit serving policy."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> Design a multi-tenant service for a 70B model. Traffic mixes 200-token chat prompts with 32K document prompts. Define the SLOs, capacity model, scheduler, overload behavior, and evaluation plan.

Design a multi-tenant language-model service that handles short chat and long document prompts under explicit capacity and latency requirements.

## Problem

**Context.** A hypothetical service hosts a 70B model and mixes 200-token chat prompts with 32K-token document prompts.
Output lengths, arrival rates, hardware, and tenant service tiers must be clarified.

**Task.** Define service objectives, estimate compute and cache capacity, and design scheduling, admission, isolation, overload handling, and rollout evaluation.
Explain the trade-off between time to first token, inter-token latency, throughput, fairness, and cost.

**Constraints.** Account separately for prompt processing and iterative generation, including future cache growth and cancellation.
State workload and precision assumptions; naming a serving framework does not establish feasible capacity or tail-latency guarantees.

## Split the two phases

**Prefill** processes prompt tokens in parallel. Large prefills with efficient matrix shapes are often compute-bound and drive time to first token. Short prompts or small batches can be limited by memory traffic or launch overhead.

**Decode** produces one token per active sequence per iteration. Small decode batches are usually memory-bandwidth-bound because every step reads model weights and KV state for little new arithmetic. Large batches can reach a compute limit. Decode drives inter-token latency.

A scheduler that treats both as interchangeable work will let long prefills stall decode or will starve new prompts to protect active streams.

## Decide whether to separate prefill and decode

A shared worker pool is the simpler starting point. Chunked prefill lets one scheduler mix prompt work with active decode.

At larger scale, **disaggregated serving** places prefill and decode on separate worker pools. Prefill workers produce KV state and transfer it to decode workers. This can:

- isolate long-prompt work from inter-token latency;
- scale the two hardware pools independently;
- use different parallel layouts or accelerator types for each phase;
- keep decode batches full.

It also adds a queue boundary, KV transfer, failure states, and network load. Approximate transfer time as:

$$
T_{\text{transfer}} \approx \frac{\text{KV bytes for the prompt}}{\text{effective network bandwidth}}.
$$

Do not disaggregate when that transfer and queueing cost is larger than the scheduling interference it removes.

Measure accelerator-seconds of prefill and decode work on the real request distribution. At the same target utilization, the ratio of prefill workers to decode workers should start near the ratio of those two offered workloads. Then load-test tail latency and rebalance.

## Quantify before drawing

For a 70B model, start with:

- weight memory by precision;
- KV bytes per token;
- prompt and generation distributions, not only averages;
- arrival rate and burst factor;
- target time to first token and inter-token latency by traffic class;
- GPU topology and effective memory bandwidth;
- tokens per second by prefill and decode regime.

KV memory per request is approximately:

$$
2 \times L \times H_{kv} \times d_h \times T \times b,
$$

where $L$ is layer count, $H_{kv}$ is the number of KV heads, $d_h$ is head dimension, $T$ is cached tokens, and $b$ is bytes per element. This value, not free GPU percentage, should drive admission.

## The serving path

1. **Gateway:** authentication, quotas, model and latency class, request validation.
2. **Router:** choose replica or parallel group using queue state, KV capacity, prefix locality, and health.
3. **Admission controller:** reserve worst-case or policy-bounded KV blocks and reject, queue, or degrade before overload becomes universal timeout.
4. **Scheduler:** continuous batching with chunked prefill, decode priority, and fairness across tenants or classes.
5. **Model executor:** paged KV cache, fused kernels, tensor or pipeline parallelism as required.
6. **Streaming response:** backpressure, cancellation, usage accounting, and partial-failure semantics.
7. **Telemetry:** per-request phase timing, queue age, batch composition, KV occupancy, token throughput, requests or tokens that meet their SLO, errors, and quality version.

## Scheduling policy

A defensible baseline is:

- reserve KV capacity at admission using prompt plus bounded output length;
- split long prefills into chunks;
- serve active decode each iteration to protect inter-token latency;
- fill remaining token budget with prefill chunks;
- rotate across tenants within a class;
- cap per-tenant active sequences and reserved KV;
- release pages immediately on EOS, cancellation, or limit;
- reject or route elsewhere before queue age guarantees an SLO miss.

Priority without quotas becomes starvation. Fairness without class-aware SLOs can make interactive traffic wait behind batch jobs. State the policy and the sacrifice.

<p class="visual-kicker">Learning objective</p>
<!-- visual:llm-inference-two-budget-scheduler -->
<figure class="learning-figure editorial-illustration" aria-labelledby="llm-inference-two-budget-scheduler-edition-title"><p class="visual-title" id="llm-inference-two-budget-scheduler-edition-title">Memory stays reserved while iteration work starts fresh</p><svg viewBox="0 0 360 421" role="img" aria-labelledby="llm-inference-two-budget-scheduler-edition-svg-title llm-inference-two-budget-scheduler-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="llm-inference-two-budget-scheduler-edition-svg-title">Memory stays reserved while iteration work starts fresh</title><desc id="llm-inference-two-budget-scheduler-edition-svg-desc">Toy KV memory has twelve pages: A reserves five, B reserves four, and three remain free. A separate eight-slot iteration protects three decode slots and uses five for bounded prefill. When A stops, its five pages are released, leaving eight free. These units are illustrative, not performance measurements.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Toy units: KV pages persist; token slots</tspan><tspan x="180" dy="19">refresh</tspan></text><text x="18" y="65" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">PERSISTENT KV POOL: 12 PAGES</tspan></text><rect x="25" y="84" width="23" height="30" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="36.5" y="104" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="36.5" dy="0">A</tspan></text><rect x="51" y="84" width="23" height="30" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="62.5" y="104" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="62.5" dy="0">A</tspan></text><rect x="77" y="84" width="23" height="30" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="88.5" y="104" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="88.5" dy="0">A</tspan></text><rect x="103" y="84" width="23" height="30" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="114.5" y="104" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="114.5" dy="0">A</tspan></text><rect x="129" y="84" width="23" height="30" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="140.5" y="104" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="140.5" dy="0">A</tspan></text><rect x="155" y="84" width="23" height="30" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="166.5" y="104" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="166.5" dy="0">B</tspan></text><rect x="181" y="84" width="23" height="30" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="192.5" y="104" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="192.5" dy="0">B</tspan></text><rect x="207" y="84" width="23" height="30" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="218.5" y="104" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="218.5" dy="0">B</tspan></text><rect x="233" y="84" width="23" height="30" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="244.5" y="104" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="244.5" dy="0">B</tspan></text><rect x="259" y="84" width="23" height="30" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="270.5" y="104" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270.5" dy="0">-</tspan></text><rect x="285" y="84" width="23" height="30" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="296.5" y="104" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="296.5" dy="0">-</tspan></text><rect x="311" y="84" width="23" height="30" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="322.5" y="104" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="322.5" dy="0">-</tspan></text><text x="180" y="145" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">A: 5 reserved + B: 4 reserved + 3 free</tspan></text><text x="18" y="202" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">ONE ITERATION: 8 TOKEN-WORK SLOTS</tspan></text><rect x="25" y="224" width="35" height="30" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="42.5" y="244" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="42.5" dy="0">D</tspan></text><rect x="64" y="224" width="35" height="30" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="81.5" y="244" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="81.5" dy="0">D</tspan></text><rect x="103" y="224" width="35" height="30" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="120.5" y="244" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="120.5" dy="0">D</tspan></text><rect x="142" y="224" width="35" height="30" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="159.5" y="244" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="159.5" dy="0">P</tspan></text><rect x="181" y="224" width="35" height="30" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="198.5" y="244" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="198.5" dy="0">P</tspan></text><rect x="220" y="224" width="35" height="30" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="237.5" y="244" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="237.5" dy="0">P</tspan></text><rect x="259" y="224" width="35" height="30" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="276.5" y="244" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="276.5" dy="0">P</tspan></text><rect x="298" y="224" width="35" height="30" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="315.5" y="244" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="315.5" dy="0">P</tspan></text><text x="180" y="288" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">3 decode slots first; 5 bounded prefill slots</tspan><tspan x="180" dy="19">remain</tspan></text><text x="180" y="335" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">A stops -&gt; 5 KV pages released -&gt; 8 pages free</tspan></text><text x="180" y="385" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">KV admission and iteration work are different</tspan><tspan x="180" dy="19">budgets.</tspan></text></svg><figcaption><strong>Read it this way:</strong> KV pages persist across iterations; token-work slots refresh each iteration. Stopping A releases its reservation for later admission. All units are illustrative. <span class="visual-reader-check" data-reader-check="llm-inference-two-budget-scheduler"><strong>Check:</strong> Why can free iteration tokens coexist with insufficient KV capacity to admit another request?</span></figcaption></figure>
<p class="diagram-source">Original synthesis informed by the <a href="https://www.usenix.org/conference/osdi24/presentation/agrawal">Sarathi-Serve scheduler</a>, the <a href="https://arxiv.org/abs/2309.06180">PagedAttention paper</a>, and <a href="https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin">DistServe</a>. No source figure or layout is reproduced.</p>

## What an L4 answer sounds like

> "Put the model behind an API, add a queue, batch requests, cache prompts, autoscale GPUs, and monitor latency."

The components are plausible but disconnected. There is no phase model, memory budget, admission rule, or overload decision. Autoscaling cannot rescue a request already trapped behind a 32K-token prefill.

## What an L5 answer adds

An L5 candidate separates prefill and decode, computes KV pressure, uses paged caching and continuous batching, chunks long prefills, and defines time to first token and inter-token latency separately. They describe cancellation, quotas, health checks, and load shedding.

They connect metrics to diagnoses:

| Symptom | Likely evidence |
| --- | --- |
| High time to first token, healthy decode | Prefill queue age, prompt-token backlog, prefill utilization |
| Slow inter-token latency | Decode batch size, memory bandwidth, KV read volume |
| Low utilization with long queues | Scheduler gaps, synchronization, small batches, routing imbalance |
| Admission failures | KV blocks reserved, context distribution, fragmentation |
| One tenant degrades others | Active sequence share, reserved KV share, queue age by tenant |

## What an L6 answer adds

An L6 candidate designs the control plane around uncertainty. Output length is not known at admission, model versions have different KV footprints, prefix caching changes routing value, and speculative decoding changes the relationship between compute and delivered tokens.

They treat co-located versus disaggregated serving as a measured choice. Separate pools help only when phase isolation and independent scaling repay KV-transfer and queueing costs.

They define explicit overload stages:

1. stop admitting low-priority batch traffic;
2. cap output tokens or long contexts for degradable classes;
3. route to a smaller or quantized model when product policy permits;
4. reject early with a retry signal;
5. preserve capacity for health probes and critical tenants.

They also separate product quality from serving health. Quantization, speculative decoding, or a fallback model may change output distribution. A latency win is not free if task success or safety regresses on the traffic slices that trigger degradation.

## Tells that get you a strong-hire vote

- You model prefill and decode separately.
- Admission follows KV capacity and output policy.
- Long prefills are chunked so decode remains responsive.
- Co-located and disaggregated serving are compared with KV-transfer cost included.
- Time to first token and inter-token latency have separate SLOs.
- Tenant fairness and priority are explicit, not implied.
- Overload behavior starts before universal timeout.
- Cancellation releases compute and KV state.
- Performance changes have a quality evaluation path.

## Tells that get you down-leveled

- Starting with a serving framework name.
- One latency percentile for the entire request.
- Batching with no scheduler policy.
- Autoscaling as the answer to every burst.
- No KV-cache calculation.
- Letting the queue absorb overload indefinitely.
- Ignoring cancellation, multi-tenancy, or fallback quality.

## Common follow-up

"A 128K prompt arrives while hundreds of chat requests are decoding. What happens?"

Do not admit it merely because one replica has enough total memory. Route it to a long-context class or queue, reserve its KV budget, chunk prefill, and cap how much prefill work enters each iteration. If its predicted start time already violates the class SLO, reject or offer asynchronous processing. Protect active decode and disclose the product trade-off.

Try the [inference scheduler lab](/prep/labs/inference-scheduler/) before reading implementation details again.

*Related: [Transformer compute and memory accounting](/concepts/transformer-compute-memory-accounting/), [KV cache](/concepts/kv-cache/), and [continuous batching](/concepts/continuous-batching/).*
