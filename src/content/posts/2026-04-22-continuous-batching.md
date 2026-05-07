---
title: "Continuous batching for LLM serving"
description: "Let new requests join an in-flight batch at every decode step instead of waiting for the slowest one. The other half of why vLLM is fast."
date: "2026-04-22"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

Continuous batching (a.k.a. iteration-level scheduling) processes a batch one decode step at a time and lets new requests enter the batch as soon as another request finishes, instead of waiting for the entire static batch to complete.

## Why it matters

LLM decoding is memory-bound: the cost is dominated by reading model weights from HBM, not by per-token compute. So increasing batch size is nearly free in latency for each request. But only if the batch stays full.

With **static batching**, you wait for the longest request in the batch before reusing GPU. If one request generates 1000 tokens and another generates 50, the second request's GPU slot sits idle for 950 steps.

Continuous batching keeps the GPU saturated. Combined with PagedAttention (see [paged attention](/reference/paged-attention/)), it is the foundation of vLLM, TGI, and other modern LLM servers. Throughput improvements over static batching are 2–10× depending on the request mix.

## The mechanism

Each "step" of the server runs one forward pass for the current batch:

1. Maintain a queue of pending requests and a set of active (in-flight) requests.
2. Each step, build a batch of (a) one decode token from each active request whose KV cache exists and (b) prefill tokens for newly admitted requests.
3. Run one forward pass. Update each active request's KV cache.
4. For requests that hit EOS or `max_tokens`, mark complete and free their KV blocks.
5. Admit new requests from the queue if there is enough free KV-cache capacity.

This requires the attention kernel to handle variable per-request lengths in the same batch (cu_seqlens-style cumulative offsets) and a non-contiguous KV cache (PagedAttention).

## Prefill vs. decode

Two phases with very different cost profiles:

- **Prefill**: process the full prompt in one parallel matmul. Compute-bound; high arithmetic intensity.
- **Decode**: one new token per step per request. Memory-bound; benefits hugely from batching.

Most servers either alternate prefill and decode steps or interleave (chunked prefill, [Patel et al., 2023](https://arxiv.org/abs/2308.16369)) so neither phase starves the other.

## Tradeoffs

- **Throughput vs. latency**: larger batches mean higher tokens/sec across the server but slightly higher per-request latency. SLO-aware servers cap batch size or fragment large prefills.
- **Memory pressure**: continuous batching is throughput-limited by KV-cache memory, not by compute. PagedAttention removes most fragmentation; GQA / MQA shrink per-request cache.
- **Fairness**: a long-context request consumes more KV per step. Without admission control, it can starve short requests.

## Common pitfalls

- **Profiling decode without batching.** Single-request decode benchmarks dramatically understate server throughput.
- **Confusing batch size with sequence length.** Batch size grows the number of concurrent *requests*; longer sequences grow per-request KV.
- **Assuming static batching is fine for production.** It almost never is. The GPU sits idle whenever any request finishes.
