---
title: "Prefill vs. decode: the two phases of LLM inference"
description: "LLM inference has two cost regimes with very different bottlenecks. Mixing them up leads to wrong cost models and bad serving decisions."
date: "2025-12-14"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

LLM inference splits into **prefill** (one parallel pass over the prompt) and **decode** (one new token per autoregressive step). Large, well-batched prefills are often compute-bound. Small decode batches are often memory-bound. Almost every serving optimization makes sense only when you know which phase and shape it targets.

Prefill can process many tokens at once with high arithmetic intensity. A single-request decode step processes one token but may read the model weights from HBM for very little arithmetic. Their per-token costs and latency limits can differ greatly.

Picking the wrong cost model leads to wrong decisions: batching helps decode but barely affects prefill latency; quantization helps decode bandwidth but not prefill compute; speculative decoding only accelerates the decode phase.

## Prefill

For a prompt of length $P$:

- Single forward pass: compute K, V, and output for all $P$ tokens in parallel.
- FLOPs: $\approx 2 \cdot P \cdot N_\text{params}$ for the FFN and Q/K/V/O matmuls, plus $O(P^2 \cdot d \cdot \text{layers})$ for attention.
- Arithmetic intensity is high because Q has $P$ rows; matmuls are square-shaped and saturate tensor cores.
- Time to first token grows with prompt length. Matrix efficiency, attention's $O(P^2)$ work, batching, and scheduler load determine the exact curve.

Likely bottleneck: compute for large efficient matrix shapes. Short prompts, small batches, or poor kernels can be memory-bound or launch-bound.

## Decode

For each subsequent generated token:

- One forward pass with sequence length 1 (just the new token).
- Q is a single vector; K and V come from the [KV cache](/concepts/kv-cache/).
- FLOPs: $\approx 2 \cdot N_\text{params}$. One multiply-add per parameter for the matmul.
- Bytes moved from HBM: at least $N_\text{params} \cdot \text{dtype\_bytes}$ (must read all weights).
- Arithmetic intensity from BF16 weight reads: about 1 FLOP/byte at batch 1 because one multiply-add uses a two-byte weight. Other reads and writes lower the effective value.

Likely bottleneck: HBM bandwidth at small batch sizes. Batching increases arithmetic intensity until compute or KV-cache traffic becomes limiting.

## What follows from the asymmetry

| Optimization | Helps prefill? | Helps decode? | Why |
|--------------|---------------|---------------|-----|
| Larger batch | workload-dependent | large until another limit | amortizes weight reads across requests |
| FlashAttention | yes (long prompts) | yes (long context) | reduces HBM traffic in attention |
| Weight quantization (int8/4) | small | huge | cuts decode bandwidth proportionally |
| KV-cache quantization | no | yes (long context) | cuts decode-time KV reads |
| Speculative decoding | no | huge | parallelizes decode steps |
| GQA / MQA | small | yes (long context) | shrinks KV cache |
| Continuous batching | small | huge | keeps batch full during decode |

## Latency metrics

Production serving SLOs typically use both:

- **TTFT** (Time To First Token): prefill time. Bottleneck for chat UI responsiveness.
- **TPOT** (Time Per Output Token, or inter-token latency): decode time per token. Bottleneck for sustained generation.
- **End-to-end latency** = TTFT + (output_length − 1) × TPOT.

For a 1000-token output, TPOT dominates. For a search query that gets a 50-token answer, TTFT dominates.

## Common pitfalls

- **Quoting one cost number for "inference."** Prefill and decode are different problems with different solutions.
- **Optimizing decode without measuring TTFT.** Speculative decoding can hurt latency on short outputs (overhead dominates).
- **Ignoring chunked prefill.** Long prefills block decode steps for other requests in the same batch; chunked prefill [(Patel et al., 2023)](https://arxiv.org/abs/2308.16369) interleaves them.

## Related

- [GPU memory hierarchy](/concepts/gpu-memory-hierarchy/). Why decode is bandwidth-bound.
- [Continuous batching](/concepts/continuous-batching/). How servers exploit decode batching.
- [Speculative decoding](/concepts/speculative-decoding/). The main lever for decode speedup.
