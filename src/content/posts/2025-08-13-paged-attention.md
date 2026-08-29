---
title: "PagedAttention and the vLLM serving model"
description: "Treat the KV cache like virtual memory: allocate in fixed-size pages, share pages across sequences, eliminate fragmentation. The reason vLLM is the default LLM server."
date: "2025-08-13"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

PagedAttention stores the KV cache in fixed-size physical blocks indexed by a per-sequence block table, so the cache is no longer a contiguous tensor and can be allocated, freed, and shared at block granularity. Like virtual memory pages in an OS.

Naive KV-cache management pre-allocates `max_seq_len` of contiguous memory for every request in the batch. Most requests are short, so most of that memory is wasted (internal fragmentation). Shared prefixes (system prompts, few-shot examples) are duplicated across requests (no sharing). The result: serving throughput is bottlenecked by KV-cache memory, not by compute.

PagedAttention ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180), the core idea behind vLLM) solves both: physical fragmentation drops near zero, and shared prefixes use the same physical blocks. vLLM has become the default open-source LLM serving runtime since 2023.

## The mechanism

1. **Block size**: pick a small fixed number of tokens per block, e.g. $B = 16$.
2. **Physical block pool**: allocate a large pool of $B$-sized KV blocks in HBM at startup.
3. **Per-sequence block table**: each in-flight request has a logical-to-physical block table mapping its position $p$ to a physical block index.
4. **Attention kernel**: a custom CUDA kernel reads K/V through the block table, gathering blocks as needed for the attention computation.
5. **Allocation**: when a sequence grows, allocate a new physical block on demand; when it finishes, free its blocks back to the pool.

Internal fragmentation is at most $B - 1$ tokens per request (vs. potentially thousands without paging).

## Prefix sharing

Two requests with the same system prompt can share the physical blocks for that prefix:

- The first request fills physical blocks for the prefix.
- The second request's block table points to the same physical blocks for matching positions.
- Reference counts track when a shared block can be freed.

For systems with long shared prefixes (chat with a long system prompt; structured generation; agent loops with tool descriptions), prefix sharing gives a multiplicative throughput boost. Fewer KV-cache writes, less memory used per request, more batch concurrency.

## Continuous batching pairs naturally with paging

PagedAttention enables **continuous batching**: requests of different lengths can be batched together because the kernel reads each sequence's KV through its own block table. New requests can join an in-flight batch at any step (instead of waiting for the slowest request in a static batch to finish). See [continuous batching](/concepts/continuous-batching/).

## Tradeoffs

- **Kernel overhead**: gathering K/V through a block table is slightly slower per FLOP than reading contiguous memory; the throughput gain from higher batch concurrency dominates in practice.
- **Block size**: smaller $B$ means less fragmentation but more block-table lookups; $B = 16$ is the standard.
- **Implementation complexity**: paging logic, block tables, copy-on-write for shared blocks. Worth it for any production deployment.

## Common pitfalls

- **Confusing PagedAttention with FlashAttention.** PagedAttention is about memory layout for the KV cache; FlashAttention is about the attention kernel itself. Both can (and should) coexist.
- **Treating throughput as the only metric.** Paging trades a small per-token compute overhead for much higher batch sizes; check tail latency under load.
