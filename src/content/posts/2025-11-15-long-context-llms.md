---
title: "Long-context LLMs: training and serving techniques"
description: "What makes a 1M-token context model work. Position-encoding extension, attention kernels, KV-cache management, and the tradeoffs."
date: "2025-11-15"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

Long-context LLMs combine **position-encoding extension** (so the model generalizes past its training length), **I/O-aware attention kernels** (so attention fits in memory), and **KV-cache management** (so serving stays affordable at long inputs).

## Why it matters

Frontier models in 2026 advertise 128K–2M-token context windows. The headline number hides three independent engineering problems, each with its own state of the art. Knowing which technique addresses which problem is the senior-level test.

## The three problems

### 1. Position encoding has to extrapolate

Trained absolute or learned positions don't work past the training length. Modern decoder LLMs use [RoPE](/concepts/rotary-position-embeddings/) and extend it via:

- **Position interpolation** [(Chen et al., 2023)](https://arxiv.org/abs/2306.15595): linearly compress positions.
- **NTK-aware scaling**: increase the RoPE frequency base so high-frequency components don't alias.
- **YaRN** [(Peng et al., 2023)](https://arxiv.org/abs/2309.00071): per-frequency interpolation tuned by training-length statistics.

Most production long-context models use YaRN or NTK scaling, often combined with a brief continued-pretraining stage on long documents.

### 2. Attention must fit in memory

Dense attention is $O(n^2)$ in memory; at $n = 128{,}000$ the $n \times n$ matrix alone is 16 GB at FP32. Solutions:

- **[FlashAttention](/concepts/flashattention/)**: exact, tiled streaming softmax in SRAM. Memory drops to $O(n)$. Standard for both training and serving.
- **[Sparse attention](/concepts/sparse-attention/)** (BigBird, Longformer): mask is sparse. Used for some encoder long-context models.
- **[Linear attention](/concepts/linear-attention/)** (Performer, Linformer): low-rank approximation. Used in research and a few production niches; quality lags dense at chat-model scale.

Production decoder LLMs at long context use dense FlashAttention plus aggressive KV-cache compression rather than sparse / linear approximations.

### 3. KV cache becomes the cost driver

KV cache size scales linearly with context (see [KV cache](/concepts/kv-cache/)). A 70B model at 128K context: ~40 GB of KV per request. Solutions:

- **[GQA / MQA](/concepts/grouped-query-attention/)**: share K/V heads. 4–8× cache reduction.
- **[PagedAttention](/concepts/paged-attention/)**: eliminate cache fragmentation, share prefixes across requests.
- **[Quantization](/concepts/quantization/)**: int8 or int4 KV cache. 2–4× cache reduction.
- **Sliding-window attention** (Mistral): keep only the last $w$ KV positions per layer; lose strict global attention.
- **KV cache eviction**: heuristics like H2O [(Zhang et al., 2023)](https://arxiv.org/abs/2306.14048) keep only "heavy hitter" tokens.

## A typical 2026 long-context production stack

- Llama-class architecture with **GQA-8** and **RoPE** (YaRN-extended for long context).
- Training: **BF16 mixed precision**, **FlashAttention-2** kernels, **sequence packing**, **FSDP** sharding.
- Serving: **vLLM** with **PagedAttention** + **continuous batching**, **int8 weights**, optional **int8 KV cache**.

## Common pitfalls

- **Quoting context length without measuring quality.** A model can run at 128K but degrade rapidly past 32K. Use needle-in-a-haystack and long-doc QA evals.
- **Confusing training length with usable context.** Models often degrade on inputs longer than the longest examples seen during training (or RoPE extension).
- **Ignoring serving cost.** A 1M context window is feasible to compute but may cost $10+ per request at frontier prices.
