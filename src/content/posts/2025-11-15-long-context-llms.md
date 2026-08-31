---
title: "Long-context LLMs: training and serving techniques"
description: "What makes a 1M-token context model work. Position-encoding extension, attention kernels, KV-cache management, and the tradeoffs."
date: "2025-11-15"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Long-context LLMs combine **position-encoding extension** (so the model generalizes past its training length), **I/O-aware attention kernels** (so attention fits in memory), and **KV-cache management** (so serving stays affordable at long inputs).

Frontier models in 2026 advertise 128K–2M-token context windows. The headline number hides three independent engineering problems, each with its own state of the art. Knowing which technique addresses which problem is the senior-level test.

<!-- visual:long-context-three-bottlenecks -->
<figure class="learning-figure" aria-labelledby="long-context-bottlenecks-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="long-context-bottlenecks-title">A longer context window is three different scaling problems.</p>
	<table class="cm-grid" aria-label="Three independent long-context bottlenecks and the techniques that address each one">
		<thead>
			<tr><th scope="col">Bottleneck</th><th scope="col">What breaks as n grows?</th><th scope="col">Matching lever</th></tr>
		</thead>
		<tbody>
			<tr><th scope="row">Position meaning</th><td>RoPE phases leave the trained range</td><td class="cm-selected"><strong>PI / YaRN</strong>Rescale positions + train on long examples</td></tr>
			<tr><th scope="row">Attention execution</th><td>Dense work is n²; naive score storage is n²</td><td class="cm-selected"><strong>FlashAttention</strong>Exact tiling removes n² materialization, not n² math</td></tr>
			<tr><th scope="row">Serving state</th><td>Retained K/V bytes grow with n per live request</td><td class="cm-selected"><strong>GQA · quantize · page</strong>Reduce heads/bytes/waste; paging alone does not shrink logical K/V</td></tr>
		</tbody>
	</table>
	<p class="cm-equation">usable context = positional quality ∩ executable attention ∩ affordable serving state</p>
	<figcaption><strong>Read it this way:</strong> read across one row, not down one column. Position scaling can preserve meaning at unseen distances but does not reduce attention work or cache bytes. FlashAttention avoids writing the full score matrix while keeping exact dense attention and its quadratic arithmetic. GQA and quantization shrink the logical KV cache, while paging mainly removes allocation waste. A production context length is usable only where all three constraints hold. Original comparison checked against <a href="https://arxiv.org/abs/2306.15595">Chen et al. on position interpolation</a>, <a href="https://arxiv.org/abs/2205.14135">Dao et al. on FlashAttention</a>, and <a href="https://arxiv.org/abs/2309.06180">Kwon et al. on PagedAttention</a>.</figcaption>
</figure>

## The three problems

### 1. Position encoding has to extrapolate

Trained absolute or learned positions don't work past the training length. Modern decoder LLMs use [RoPE](/concepts/rotary-position-embeddings/) and extend it via:

- **Position interpolation** [(Chen et al., 2023)](https://arxiv.org/abs/2306.15595): linearly compress positions.
- **NTK-aware scaling**: increase the RoPE frequency base so high-frequency components don't alias.
- **YaRN** [(Peng et al., 2023)](https://arxiv.org/abs/2309.00071): per-frequency interpolation tuned by training-length statistics.

Most production long-context models use YaRN or NTK scaling, often combined with a brief continued-pretraining stage on long documents.

### 2. Attention must fit in memory

Naively materialized dense attention is $O(n^2)$ in memory; at $n = 128{,}000$, one $n \times n$ score matrix alone is about 61 GiB (65.5 GB) at FP32. Solutions:

- **[FlashAttention](/concepts/flashattention/)**: exact, tiled streaming softmax in SRAM. Memory drops to $O(n)$. Standard for both training and serving.
- **[Sparse attention](/concepts/sparse-attention/)** (BigBird, Longformer): mask is sparse. Used for some encoder long-context models.
- **[Linear attention](/concepts/linear-attention/)** (Performer, Linformer): low-rank approximation. Used in research and a few production niches; quality lags dense at chat-model scale.

Production decoder LLMs at long context use dense FlashAttention plus aggressive KV-cache compression rather than sparse / linear approximations.

### 3. KV cache becomes the cost driver

KV cache size scales linearly with context (see [KV cache](/concepts/kv-cache/)). A 70B model at 128K context: ~40 GB of KV per request. Solutions:

- **[GQA / MQA](/concepts/grouped-query-attention/)**: share K/V heads. 4–8× cache reduction.
- **[PagedAttention](/concepts/paged-attention/)**: eliminate cache fragmentation and enable block reuse, including shared prefixes; it does not reduce the logical K/V bytes for a unique sequence.
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

## Related

- [Context parallelism and ring attention](/concepts/context-parallelism-and-ring-attention/). Split exact long-context attention across devices.
- [FlashAttention](/concepts/flashattention/). Compute exact attention without writing the full score matrix to HBM.
- [KV cache](/concepts/kv-cache/). Account for long-context serving memory.
