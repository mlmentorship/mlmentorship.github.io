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
<figure class="learning-figure editorial-illustration" aria-labelledby="long-context-three-bottlenecks-edition-title"><span class="visual-fragment-anchor" id="long-context-bottlenecks-title" aria-hidden="true"></span><p class="visual-title" id="long-context-three-bottlenecks-edition-title">Paging removes allocation waste, not context growth or attention work</p><svg viewBox="0 0 360 822" role="img" aria-labelledby="long-context-three-bottlenecks-edition-svg-title long-context-three-bottlenecks-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="long-context-three-bottlenecks-edition-svg-title">Fixing cache allocation does not flatten attention work</title><desc id="long-context-three-bottlenecks-edition-svg-desc">Three tracks share relative context-length positions one, two, four, eight. A declared toy trained-position span ends at two, with unvalidated positions beyond it. Exact dense attention work scales one, four, sixteen, sixty-four; logical KV bytes scale one, two, four, eight. FlashAttention does not remove quadratic arithmetic, and paging alone does not reduce logical KV growth. No measured usable threshold is invented.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Relative context n; no measured usability</tspan><tspan x="180" dy="19">threshold</tspan></text><text x="18" y="68" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">1. POSITION QUALITY</tspan></text><text x="18" y="98" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Toy trained span ends at n=2; beyond it is</tspan><tspan x="18" dy="19">unvalidated.</tspan></text><path d="M38 146L78 146" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M78 146L318 146" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><text x="220" y="137" style="font-size:12px;font-weight:400;fill:var(--viz-state-stroke);text-anchor:middle"><tspan x="220" dy="0">needs evaluation</tspan></text><text x="38" y="179" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="38" dy="0">1</tspan></text><text x="78" y="179" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="78" dy="0">2</tspan></text><text x="158" y="179" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="158" dy="0">4</tspan></text><text x="318" y="179" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="318" dy="0">8</tspan></text><text x="18" y="233" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">2. DENSE ATTENTION WORK</tspan></text><path d="M38 352L318 352" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M38 350.78125L78 347.125L158 332.5L318 274" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path><circle cx="38" cy="350.78125" r="3" style="fill:var(--viz-focus-stroke)"></circle><text x="38" y="338.78125" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="38" dy="0">1</tspan></text><text x="38" y="376" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="38" dy="0">1</tspan></text><circle cx="78" cy="347.125" r="3" style="fill:var(--viz-focus-stroke)"></circle><text x="78" y="335.125" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="78" dy="0">4</tspan></text><text x="78" y="376" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="78" dy="0">2</tspan></text><circle cx="158" cy="332.5" r="3" style="fill:var(--viz-focus-stroke)"></circle><text x="158" y="320.5" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="158" dy="0">16</tspan></text><text x="158" y="376" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="158" dy="0">4</tspan></text><circle cx="318" cy="274" r="3" style="fill:var(--viz-focus-stroke)"></circle><text x="318" y="262" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="318" dy="0">64</tspan></text><text x="318" y="376" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="318" dy="0">8</tspan></text><text x="180" y="409" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">FlashAttention removes score materialization,</tspan><tspan x="180" dy="19">not n^2 work.</tspan></text><text x="18" y="462" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">3. LOGICAL KV BYTES</tspan></text><path d="M38 581L318 581" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M38 571.25L78 561.5L158 542L318 503" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2"></path><circle cx="38" cy="571.25" r="3" style="fill:var(--viz-input-stroke)"></circle><text x="38" y="559.25" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="38" dy="0">1</tspan></text><text x="38" y="605" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="38" dy="0">1</tspan></text><circle cx="78" cy="561.5" r="3" style="fill:var(--viz-input-stroke)"></circle><text x="78" y="549.5" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="78" dy="0">2</tspan></text><text x="78" y="605" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="78" dy="0">2</tspan></text><circle cx="158" cy="542" r="3" style="fill:var(--viz-input-stroke)"></circle><text x="158" y="530" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="158" dy="0">4</tspan></text><text x="158" y="605" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="158" dy="0">4</tspan></text><circle cx="318" cy="503" r="3" style="fill:var(--viz-input-stroke)"></circle><text x="318" y="491" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="318" dy="0">8</tspan></text><text x="318" y="605" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="318" dy="0">8</tspan></text><text x="180" y="638" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Paging removes allocation waste, not this linear</tspan><tspan x="180" dy="19">growth.</tspan></text><g data-editorial-node><rect x="20" y="716" width="320" height="82" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="180" y="733.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">USABLE means all three hold:</tspan><tspan x="180" dy="19">validated positions AND executable</tspan><tspan x="180" dy="19">work</tspan><tspan x="180" dy="19">AND affordable serving state</tspan></text></g></svg><figcaption><strong>Read it this way:</strong> Compare horizontal context positions, not vertical units across tracks. Paging does not change logical KV growth. FlashAttention does not change dense quadratic arithmetic. Usable context must satisfy every constraint. <span class="visual-reader-check" data-reader-check="long-context-three-bottlenecks"><strong>Check:</strong> After paging reduces allocation waste, which logical KV growth and attention-work limits still remain?</span></figcaption></figure>

## The three problems

### 1. Position encoding has to extrapolate

A learned position table has a finite index range; other mechanisms may be defined beyond training positions without working reliably there.
For RoPE checkpoints, candidate extension methods include:

- **Position interpolation** [(Chen et al., 2023)](https://arxiv.org/abs/2306.15595): linearly compress positions.
- **NTK-aware scaling**: alter the rotary frequency grid, with implementation-specific changes to different bands.
- **YaRN** [(Peng et al., 2023)](https://arxiv.org/abs/2309.00071): per-frequency interpolation tuned by training-length statistics.

These mapping changes may be paired with continued training.
Evaluate actual long-context behavior; a defined rotation formula is not evidence that the checkpoint uses distant information correctly.

### 2. Attention must fit in memory

Naively materialized dense attention is $O(n^2)$ in memory; at $n = 128{,}000$, one $n \times n$ score matrix alone is about 61 GiB (65.5 GB) at FP32. Solutions:

- **[FlashAttention](/concepts/flashattention/)**: exact, tiled streaming softmax in SRAM. Memory drops to $O(n)$. Standard for both training and serving.
- **[Sparse attention](/concepts/sparse-attention/)** (BigBird, Longformer): mask is sparse. Used for some encoder long-context models.
- **[Linear attention](/concepts/linear-attention/)**: methods change or approximate attention with different quality, arithmetic, and state trade-offs. Do not equate every such method with one low-rank construction.

Dense, sparse, and alternative attention mechanisms have different contracts.
Choose them using a matched workload and task evaluation rather than an asserted universal production recipe.

### 3. KV cache becomes the cost driver

Under a fixed cache layout, retained KV bytes scale linearly with stored positions; the [KV-cache lesson](/concepts/kv-cache/) derives the ledger.
For an illustrative 80-layer model with 8 KV heads, head width 128, two-byte entries, and 131,072 positions, one request needs 40 GiB of logical KV tensors.
That excludes weights, allocator slack, temporary buffers, replication, and effects of sharding or compression.

- **[GQA / MQA](/concepts/grouped-query-attention/)**: share K/V heads. Relative to otherwise matched multi-head attention, the logical KV reduction is the query-head count divided by the KV-head count, not a universal multiplier.
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
