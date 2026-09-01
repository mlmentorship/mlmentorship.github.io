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

<!-- visual:paged-attention-block-table-indirection -->
<figure class="learning-figure plot-panel" aria-labelledby="paged-attention-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="paged-attention-visual-title">Trace two logical KV caches through their block tables to scattered and shared physical blocks.</p>
	<svg viewBox="0 0 360 430" role="img" aria-labelledby="paged-attention-svg-title paged-attention-svg-desc">
		<title id="paged-attention-svg-title">Two sequence block tables share prefix blocks in a non-contiguous physical pool</title>
		<desc id="paged-attention-svg-desc">Request A maps logical blocks zero, one, and two to physical blocks seven, one, and five. Request B maps the same logical prefix blocks zero and one to physical blocks seven and one, then maps its private logical block two to physical block three. Physical blocks seven and one each have reference count two. The remaining private blocks have reference count one, and unused physical blocks are free.</desc>
		<defs>
			<marker id="arrow-forward" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker>
		</defs>
		<text class="viz-axis-label" x="12" y="20">1 · EACH REQUEST KEEPS LOGICAL TOKEN ORDER</text>
		<text class="viz-label" x="12" y="64">request A</text>
		<rect class="viz-node viz-node--focus" x="86" y="38" width="80" height="42" rx="5"></rect>
		<rect class="viz-node viz-node--focus" x="174" y="38" width="80" height="42" rx="5"></rect>
		<rect class="viz-node viz-node--input" x="262" y="38" width="80" height="42" rx="5"></rect>
		<text class="viz-node-label" x="126" y="56" text-anchor="middle">logical 0</text>
		<text class="viz-node-value" x="126" y="71" text-anchor="middle">→ physical 7</text>
		<text class="viz-node-label" x="214" y="56" text-anchor="middle">logical 1</text>
		<text class="viz-node-value" x="214" y="71" text-anchor="middle">→ physical 1</text>
		<text class="viz-node-label" x="302" y="56" text-anchor="middle">logical 2</text>
		<text class="viz-node-value" x="302" y="71" text-anchor="middle">→ physical 5</text>
		<text class="viz-label" x="12" y="124">request B</text>
		<rect class="viz-node viz-node--focus" x="86" y="98" width="80" height="42" rx="5"></rect>
		<rect class="viz-node viz-node--focus" x="174" y="98" width="80" height="42" rx="5"></rect>
		<rect class="viz-node viz-node--output" x="262" y="98" width="80" height="42" rx="5"></rect>
		<text class="viz-node-label" x="126" y="116" text-anchor="middle">logical 0</text>
		<text class="viz-node-value" x="126" y="131" text-anchor="middle">→ physical 7</text>
		<text class="viz-node-label" x="214" y="116" text-anchor="middle">logical 1</text>
		<text class="viz-node-value" x="214" y="131" text-anchor="middle">→ physical 1</text>
		<text class="viz-node-label" x="302" y="116" text-anchor="middle">logical 2</text>
		<text class="viz-node-value" x="302" y="131" text-anchor="middle">→ physical 3</text>
		<path class="viz-gridline" d="M86 88 H254"></path>
		<text class="viz-callout" x="170" y="93" text-anchor="middle">same prefix mappings</text>
		<text class="viz-axis-label" x="12" y="180">2 · THE TABLE RESOLVES EACH LOOKUP IN HBM</text>
		<rect class="viz-node" x="14" y="214" width="336" height="190" rx="8"></rect>
		<path class="viz-forward" d="M126 82 C126 170 278 190 278 324"></path>
		<path class="viz-forward" d="M126 142 C126 200 278 220 278 324"></path>
		<path class="viz-forward" d="M214 82 C214 170 134 190 134 244"></path>
		<path class="viz-forward" d="M214 142 C214 200 134 210 134 244"></path>
		<path class="viz-forward" d="M302 82 C350 170 134 280 134 324"></path>
		<path class="viz-forward" d="M302 142 C302 190 278 205 278 244"></path>
		<text class="viz-node-label" x="180" y="235" text-anchor="middle">physical KV blocks · scattered</text>
		<rect x="30" y="250" width="64" height="58" rx="4" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:4 3"></rect>
		<rect class="viz-node viz-node--focus" x="102" y="250" width="64" height="58" rx="4"></rect>
		<rect x="174" y="250" width="64" height="58" rx="4" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:4 3"></rect>
		<rect class="viz-node viz-node--output" x="246" y="250" width="64" height="58" rx="4"></rect>
		<text class="viz-label" x="62" y="274" text-anchor="middle">P0</text>
		<text class="viz-label" x="62" y="292" text-anchor="middle">free</text>
		<text class="viz-node-label" x="134" y="272" text-anchor="middle">P1</text>
		<text class="viz-node-value" x="134" y="288" text-anchor="middle">prefix</text>
		<text class="viz-node-value" x="134" y="301" text-anchor="middle">refs = 2</text>
		<text class="viz-label" x="206" y="274" text-anchor="middle">P2</text>
		<text class="viz-label" x="206" y="292" text-anchor="middle">free</text>
		<text class="viz-node-label" x="278" y="272" text-anchor="middle">P3</text>
		<text class="viz-node-value" x="278" y="288" text-anchor="middle">B private</text>
		<text class="viz-node-value" x="278" y="301" text-anchor="middle">refs = 1</text>
		<rect x="30" y="330" width="64" height="58" rx="4" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:4 3"></rect>
		<rect class="viz-node viz-node--input" x="102" y="330" width="64" height="58" rx="4"></rect>
		<rect x="174" y="330" width="64" height="58" rx="4" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:4 3"></rect>
		<rect class="viz-node viz-node--focus" x="246" y="330" width="64" height="58" rx="4"></rect>
		<text class="viz-label" x="62" y="354" text-anchor="middle">P4</text>
		<text class="viz-label" x="62" y="372" text-anchor="middle">free</text>
		<text class="viz-node-label" x="134" y="352" text-anchor="middle">P5</text>
		<text class="viz-node-value" x="134" y="368" text-anchor="middle">A private</text>
		<text class="viz-node-value" x="134" y="381" text-anchor="middle">refs = 1</text>
		<text class="viz-label" x="206" y="354" text-anchor="middle">P6</text>
		<text class="viz-label" x="206" y="372" text-anchor="middle">free</text>
		<text class="viz-node-label" x="278" y="352" text-anchor="middle">P7</text>
		<text class="viz-node-value" x="278" y="368" text-anchor="middle">prefix</text>
		<text class="viz-node-value" x="278" y="381" text-anchor="middle">refs = 2</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> scan each request left to right: its logical blocks stay in token order, but the block-table entries jump to physical blocks 7, 1, then 5 or 3. Both requests point to P7 and P1 for the common prefix, so those blocks have two references and exist only once. Shape, labels, line style, and reference counts carry the meaning; color is redundant. Original schematic checked against <a href="https://arxiv.org/abs/2309.06180">Kwon et al.’s PagedAttention paper</a>.</figcaption>
</figure>

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
