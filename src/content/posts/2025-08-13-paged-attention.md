---
title: "PagedAttention and the vLLM serving model"
description: "Resolve logical KV positions through block tables, calculate shared allocation and tail waste, and preserve ownership during release or branching."
date: "2025-08-13"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

PagedAttention resolves logical KV positions through per-sequence tables pointing to fixed-size physical blocks.
This permits scattered allocation and compatible-prefix sharing without changing the intended logical attention sequence.
It reduces allocation waste relative to some contiguous reservation policies, but does not eliminate tail waste or guarantee higher throughput.

## The mechanism

For token position $p$ and block capacity $b$, lookup is:

$$
\operatorname{address}(p)=(table[\lfloor p/b\rfloor],\;p\bmod b).
$$

Allocate blocks as sequences grow; the kernel gathers K/V in logical order with the same masks and position conventions.
For the pictured tables, use hypothetical $b=4$, ten valid positions per request, and shared positions zero through seven.
A has table `[7,1,5]`; B has `[7,1,3]`.
Position nine has logical block two and offset one: A reads `P5[1]`, B reads `P3[1]`.

Six logical blocks alias four physical blocks, reserving sixteen token slots rather than twenty-four without sharing.
There are twelve distinct valid KV positions: eight shared and two private per request.
Private tails still waste four slots total; sharing reserves sixteen slots, not twelve.

<!-- visual:paged-attention-block-table-indirection -->
<figure class="learning-figure plot-panel" aria-labelledby="paged-attention-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="paged-attention-visual-title">A block table preserves logical order across scattered physical storage</p><div class="visual-argument" data-visual-argument="paged-attention-block-table-indirection" data-argument-kind="ledger"><p class="visual-argument-premise">Logical order survives scattered physical placement</p><svg viewBox="0 0 360 322" role="img" aria-label="Logical order survives scattered physical placement. Logical blocks 0,1,2 map to physical pages 7,2,9. The block table, not physical adjacency, reconstructs the token sequence."><title>Logical order survives scattered physical placement</title><desc>Logical blocks 0,1,2 map to physical pages 7,2,9. The block table, not physical adjacency, reconstructs the token sequence.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Logical sequence block order</tspan></text><rect data-proof-token="0" x="20" y="58" width="102.66666666666667" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="71.33333333333334" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="71.33333333333334" dy="0">0</tspan></text><rect data-proof-token="1" x="126.66666666666667" y="58" width="102.66666666666667" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="178" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">1</tspan></text><rect data-proof-token="2" x="233.33333333333334" y="58" width="102.66666666666667" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="284.6666666666667" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="284.6666666666667" dy="0">2</tspan></text><text x="20" y="126" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Block table: physical pages</tspan></text><rect data-proof-token="7" x="20" y="156" width="102.66666666666667" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="71.33333333333334" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="71.33333333333334" dy="0">7</tspan></text><rect data-proof-token="2" x="126.66666666666667" y="156" width="102.66666666666667" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="178" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">2</tspan></text><rect data-proof-token="9" x="233.33333333333334" y="156" width="102.66666666666667" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="284.6666666666667" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="284.6666666666667" dy="0">9</tspan></text><text x="20" y="224" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Attention follows logical order</tspan></text><rect data-proof-token="page 7" x="20" y="254" width="102.66666666666667" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="71.33333333333334" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="71.33333333333334" dy="0">page 7</tspan></text><rect data-proof-token="page 2" x="126.66666666666667" y="254" width="102.66666666666667" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="178" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">page 2</tspan></text><rect data-proof-token="page 9" x="233.33333333333334" y="254" width="102.66666666666667" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="284.6666666666667" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="284.6666666666667" dy="0">page 9</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
		<text class="viz-label" x="180" y="197" text-anchor="middle">Match each physical ID to the pool below.</text>
		<rect class="viz-node" x="14" y="214" width="336" height="190" rx="8"></rect>
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
	<div class="visual-derivation-caption">logical order stays fixed while physical addresses jump. Both tables reference P7 and P1; each shared block has two owners. P5 and P3 are private tails. The original schematic follows <a href="https://arxiv.org/abs/2309.06180">Kwon et al.</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="paged-attention-block-table-indirection"><strong>Check:</strong> Why can request A read logical blocks in order even though their physical locations are 7, 1 and 5?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="paged-attention-block-table-indirection">Logical blocks 0,1,2 map to physical pages 7,2,9. The block table, not physical adjacency, reconstructs the token sequence.</span></figcaption>
</figure>

## Prefix sharing

After A's outstanding uses finish, release P5 and decrement P7/P1 reference counts from two to one.
B's prefix must remain valid; a shared block is freed only after all live owners and in-flight uses are gone.
Matching visible prompt text is insufficient: tokenization, model/adapters, positions, and relevant cache state must be compatible.

### Append into a Shared Partial Tail

**New task:** Two requests share six positions in four-token blocks, then diverge.

**Picture and change:** Copy the two valid entries of shared logical block one before A appends at offset two.
Redirect A's table entry and decrement the old block's reference count before its private write.

**Check:** B's position six remains absent and B's existing bytes are unchanged.

**Use when:** A shared writable tail branches; directly mutating the shared block would corrupt isolation.

### Use Larger Blocks

**New task:** Change a unique ten-token request from four-token to eight-token blocks.

**Picture and change:** Rebuild table indexing and allocator units together without changing logical positions.

**Check:** Reserved slots rise from twelve to sixteen; tail waste rises from two to six.
Generally, one unshared sequence wastes $\lceil S/b\rceil b-S\le b-1$ slots.

**Use when:** Measured lookup costs justify additional tail waste; fewer table entries alone do not establish faster execution.

<!-- learning:core-end -->
<div data-learning-extension>

## Continuous batching pairs naturally with paging

Paging can support [continuous batching](/concepts/continuous-batching/) through per-request cache addressing.
Admission and scheduling remain separate policies constrained by available blocks, compute, and latency targets.

## Tradeoffs

- **Kernel/layout cost:** indirection, bandwidth, locality, and concurrency interact; measure actual workloads.
- **Block size:** smaller blocks reduce bounded tail waste but increase metadata and potentially lookup work.
- **Policy state:** allocation, eviction, caching, and copy-on-write must preserve ownership and identity.

## Common pitfalls

- **Different mechanisms:** paging addresses cache layout/ownership; FlashAttention addresses attention's memory-access computation. Implementations can combine them when supported.
- **Incomplete measurement:** report throughput, memory, and tail latency under stated load rather than a universal serving-speed claim.

</div>
