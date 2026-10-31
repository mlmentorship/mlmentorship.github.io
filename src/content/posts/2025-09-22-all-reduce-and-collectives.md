---
title: "All-reduce and other collectives"
description: "Choose a collective from each rank's required output, then account for reduction, shard ownership, traffic, latency and overlap."
date: "2025-09-22"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["allreduce", "allgather", "reduce scatter", "collective communication"]
---

## Summary

Choose a collective by each rank's required output: concatenate pieces, combine aligned values, or distribute a completed result.

## Problem

The process group contains ranks 0 and 1, holding `[1, 2]` and `[3, 4]`.
Both buffers have shape `(2,)`, dtype int64 and the same index meanings.
Specify what each rank holds after a sum all-reduce, reduce-scatter and all-gather.
Use rank order for concatenation and equal, index-aligned chunks for scattering.

## Picture: Add Aligned Positions or Join Pieces

Sum reduction combines corresponding positions: `[1+3, 2+4]` becomes `[4, 6]`.
All-reduce gives that result to both ranks.
Reduce-scatter keeps `[4]` on rank 0 and `[6]` on rank 1.
All-gather of those reduced shards restores `[4, 6]` on both.

All-gather of the original buffers instead produces `[1, 2, 3, 4]`.
It concatenates; it does not add.
The all-reduce identity therefore needs compatible reduction, chunk boundaries and rank order.

<span id="the-five-primitives"></span>

## Six Output Contracts

<span id="broadcast"></span>
<span id="reduce"></span>
<span id="all-reduce"></span>
<span id="all-gather"></span>
<span id="reduce-scatter"></span>

| Operation on the stated inputs | Rank 0 afterward | Rank 1 afterward |
| --- | --- | --- |
| Broadcast rank 0 | `[1, 2]` | `[1, 2]` |
| Sum reduce to rank 0 | `[4, 6]` | No required result |
| Sum all-reduce | `[4, 6]` | `[4, 6]` |
| All-gather original buffers | `[1, 2, 3, 4]` | `[1, 2, 3, 4]` |
| Sum reduce-scatter | `[4]` | `[6]` |
| All-to-all, one value per destination | `[1, 3]` | `[2, 4]` |

Sum does not divide by rank count.
Equal-sized data-parallel batches can average by dividing the summed gradients appropriately.
Unequal local batches require weighting by example counts, not blindly averaging local means.

## Reference the Values Before the Network

This reference models values and order, not transport, dtype enforcement or synchronization.

```python
def sum_by_index(buffers):
	if not buffers:
		raise ValueError("the process group must be nonempty")
	width = len(buffers[0])
	if any(len(buffer) != width for buffer in buffers):
		raise ValueError("rank inputs must have matching shapes")
	return [sum(contributions) for contributions in zip(*buffers)]


def reduce_scatter_reference(buffers):
	reduced = sum_by_index(buffers)
	ranks = len(buffers)
	if len(reduced) % ranks:
		raise ValueError("equal chunks require a divisible width")
	width = len(reduced) // ranks
	return [reduced[rank * width:(rank + 1) * width] for rank in range(ranks)]


def all_gather_reference(shards):
	if not shards or any(len(shard) != len(shards[0]) for shard in shards):
		raise ValueError("this reference requires equal-sized rank shards")
	return [value for shard in shards for value in shard]


buffers = [[1, 2], [3, 4]]
assert sum_by_index(buffers) == [4, 6]
assert reduce_scatter_reference(buffers) == [[4], [6]]
assert all_gather_reference(reduce_scatter_reference(buffers)) == [4, 6]
assert all_gather_reference(buffers) == [1, 2, 3, 4]
```

## The Ring All-Reduce

Now use four ranks and full input size $B=8$ MB, with decimal MB.
Split each input into four equal, aligned chunks.
For chunk A, rank-local contributions are $a_0,a_1,a_2,a_3$.
Three receive-and-add steps finish $A^*$; three forwarding steps distribute it without further addition.
Other chunks follow the corresponding schedule concurrently.

Each rank sends $B/4=2$ MB per step: six steps send 12 MB.
It also receives 12 MB; do not count sent-plus-received bytes as sent traffic.
Generally, ring sent traffic is $2B(N-1)/N$ per rank over $2(N-1)$ steps.

<!-- visual:ring-all-reduce-chunk-trace -->
<figure class="learning-figure editorial-illustration" data-illustration-print-page aria-labelledby="ring-all-reduce-chunk-trace-edition-title"><span class="visual-fragment-anchor" id="ring-trace-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="ring-trace-svg-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="ring-trace-svg-desc" aria-hidden="true"></span><p class="visual-title" id="ring-all-reduce-chunk-trace-edition-title">Reduce each chunk on the ring, then circulate its completed sum</p><div class="visual-argument" data-visual-argument="ring-all-reduce-chunk-trace" data-argument-kind="ledger" data-argument-composition="transform"><svg viewBox="0 0 360 394" role="img" aria-label="Reduce a chunk, then circulate its completed sum. For one illustrative chunk, rank values 1,2,3,4 reduce to 10. All-gather distributes that completed chunk; it does not add the sum again."><title>Reduce a chunk, then circulate its completed sum</title><desc>For one illustrative chunk, rank values 1,2,3,4 reduce to 10. All-gather distributes that completed chunk; it does not add the sum again.</desc><circle cx="23" cy="24" r="10" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="23" y="28" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="23" dy="0">1</tspan></text><text x="44" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="44" dy="0">Local values of one chunk</tspan></text><g data-proof-token="1" data-proof-row="0" data-proof-position="0" data-proof-box="44,60,68,48"><rect x="44" y="60" width="68" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="78" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="78" dy="0">1</tspan></text></g><g data-proof-token="2" data-proof-row="0" data-proof-position="1" data-proof-box="118,60,68,48"><rect x="118" y="60" width="68" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="152" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="152" dy="0">2</tspan></text></g><g data-proof-token="3" data-proof-row="0" data-proof-position="2" data-proof-box="192,60,68,48"><rect x="192" y="60" width="68" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="226" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="226" dy="0">3</tspan></text></g><g data-proof-token="4" data-proof-row="0" data-proof-position="3" data-proof-box="266,60,68,48"><rect x="266" y="60" width="68" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="300" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="300" dy="0">4</tspan></text></g><path d="M23 41L23 118" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M19 112L23 118" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M27 112L23 118" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="23" cy="140" r="10" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.75"></circle><text x="23" y="144" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="23" dy="0">2</tspan></text><text x="44" y="144" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="44" dy="0">Reduction partials along the ring</tspan></text><g data-proof-token="1" data-proof-row="1" data-proof-position="0" data-proof-box="44,176,68,48"><rect x="44" y="176" width="68" height="48" rx="0" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="78" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="78" dy="0">1</tspan></text></g><g data-proof-token="3" data-proof-row="1" data-proof-position="1" data-proof-box="118,176,68,48"><rect x="118" y="176" width="68" height="48" rx="0" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="152" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="152" dy="0">3</tspan></text></g><g data-proof-token="6" data-proof-row="1" data-proof-position="2" data-proof-box="192,176,68,48"><rect x="192" y="176" width="68" height="48" rx="0" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="226" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="226" dy="0">6</tspan></text></g><g data-proof-token="10" data-proof-row="1" data-proof-position="3" data-proof-box="266,176,68,48"><rect x="266" y="176" width="68" height="48" rx="0" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="300" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="300" dy="0">10</tspan></text></g><path d="M23 157L23 234" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M19 228L23 234" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M27 228L23 234" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="23" cy="256" r="10" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="23" y="260" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="23" dy="0">3</tspan></text><text x="44" y="260" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="44" dy="0">Completed chunk distributed to</tspan><tspan x="44" dy="18">ranks</tspan></text><g data-proof-token="10" data-proof-row="2" data-proof-position="0" data-proof-box="44,310,68,48"><rect x="44" y="310" width="68" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="78" y="339.5" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="78" dy="0">10</tspan></text></g><g data-proof-token="10" data-proof-row="2" data-proof-position="1" data-proof-box="118,310,68,48"><rect x="118" y="310" width="68" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="152" y="339.5" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="152" dy="0">10</tspan></text></g><g data-proof-token="10" data-proof-row="2" data-proof-position="2" data-proof-box="192,310,68,48"><rect x="192" y="310" width="68" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="226" y="339.5" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="226" dy="0">10</tspan></text></g><g data-proof-token="10" data-proof-row="2" data-proof-position="3" data-proof-box="266,310,68,48"><rect x="266" y="310" width="68" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="300" y="339.5" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="300" dy="0">10</tspan></text></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 924" role="img" aria-labelledby="ring-all-reduce-chunk-trace-edition-svg-title ring-all-reduce-chunk-trace-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="ring-all-reduce-chunk-trace-edition-svg-title">All four chunks move concurrently; reduction adds, while all-gather only copies</title><desc id="ring-all-reduce-chunk-trace-edition-svg-desc">Four ranks form a directed ring. At one concurrent send step ranks zero, one, two, and three send chunks A, B, C, and D respectively to the next rank. Every message has size B over four. A separate trace follows one coordinate of chunk A with local values one, two, three, and four: three receives and additions produce three, six, then ten at rank three. Three all-gather hops copy ten unchanged to ranks zero, one, and two. Six equal sent-chunk cells give three B over two sent bytes per rank, not sent-plus-received traffic.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Four ranks; four equal chunks A, B, C, D</tspan></text><text x="180" y="50" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">ONE CONCURRENT SEND STEP</tspan></text><circle cx="90" cy="110" r="23" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="90" y="115" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">r0</tspan></text><circle cx="270" cy="110" r="23" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="270" y="115" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">r1</tspan></text><circle cx="270" cy="270" r="23" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="270" y="275" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">r2</tspan></text><circle cx="90" cy="270" r="23" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="90" y="275" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">r3</tspan></text><g data-ring-chunk="A" data-sender="0" data-receiver="1" data-chunk-bytes="B/4"><path d="M114 110L246 110" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M239.8569220667674,113.35597877022943L246,110L239.8569220667674,106.64402122977057" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><text x="180" y="87" style="font-size:12px;font-weight:700;fill:var(--viz-focus-stroke);text-anchor:middle"><tspan x="180" dy="0">A: B/4</tspan></text></g><g data-ring-chunk="B" data-sender="1" data-receiver="2" data-chunk-bytes="B/4"><path d="M270 134L270 246" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M266.6440212297706,239.8569220667674L270,246L273.3559787702294,239.8569220667674" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="323" y="184" style="font-size:12px;font-weight:700;fill:var(--viz-input-stroke);text-anchor:middle"><tspan x="323" dy="0">B: B/4</tspan></text></g><g data-ring-chunk="C" data-sender="2" data-receiver="3" data-chunk-bytes="B/4"><path d="M246 270L114 270" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75"></path><path d="M120.1430779332326,266.6440212297706L114,270L120.1430779332326,273.3559787702294" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75"></path><text x="180" y="309" style="font-size:12px;font-weight:700;fill:var(--viz-state-stroke);text-anchor:middle"><tspan x="180" dy="0">C: B/4</tspan></text></g><g data-ring-chunk="D" data-sender="3" data-receiver="0" data-chunk-bytes="B/4"><path d="M90 246L90 134" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M93.35597877022943,140.1430779332326L90,134L86.64402122977057,140.1430779332326" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><text x="37" y="184" style="font-size:12px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="37" dy="0">D: B/4</tspan></text></g><text x="180" y="357" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">FOLLOW ONE COORDINATE IN CHUNK A</tspan></text><text x="180" y="395" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Local contributions at r0..r3: 1, 2, 3, 4</tspan></text><text x="180" y="428" style="font-size:12px;font-weight:700;fill:var(--viz-focus-stroke);text-anchor:middle"><tspan x="180" dy="0">REDUCE-SCATTER: receive, then add</tspan></text><g data-ring-phase="reduce-scatter" data-step="1" data-incoming="1" data-local="2" data-result="3"><text x="20" y="466" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">r0 -&gt; r1</tspan></text><text x="235" y="466" style="font-size:17px;font-weight:700;fill:var(--viz-focus-stroke);text-anchor:middle"><tspan x="235" dy="0">1 + 2 = 3</tspan></text></g><g data-ring-phase="reduce-scatter" data-step="2" data-incoming="3" data-local="3" data-result="6"><text x="20" y="518" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">r1 -&gt; r2</tspan></text><text x="235" y="518" style="font-size:17px;font-weight:700;fill:var(--viz-focus-stroke);text-anchor:middle"><tspan x="235" dy="0">3 + 3 = 6</tspan></text></g><g data-ring-phase="reduce-scatter" data-step="3" data-incoming="6" data-local="4" data-result="10"><text x="20" y="570" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">r2 -&gt; r3</tspan></text><text x="235" y="570" style="font-size:17px;font-weight:700;fill:var(--viz-focus-stroke);text-anchor:middle"><tspan x="235" dy="0">6 + 4 = 10</tspan></text></g><text x="180" y="610" style="font-size:12px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="180" dy="0">ALL-GATHER: forward the same finished value</tspan></text><g data-ring-phase="all-gather" data-step="1" data-incoming="10" data-result="10"><text x="20" y="650" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">r3 -&gt; r0</tspan></text><text x="235" y="650" style="font-size:17px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="235" dy="0">10 -&gt; 10</tspan></text></g><g data-ring-phase="all-gather" data-step="2" data-incoming="10" data-result="10"><text x="20" y="694" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">r0 -&gt; r1</tspan></text><text x="235" y="694" style="font-size:17px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="235" dy="0">10 -&gt; 10</tspan></text></g><g data-ring-phase="all-gather" data-step="3" data-incoming="10" data-result="10"><text x="20" y="738" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">r1 -&gt; r2</tspan></text><text x="235" y="738" style="font-size:17px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="235" dy="0">10 -&gt; 10</tspan></text></g><text x="180" y="789" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">EACH RANK SENDS SIX B/4 CHUNKS</tspan></text><g data-ring-sent-chunk="0" data-phase="reduce-scatter"><rect x="15" y="813" width="55" height="35" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="42.5" y="836" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="42.5" dy="0">B/4</tspan></text></g><g data-ring-sent-chunk="1" data-phase="reduce-scatter"><rect x="70" y="813" width="55" height="35" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="97.5" y="836" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="97.5" dy="0">B/4</tspan></text></g><g data-ring-sent-chunk="2" data-phase="reduce-scatter"><rect x="125" y="813" width="55" height="35" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="152.5" y="836" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="152.5" dy="0">B/4</tspan></text></g><g data-ring-sent-chunk="3" data-phase="all-gather"><rect x="180" y="813" width="55" height="35" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="207.5" y="836" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="207.5" dy="0">B/4</tspan></text></g><g data-ring-sent-chunk="4" data-phase="all-gather"><rect x="235" y="813" width="55" height="35" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="262.5" y="836" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="262.5" dy="0">B/4</tspan></text></g><g data-ring-sent-chunk="5" data-phase="all-gather"><rect x="290" y="813" width="55" height="35" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="317.5" y="836" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="317.5" dy="0">B/4</tspan></text></g><text x="180" y="885" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">(3 + 3) * B/4 = 3B/2 sent; not sent + received.</tspan></text></svg><div class="visual-derivation-caption">The ring inset is one simultaneous step, not the whole schedule. Follow A separately: 1 becomes 3, then 6, then 10; all-gather preserves 10. Every rank sends one B/4 chunk in each of six steps, for 3B/2 sent bytes.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="ring-all-reduce-chunk-trace"><strong>Check:</strong> Where are the other chunks while A moves? Why does each rank send six chunks of size B/4?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="ring-all-reduce-chunk-trace">For one illustrative chunk, rank values 1,2,3,4 reduce to 10. All-gather distributes that completed chunk; it does not add the sum again.</span></figcaption></figure>

## Change 1: The Consumer Needs Only Its Shard

**New task.** The optimizer needs its own portion of the summed gradient, not the full sum everywhere.

**Picture and change.** Stop after reduce-scatter and preserve the optimizer's shard mapping.
Do not all-gather a result merely to discard the other shards.

**Check.** The two-rank outputs remain `[4]` and `[6]`, not two full buffers.
In the four-rank ring model, this phase sends 6 MB per rank instead of 12 MB.

**Use when.** Downstream work can consume sharded results.

## Change 2: Thousands of Tiny Buffers

**New task.** Replace one 8 MB buffer with 1,000 separate 8 KB buffers.

**Picture and change.** Total ring bytes stay fixed, but sequential communication steps multiply.
For effective bandwidth $b$ and per-step latency $\alpha$, use the simplified unoverlapped model

$$
T_{\mathrm{ring}}\approx2(N-1)\alpha+\frac{2B(N-1)}{Nb}.
$$

At four ranks, $\alpha=5$ microseconds and $b=1$ GB/s give 12.03 ms for the single buffer.
The 1,000 serial collectives take 42 ms in this model, despite sending the same 12 MB per rank.

**Check.** Measure backend launch/hop costs, bucketing, tree alternatives and actual topology before selecting an algorithm.
These calculations are not hardware measurements.

**Use when.** Fixed overhead dominates small messages.

## Failure: Ranks Disagree About the Operation

A skipped call or mismatched shape, dtype, chunk order or collective sequence can hang, fail or invalidate results.
All participating ranks must follow the process group's matching communication contract.

## Transfer Check

Does a long total collective duration establish the training bottleneck?

### Reasoned Answer

No. Measure its exposed critical-path tail after overlap and contention.
Summed durations can include work already hidden behind computation.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Where Each Appears

| Distributed pattern | Collectives |
|---------------------|------------|
| **DDP** (data parallel) | Gradient all-reduce at synchronized backward boundaries; accumulation can defer synchronization |
| **FSDP / ZeRO-3** | Parameter all-gather when needed; gradient reduce-scatter under the selected sharding strategy |
| **Tensor parallelism** | All-reduce, reduce-scatter or all-gather according to input/output layouts |
| **Pipeline parallelism** | Point-to-point sends (not collective) between adjacent stages |
| **Expert parallelism (MoE)** | All-to-all to route tokens to experts |
| **Embedding lookup at scale** | All-to-all to gather sharded embedding rows |

## All-to-all

Each rank sends destination-specific pieces; receive order follows the agreed source layout.
MoE routing and layout changes use this exchange.
Cost depends on peer traffic, balance, bisection bandwidth and topology, not the operation name alone.

## Hardware backends

- **NCCL:** NVIDIA accelerator communication.
- **RCCL:** AMD accelerator communication.
- **MPI:** broader high-performance message passing.
- **Gloo:** commonly used for CPU collectives and supported distributed interfaces.

Supported operations and selected algorithms depend on version, hardware, group and message size.

## Bandwidth and topology

Distinguish local accelerator links from inter-node or inter-slice networks.
Hierarchical algorithms combine local and scale-out phases.
Use measured effective bandwidth for the message and schedule, not advertised peak bandwidth.

## Cost model in DDP

For $P$ parameters and gradient element size $s$, $B=Ps$ bytes in the ring model.
Suppose seven billion gradients use BF16, 32 ranks participate, and effective ring bandwidth is 200 GB/s.
Then $B=14$ GB and per-rank sent traffic is 27.125 GB.
The bandwidth term is 135.625 ms, not a measured step time.

## Common pitfalls

Buckets trade fixed overhead against readiness, memory and overlap.
DDP/FSDP overlap depends on configuration and schedule; inspect traces rather than assuming it.
Reduction order can change floating-point rounding even when mathematical outputs agree.
The integer reference does not model those numerical differences or diagnose distributed hangs.

## Related

- [Tensor parallelism](/concepts/tensor-parallelism/). Heavy collective user.
- [FSDP and ZeRO](/concepts/fsdp-and-zero/). Uses all-gather and reduce-scatter.
- [Pipeline parallelism](/concepts/pipeline-parallelism/). Uses point-to-point, not collectives.
- [Sharded matrix multiplication](/concepts/sharded-matrix-multiplication/). Select collectives from array layouts.
- [Accelerator network topology](/concepts/accelerator-network-topology/). Understand the links that carry them.

</div>
