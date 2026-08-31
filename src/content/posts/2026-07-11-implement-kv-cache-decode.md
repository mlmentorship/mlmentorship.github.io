---
title: "Implement incremental decoding with a KV cache"
description: "Prove that one-token cached attention matches full-prefix attention while preserving shape, dtype, growth, and memory invariants."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> Implement KV-cache append and one-token attention for autoregressive decoding. Show equivalence to full attention at every prefix.

The core invariant is exactness: for the same model state and prefix, the newest-token output from cached decoding should match the newest-token output from a full causal forward pass, up to numerical tolerance.

## Contract first

Use tensors shaped `[batch, kv_heads, time, head_dim]` for keys and values and `[batch, query_heads, 1, head_dim]` for the newest query. State whether query heads equal KV heads or require grouped expansion.

For a basic equal-head implementation:

1. validate new key and value shapes;
2. append along the time dimension;
3. compute scores from the newest query against all cached keys;
4. scale by $1 / \sqrt{d_h}$;
5. softmax over cached positions;
6. multiply by cached values;
7. return one output position and retain the grown cache.

No causal mask is needed when the query is the newest token and the cache contains only its prefix plus itself. There is no future position to block.

<!-- visual:kv-cache-newest-row-equivalence -->
<figure class="learning-figure" aria-labelledby="kv-equivalence-title" aria-describedby="kv-equivalence-description">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="kv-equivalence-title">Why does cached decoding produce the same newest-token output?</p>
	<p id="kv-equivalence-description">At prefix length four, compare the projection work in a full causal pass with a cached step. Both compute the fourth output from the same newest query, four keys, and four values.</p>
	<div class="visual-grid--two" role="group" aria-label="Full-prefix and cached attention at prefix length four">
		<section class="visual-panel" aria-labelledby="kv-full-title">
			<h4 id="kv-full-title">Full-prefix pass: recompute history</h4>
			<table class="cm-grid" aria-label="Projection work and newest-row attention in a full-prefix pass">
				<tbody>
					<tr><th scope="row">Project now</th><td>Q<sub>1</sub>–Q<sub>4</sub>, K<sub>1</sub>–K<sub>4</sub>, and V<sub>1</sub>–V<sub>4</sub></td></tr>
					<tr><th scope="row">Use for newest row</th><td>q<sub>4</sub></td></tr>
					<tr><th scope="row">Attend to</th><td>K = [k<sub>1</sub>, k<sub>2</sub>, k<sub>3</sub>, k<sub>4</sub>]<br>V = [v<sub>1</sub>, v<sub>2</sub>, v<sub>3</sub>, v<sub>4</sub>]</td></tr>
					<tr><th scope="row">Keep</th><td>Newest row o<sub>4</sub></td></tr>
				</tbody>
			</table>
		</section>
		<section class="visual-panel" aria-labelledby="kv-cached-title">
			<h4 id="kv-cached-title">Cached step: reuse history</h4>
			<table class="cm-grid" aria-label="Projection work and one-token attention in a cached decoding step">
				<tbody>
					<tr><th scope="row">Reuse cached</th><td>K<sub>1</sub>–K<sub>3</sub> and V<sub>1</sub>–V<sub>3</sub></td></tr>
					<tr><th scope="row">Project now</th><td>q<sub>4</sub>, k<sub>4</sub>, and v<sub>4</sub></td></tr>
					<tr><th scope="row">Attend to</th><td>K = [k<sub>1</sub>, k<sub>2</sub>, k<sub>3</sub>, k<sub>4</sub>]<br>V = [v<sub>1</sub>, v<sub>2</sub>, v<sub>3</sub>, v<sub>4</sub>]</td></tr>
					<tr><th scope="row">Return</th><td>One position o<sub>4</sub></td></tr>
				</tbody>
			</table>
		</section>
	</div>
	<p class="cm-equation">Both paths: softmax(q<sub>4</sub>K<sup>T</sup> / √d<sub>h</sub>)V = o<sub>4</sub></p>
	<figcaption><strong>Read it this way:</strong> compare the “Attend to” rows first: they contain the same ordered keys and values, so the newest query performs the same computation and produces the same o<sub>4</sub> up to floating-point tolerance. Then compare the projection rows: caching saves work by reusing k<sub>1</sub>–k<sub>3</sub> and v<sub>1</sub>–v<sub>3</sub>; it does not approximate or shorten attention. Original comparison checked against the <a href="https://arxiv.org/abs/1706.03762">Transformer attention definition</a> and <a href="https://huggingface.co/docs/transformers/main/en/cache_explanation">Hugging Face cache documentation</a>.</figcaption>
</figure>

## Reference sketch

```python
if cache.keys is None:
    cache.keys = new_key
    cache.values = new_value
else:
    if new_key.shape[:-2] != cache.keys.shape[:-2] or new_key.shape[-1] != cache.keys.shape[-1]:
        raise ValueError("incompatible key shape")
    if new_key.dtype != cache.keys.dtype or new_key.device != cache.keys.device:
        raise ValueError("key dtype or device does not match cache")
    cache.keys = torch.cat((cache.keys, new_key), dim=-2)
    cache.values = torch.cat((cache.values, new_value), dim=-2)

scores = query @ cache.keys.transpose(-1, -2) / math.sqrt(query.size(-1))
weights = torch.softmax(scores.float(), dim=-1).to(cache.values.dtype)
return weights @ cache.values
```

This baseline is correct but reallocates on every append. That is acceptable for the first implementation and unacceptable for a production server.

## What an L4 answer sounds like

The candidate stores previous hidden states rather than projected keys and values, recomputes the prefix, or appends along the head dimension. The code produces a plausible shape but no equivalence test.

## What an L5 answer adds

An L5 candidate writes a prefix-by-prefix test against full attention. They validate batch, head, and head-dimension compatibility, preserve dtype and device, and explain why only K and V are cached.

They notice cache lifecycle:

- reset between unrelated sequences;
- reorder after beam expansion;
- release on EOS or cancellation;
- track actual sequence length separately from allocated capacity;
- handle batched sequences with different lengths.

## What an L6 answer adds

An L6 candidate distinguishes the simple tensor cache from a serving cache. Production systems preallocate or page memory rather than concatenate every step. They discuss block tables, fragmentation, copy-on-write for shared prefixes, GQA or MQA, quantized KV, eviction policy, and admission based on future cache growth.

They also cover position semantics. Rotary position encoding must use the absolute position of the new token. A cache copied into a new sequence with the wrong position offset can preserve shapes while producing incorrect attention.

For beam search, cache state follows hypotheses. Reordering beams requires reindexing each layer's K and V consistently. Finished beams stop growing.

## Tells that get you a strong-hire vote

- The newest cached output is compared against full-prefix attention at every step.
- You append on the sequence dimension and validate all other dimensions.
- You know why a newest-token query needs no future mask.
- Cache reset, release, and beam reorder semantics are explicit.
- FP32 softmax and low-precision cache behavior are considered.
- You call out repeated concatenation as a toy-only allocation strategy.
- Position encoding and cache length agree.

## Tells that get you down-leveled

- Caching Q as well as K and V with no reason.
- Recomputing projections for old tokens.
- A shape-only test.
- Ignoring cache ownership across requests.
- Claiming total generation becomes linear. Per-step attention is linear in prefix, so total attention remains quadratic without further changes.
- Discussing paged attention before a correct baseline exists.

## Common follow-up

"How does GQA change your implementation?"

The cache stores fewer KV heads than query heads. Query heads are partitioned into groups that share one K and V head. The implementation maps or expands KV heads for the attention operation without physically duplicating the stored cache. Memory falls roughly with the ratio of query heads to KV heads.

Use the [KV-cache starter and equivalence tests](/prep/labs/implementation/) before reading this page twice.

*Related: [KV cache](/concepts/kv-cache/), [paged attention](/concepts/paged-attention/), and [continuous batching](/concepts/continuous-batching/).*
