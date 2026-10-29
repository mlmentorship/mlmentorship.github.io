---
title: "Causal Attention"
description: "Compute one attention head where each token can read only itself and earlier tokens."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Intermediate"
priority: "Role-specific"
aliases: []
prerequisites: []
---

Compute attention outputs for a sequence while preventing every position from using information from future positions.

## Problem

**Input.** Float64 arrays `query` and `key` with shape `(n, d)`, and `value` with shape `(n, dv)`.

**Output.** Return an `(n, dv)` array containing one scaled dot-product attention result per query position.
Scores use dot products divided by the square root of `d`, normalized over permitted keys.

**Rules.** Query position `i` may use positions `0` through `i`, including itself, but no later position.
Require positive `n` and `d`, aligned sequence positions, and representable finite scores.
Preserve the inputs. Repeated values and positions' content are allowed.

**Example.** With two all-zero query/key rows and value rows `[[2], [6]]`, return `[[2], [4]]`.
The first position sees only 2; the second averages both values because their scores are equal.

## Picture

With `Q=K=[[1,0],[0,1]]`, scores are the identity divided by $\sqrt2$.
Query zero sees only key zero; query one sees both keys.
For `V=[[10,0],[0,20]]`, output zero is `[10,0]`; output one is approximately `[3.3024,13.3952]`.

<!-- visual:causal-attention-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="causal-attention-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="causal-attention-state-title">Future positions receive exactly zero attention weight</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="causal-attention" role="group" tabindex="0" aria-label="Causal Attention: Form scaled query-key scores, mask the strict upper triangle before softmax, then use prefix-only weights to mix values."><div class="coding-visual-example"><span>Input and goal</span><strong>Compute one attention head where each token can read only itself and earlier tokens.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="dot-products" role="group" aria-label="Multiply queries by keys"><div class="coding-trace-frame-heading"><span>Multiply queries by keys</span><strong>Use Q=K=[[1,0],[0,1]] and V=[[10,0],[0,20]]. QK^T is the 2 by 2 identity score matrix.</strong></div><div class="coding-trace-attention" style="--trace-cols:2"><span class="coding-trace-attention-cell " data-motion-key="attention-0-0">1</span><span class="coding-trace-attention-cell " data-motion-key="attention-0-1">0</span><span class="coding-trace-attention-cell " data-motion-key="attention-1-0">0</span><span class="coding-trace-attention-cell " data-motion-key="attention-1-1">1</span></div><div class="coding-trace-meta"><span><b>axes</b>query rows x key rows</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>axes</dt><dd><strong class="coding-state-after">query rows x key rows</strong></dd></div><div class="coding-state-change"><dt>column 0 / row 0</dt><dd><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>column 1 / row 0</dt><dd><strong class="coding-state-after">0</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="scale" hidden role="group" aria-label="Scale by square root of width"><div class="coding-trace-frame-heading"><span>Scale by square root of width</span><strong>Key width is 2, so divide by sqrt(2). Scores become [[0.7071,0],[0,0.7071]].</strong></div><div class="coding-trace-attention" style="--trace-cols:2"><span class="coding-trace-attention-cell " data-motion-key="attention-0-0">0.7071</span><span class="coding-trace-attention-cell " data-motion-key="attention-0-1">0</span><span class="coding-trace-attention-cell " data-motion-key="attention-1-0">0</span><span class="coding-trace-attention-cell " data-motion-key="attention-1-1">0.7071</span></div><div class="coding-trace-meta"><span><b>scale</b>1 / sqrt(2)</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>scale</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1 / sqrt(2)</strong></dd></div><div class="coding-state-change"><dt>axes</dt><dd><span class="coding-state-before">query rows x key rows</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>column 0 / row 0</dt><dd><span class="coding-state-before">1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0.7071</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="mask" hidden role="group" aria-label="Mask every future key"><div class="coding-trace-frame-heading"><span>Mask every future key</span><strong>The strict upper triangle is future context. Set score (query 0,key 1) to -infinity before softmax; row 1 may still read keys 0 and 1.</strong></div><div class="coding-trace-attention" style="--trace-cols:2"><span class="coding-trace-attention-cell " data-motion-key="attention-0-0">0.7071</span><span class="coding-trace-attention-cell is-mask" data-motion-key="attention-0-1">mask</span><span class="coding-trace-attention-cell " data-motion-key="attention-1-0">0</span><span class="coding-trace-attention-cell " data-motion-key="attention-1-1">0.7071</span></div><div class="coding-trace-meta"><span><b>rule</b>key index &gt; query index -&gt; -infinity</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>rule</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">key index &gt; query index -&gt; -infinity</strong></dd></div><div class="coding-state-change"><dt>scale</dt><dd><span class="coding-state-before">1 / sqrt(2)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>column 1 / row 0</dt><dd><span class="coding-state-before">0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">mask</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="softmax" hidden role="group" aria-label="Normalize each allowed prefix"><div class="coding-trace-frame-heading"><span>Normalize each allowed prefix</span><strong>Row 0 softmax is [1,0]. Row 1 softmax of [0,0.7071] is approximately [0.3302,0.6698]. Masked future weight is exactly zero.</strong></div><div class="coding-trace-attention" style="--trace-cols:2"><span class="coding-trace-attention-cell " data-motion-key="attention-0-0">1.0000</span><span class="coding-trace-attention-cell is-mask" data-motion-key="attention-0-1">mask</span><span class="coding-trace-attention-cell " data-motion-key="attention-1-0">0.3302</span><span class="coding-trace-attention-cell " data-motion-key="attention-1-1">0.6698</span></div><div class="coding-trace-meta"><span><b>rowSums</b>1.0000; 1.0000</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>row sums</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1.0000; 1.0000</strong></dd></div><div class="coding-state-change"><dt>rule</dt><dd><span class="coding-state-before">key index &gt; query index -&gt; -infinity</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>column 0 / row 0</dt><dd><span class="coding-state-before">0.7071</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1.0000</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="mix-values" hidden role="group" aria-label="Mix value rows"><div class="coding-trace-frame-heading"><span>Mix value rows</span><strong>Query 0 output is 1*V0 = [10,0]. Query 1 output is 0.3302*V0 + 0.6698*V1 = [3.30,13.40].</strong></div><div class="coding-trace-attention" style="--trace-cols:2"><span class="coding-trace-attention-cell " data-motion-key="attention-0-0">10.00</span><span class="coding-trace-attention-cell " data-motion-key="attention-0-1">0.00</span><span class="coding-trace-attention-cell " data-motion-key="attention-1-0">3.30</span><span class="coding-trace-attention-cell " data-motion-key="attention-1-1">13.40</span></div><div class="coding-trace-meta"><span><b>operation</b>weights @ V</span><span><b>result</b>[[10,0],[3.30,13.40]]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>operation</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">weights @ V</strong></dd></div><div class="coding-state-change"><dt>result</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[[10,0],[3.30,13.40]]</strong></dd></div><div class="coding-state-change"><dt>row sums</dt><dd><span class="coding-state-before">1.0000; 1.0000</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Multiply queries by keys</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Scale by square root of width</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Mask every future key</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Normalize each allowed prefix</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Mix value rows</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when each sequence position may aggregate only its own and earlier value rows, requiring future key positions to receive exactly zero normalized weight.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Attention row i contains scores for query i against every key; after masking and softmax, entries j&gt;i are zero and allowed prefix weights j&lt;=i sum to one.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">For constrained attention, encode allowed information flow as a score-space mask before normalization so forbidden positions receive zero mass, then mix values with the resulting row-stochastic weights.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Use Q=K=[[1,0],[0,1]] and V=[[10,0],[0,20]]. QK^T is the 2 by 2 identity score matrix. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="causal-attention-state"><strong>Check:</strong> Why can query 0 not use V1 even if that future key would have a high score?</span></figcaption></figure>

The [attention owner](/concepts/attention-mechanism/) explains the weighted mixture.
Here the extra contract is visibility: a strict upper triangle contributes no exponential mass before rowwise normalization.

## Code

```python
import numpy as np

def causal_attention(
   query: np.ndarray, key: np.ndarray, value: np.ndarray
) -> np.ndarray:
   scores = query @ key.T / np.sqrt(query.shape[-1])
   future = np.triu(np.ones(scores.shape, dtype=bool), k=1)
   scores[future] = -np.inf
   exponentials = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
   weights = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
   return weights @ value
```

**Cost:** $O(n^2(d+dv))$ arithmetic and $O(n^2)$ score/weight workspace, plus $O(ndv)$ output.

## Change the Requirement

### Add Padding or Explicit Visibility

**New task:** Accept a Boolean `(query_rows,key_rows)` visibility mask.

**Picture and change:** Compose padding and causality before softmax; reject rows without any allowed key.

**Code:** Retain positive feature width and finite representable scores.

```python
def attention_with_mask(query, key, value, allowed):
   scores = query @ key.T / np.sqrt(query.shape[-1])
   allowed = np.asarray(allowed, dtype=bool)
   if allowed.shape != scores.shape or np.any(~allowed.any(axis=-1)):
      raise ValueError("each query needs a matching nonempty visibility row")
   scores = np.where(allowed, scores, -np.inf)
   exponentials = np.exp(scores - scores.max(axis=-1, keepdims=True))
   weights = exponentials / exponentials.sum(axis=-1, keepdims=True)
   return weights @ value
```

**Check:** For the example, mask `[[True,False],[True,False]]` produces `[10,0]` for both queries.

**Use when:** Padding removes otherwise visible keys.
Multiplying weights by a mask afterward does not renormalize the distribution.

### Decode Against Cached Positions

**New task:** Permit rectangular query/key arrays with explicit integer positions.

**Picture and change:** A key is visible when its absolute position does not exceed the query position.

**Code:** Position-vector lengths must match their corresponding arrays.

```python
def cached_causal_attention(query, key, value, query_positions, key_positions):
   allowed = np.asarray(key_positions)[None, :] <= np.asarray(query_positions)[:, None]
   return attention_with_mask(query, key, value, allowed)
```

**Check:** One query at position `2` can see cached keys at positions `0,1`; both receive nonzero weights.

**Use when:** Local row zero is not absolute sequence position zero.
Costs become $O(n_qn_k(d+dv))$ arithmetic and $O(n_qn_k)$ score workspace.

<!-- learning:core-end -->
<div data-learning-extension>

## Reconstruct

Why does a one-row upper-triangular mask fail during cached decoding?

<details>
<summary>Reasoned answer</summary>

It treats the query as position zero and hides every key column after zero, even when those keys are historical.
Build visibility from absolute positions and separately exclude invalid cached entries when padding is present.
An all-false row is rejected rather than evaluated as negative-infinity minus negative-infinity.

</details>

</div>
