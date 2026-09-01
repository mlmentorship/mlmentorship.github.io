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

> Compute one attention head where each token can read only itself and earlier tokens.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:causal-attention-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="causal-attention-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="causal-attention-state-title">Causal Attention: Each attention row can read its own position and every earlier position, never a future one.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="causal-attention" role="group" tabindex="0" aria-label="Causal Attention: Each attention row can read its own position and every earlier position, never a future one."><div class="coding-visual-example"><span>Input and goal</span><strong>Compute one attention head where each token can read only itself and earlier tokens.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Build all pair scores"><div class="coding-trace-frame-heading"><span>Build all pair scores</span><strong>Query-key scores start as a full square matrix.</strong></div><div class="coding-trace-attention" style="--trace-cols:3"><span class="coding-trace-attention-cell ">.</span><span class="coding-trace-attention-cell ">.</span><span class="coding-trace-attention-cell ">.</span><span class="coding-trace-attention-cell ">.</span><span class="coding-trace-attention-cell ">.</span><span class="coding-trace-attention-cell ">.</span><span class="coding-trace-attention-cell ">.</span><span class="coding-trace-attention-cell ">.</span><span class="coding-trace-attention-cell ">.</span></div><div class="coding-trace-meta"><span><b>action</b>QK^T</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Apply the causal mask"><div class="coding-trace-frame-heading"><span>Apply the causal mask</span><strong>Future positions become forbidden before softmax.</strong></div><div class="coding-trace-attention" style="--trace-cols:3"><span class="coding-trace-attention-cell is-read">read</span><span class="coding-trace-attention-cell is-mask">mask</span><span class="coding-trace-attention-cell is-mask">mask</span><span class="coding-trace-attention-cell is-read">read</span><span class="coding-trace-attention-cell is-read">read</span><span class="coding-trace-attention-cell is-mask">mask</span><span class="coding-trace-attention-cell is-read">read</span><span class="coding-trace-attention-cell is-read">read</span><span class="coding-trace-attention-cell is-read">read</span></div><div class="coding-trace-meta"><span><b>action</b>mask future scores</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Mix allowed values"><div class="coding-trace-frame-heading"><span>Mix allowed values</span><strong>Each row can assign weights to its prefix, while every future weight is zero.</strong></div><div class="coding-trace-attention" style="--trace-cols:3"><span class="coding-trace-attention-cell ">w0</span><span class="coding-trace-attention-cell is-mask">mask</span><span class="coding-trace-attention-cell is-mask">mask</span><span class="coding-trace-attention-cell ">w0</span><span class="coding-trace-attention-cell ">w1</span><span class="coding-trace-attention-cell is-mask">mask</span><span class="coding-trace-attention-cell ">w0</span><span class="coding-trace-attention-cell ">w1</span><span class="coding-trace-attention-cell ">w2</span></div><div class="coding-trace-meta"><span><b>result</b>prefix-only reads; each row sums to 1</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Build all pair scores</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Apply the causal mask</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Mix allowed values</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Each attention row can read its own position and every earlier position, never a future one.</p></div><figcaption><strong>Read it this way:</strong> Query-key scores start as a full square matrix. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Matrix multiplication, scaling, mask, softmax, matrix multiplication.

**Simple idea:** Build query-key scores. Divide by the square root of key width. Set future
scores to negative infinity before softmax. Use the probabilities to mix value rows.

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

**Cost:** $O(sequence^2 \times width)$ time and $O(sequence^2)$ score space.
