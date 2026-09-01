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
<figure class="learning-figure coding-visual-figure" aria-labelledby="causal-attention-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="causal-attention-state-title">Causal Attention: Form scaled query-key scores, mask the strict upper triangle before softmax, then use prefix-only weights to mix values.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="causal-attention" role="group" tabindex="0" aria-label="Causal Attention: Form scaled query-key scores, mask the strict upper triangle before softmax, then use prefix-only weights to mix values."><div class="coding-visual-example"><span>Input and goal</span><strong>Compute one attention head where each token can read only itself and earlier tokens.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="dot-products" role="group" aria-label="Multiply queries by keys"><div class="coding-trace-frame-heading"><span>Multiply queries by keys</span><strong>Use Q=K=[[1,0],[0,1]] and V=[[10,0],[0,20]]. QK^T is the 2 by 2 identity score matrix.</strong></div><div class="coding-trace-attention" style="--trace-cols:2"><span class="coding-trace-attention-cell ">1</span><span class="coding-trace-attention-cell ">0</span><span class="coding-trace-attention-cell ">0</span><span class="coding-trace-attention-cell ">1</span></div><div class="coding-trace-meta"><span><b>axes</b>query rows x key rows</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="scale" hidden role="group" aria-label="Scale by square root of width"><div class="coding-trace-frame-heading"><span>Scale by square root of width</span><strong>Key width is 2, so divide by sqrt(2). Scores become [[0.7071,0],[0,0.7071]].</strong></div><div class="coding-trace-attention" style="--trace-cols:2"><span class="coding-trace-attention-cell ">0.7071</span><span class="coding-trace-attention-cell ">0</span><span class="coding-trace-attention-cell ">0</span><span class="coding-trace-attention-cell ">0.7071</span></div><div class="coding-trace-meta"><span><b>scale</b>1 / sqrt(2)</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="mask" hidden role="group" aria-label="Mask every future key"><div class="coding-trace-frame-heading"><span>Mask every future key</span><strong>The strict upper triangle is future context. Set score (query 0,key 1) to -infinity before softmax; row 1 may still read keys 0 and 1.</strong></div><div class="coding-trace-attention" style="--trace-cols:2"><span class="coding-trace-attention-cell ">0.7071</span><span class="coding-trace-attention-cell is-mask">mask</span><span class="coding-trace-attention-cell ">0</span><span class="coding-trace-attention-cell ">0.7071</span></div><div class="coding-trace-meta"><span><b>rule</b>key index &gt; query index -&gt; -infinity</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="softmax" hidden role="group" aria-label="Normalize each allowed prefix"><div class="coding-trace-frame-heading"><span>Normalize each allowed prefix</span><strong>Row 0 softmax is [1,0]. Row 1 softmax of [0,0.7071] is approximately [0.3302,0.6698]. Masked future weight is exactly zero.</strong></div><div class="coding-trace-attention" style="--trace-cols:2"><span class="coding-trace-attention-cell ">1.0000</span><span class="coding-trace-attention-cell is-mask">mask</span><span class="coding-trace-attention-cell ">0.3302</span><span class="coding-trace-attention-cell ">0.6698</span></div><div class="coding-trace-meta"><span><b>rowSums</b>1.0000; 1.0000</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="mix-values" hidden role="group" aria-label="Mix value rows"><div class="coding-trace-frame-heading"><span>Mix value rows</span><strong>Query 0 output is 1*V0 = [10,0]. Query 1 output is 0.3302*V0 + 0.6698*V1 = [3.30,13.40].</strong></div><div class="coding-trace-attention" style="--trace-cols:2"><span class="coding-trace-attention-cell ">10.00</span><span class="coding-trace-attention-cell ">0.00</span><span class="coding-trace-attention-cell ">3.30</span><span class="coding-trace-attention-cell ">13.40</span></div><div class="coding-trace-meta"><span><b>operation</b>weights @ V</span><span><b>result</b>[[10,0],[3.30,13.40]]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Multiply queries by keys</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Scale by square root of width</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Mask every future key</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Normalize each allowed prefix</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Mix value rows</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Form scaled query-key scores, mask the strict upper triangle before softmax, then use prefix-only weights to mix values.</p></div><figcaption><strong>Read it this way:</strong> Use Q=K=[[1,0],[0,1]] and V=[[10,0],[0,20]]. QK^T is the 2 by 2 identity score matrix. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
