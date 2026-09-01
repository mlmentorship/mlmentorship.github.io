---
title: "Stable Softmax"
description: "Convert logits to probabilities without numeric overflow."
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

> Convert logits to probabilities without numeric overflow.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:stable-softmax-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="stable-softmax-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="stable-softmax-state-title">Stable Softmax: Subtract the row maximum before exponentiating; relative gaps do not change.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="stable-softmax" role="group" tabindex="0" aria-label="Stable Softmax: Subtract the row maximum before exponentiating; relative gaps do not change."><div class="coding-visual-example"><span>Input and goal</span><strong>Convert logits to probabilities without numeric overflow.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="See the large logits"><div class="coding-trace-frame-heading"><span>See the large logits</span><strong>Exponentiating 1000 and 1001 directly can overflow.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1000</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="marker-row-max">row max</span><span class="coding-trace-array-cell">1001</span><small class="coding-trace-array-index">1</small></span></div><div class="coding-trace-meta"><span><b>detail</b>raw logits</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Shift the row"><div class="coding-trace-frame-heading"><span>Shift the row</span><strong>Subtract 1001 to get [-1,0].</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-shifted">shifted</span><span class="coding-trace-array-cell">-1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="marker-shifted">shifted</span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">1</small></span></div><div class="coding-trace-meta"><span><b>action</b>logits - max</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Normalize safely"><div class="coding-trace-frame-heading"><span>Normalize safely</span><strong>exp(-1) and exp(0) divide by their sum to form probabilities.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-p">p</span><span class="coding-trace-array-cell">0.2689</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="marker-p">p</span><span class="coding-trace-array-cell">0.7311</span><small class="coding-trace-array-index">1</small></span></div><div class="coding-trace-meta"><span><b>result</b>sum = 1</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>See the large logits</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Shift the row</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Normalize safely</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Subtract the row maximum before exponentiating; relative gaps do not change.</p></div><figcaption><strong>Read it this way:</strong> Exponentiating 1000 and 1001 directly can overflow. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Shift before exponentiation.

**Simple idea:** Softmax does not change when the same value is subtracted from every logit.
Subtract the largest logit so every exponent is at most 1.

```python
import numpy as np

def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
   shifted = logits - np.max(logits, axis=axis, keepdims=True)
   exponentials = np.exp(shifted)
   return exponentials / np.sum(exponentials, axis=axis, keepdims=True)
```

**Cost:** $O(n)$ time and $O(n)$ output space.
