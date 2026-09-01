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
<figure class="learning-figure coding-visual-figure" aria-labelledby="stable-softmax-state-title"><p class="visual-kicker">Stable numerical path</p><p class="visual-title" id="stable-softmax-state-title">Stable Softmax: Change the reference point before exponentiating</p><div class="coding-visual coding-visual--numerics" data-coding-visual data-coding-mode="numerics" data-coding-slug="stable-softmax" role="group" aria-label="Stable Softmax: logits [1000,1001] shift to [-1,0] before exponentiation. Subtracting one row constant changes no softmax probabilities or cross-entropy differences."><div class="coding-visual-example"><span>Concrete trace</span><strong>logits [1000,1001] shift to [-1,0] before exponentiation</strong></div><div class="coding-visual-sketch coding-visual-sketch--numerics"><div class="coding-sketch-array"><span class="coding-sketch-cell coding-sketch-cell--state">large</span><span class="coding-sketch-arrow">&minus; max</span><span class="coding-sketch-cell coding-sketch-cell--active">small</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell">safe exp</span></div><p class="coding-sketch-note">relative differences stay; raw magnitude stops causing overflow</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Anchor</span><strong>row maximum</strong><small>Choose a value that keeps shifted logits small.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Shift</span><strong>logits − max</strong><small>Preserve relative differences while avoiding overflow.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Normalize</span><strong>logsumexp or softmax</strong><small>Aggregate the shifted exponentials stably.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Select</span><strong>correct class</strong><small>Read the requested probability or loss term.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Subtracting one row constant changes no softmax probabilities or cross-entropy differences.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The large raw numbers are not the information. Their differences are. Shift the row first, then exponentiate values that are numerically safe. For this problem, hold onto the concrete trace: logits [1000,1001] shift to [-1,0] before exponentiation.</figcaption></figure>

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
