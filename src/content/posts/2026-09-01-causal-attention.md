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
<figure class="learning-figure coding-visual-figure" aria-labelledby="causal-attention-state-title"><p class="visual-kicker">Causal information flow</p><p class="visual-title" id="causal-attention-state-title">Causal Attention: Mask future positions before probabilities are formed</p><div class="coding-visual coding-visual--attention" data-coding-visual data-coding-mode="attention" data-coding-slug="causal-attention" role="group" aria-label="Causal Attention: token 2 can read tokens 0,1,2 but the future score is masked out. Row i assigns probability only to keys at positions 0 through i."><div class="coding-visual-example"><span>Concrete trace</span><strong>token 2 can read tokens 0,1,2 but the future score is masked out</strong></div><div class="coding-visual-sketch coding-visual-sketch--attention"><div class="coding-sketch-attention-grid"><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">read</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">mask</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">mask</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">read</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">read</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">mask</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">read</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">read</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">read</span></div><p class="coding-sketch-note">row i keeps columns 0 through i and masks every future column</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Score</span><strong>QKᵀ</strong><small>Compare every query with every key.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Scale</span><strong>divide by √d</strong><small>Keep score magnitudes stable across key widths.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Mask</span><strong>future = −∞</strong><small>Make forbidden positions receive zero probability.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Mix</span><strong>weights × V</strong><small>Read only the prefix allowed for each token.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Row i assigns probability only to keys at positions 0 through i.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The triangular mask is the lesson. Scores may be computed for every pair, but future entries are removed before softmax can give them weight. For this problem, hold onto the concrete trace: token 2 can read tokens 0,1,2 but the future score is masked out.</figcaption></figure>

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
