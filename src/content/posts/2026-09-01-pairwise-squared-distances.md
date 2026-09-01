---
title: "Pairwise Squared Distances"
description: "Compute the squared distance from every point to every center without Python loops."
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

> Compute the squared distance from every point to every center without Python loops.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:pairwise-squared-distances-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="pairwise-squared-distances-state-title"><p class="visual-kicker">Singleton axes expand</p><p class="visual-title" id="pairwise-squared-distances-state-title">Pairwise Squared Distances: One point meets every center without a Python loop</p><div class="coding-visual coding-visual--broadcast" data-coding-visual data-coding-mode="broadcast" data-coding-slug="pairwise-squared-distances" role="group" aria-label="Pairwise Squared Distances: points [n,1,d] and centers [1,k,d] broadcast to [n,k,d]. The final two axes identify one point-center pair and its feature coordinates."><div class="coding-visual-example"><span>Concrete trace</span><strong>points [n,1,d] and centers [1,k,d] broadcast to [n,k,d]</strong></div><div class="coding-visual-sketch coding-visual-sketch--broadcast"><div class="coding-sketch-shapes"><span class="coding-sketch-shape coding-sketch-shape--input">[n,1,d]</span><span class="coding-sketch-arrow">&times;</span><span class="coding-sketch-shape coding-sketch-shape--state">[1,k,d]</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-shape coding-sketch-shape--active">[n,k,d]</span></div><p class="coding-sketch-note">singleton axes expand; the feature axis remains available for reduction</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Reshape</span><strong>points [n,1,d]</strong><small>Leave a singleton axis for the centers.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Align</span><strong>centers [1,k,d]</strong><small>Leave a singleton axis for the points.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Broadcast</span><strong>difference [n,k,d]</strong><small>Pair every point with every center by shape.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Reduce</span><strong>sum over d</strong><small>Collapse feature coordinates into squared distances.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The final two axes identify one point-center pair and its feature coordinates.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Broadcasting is a shape construction. The singleton axes make the full pair grid visible before subtraction, then the feature axis is the only one reduced. For this problem, hold onto the concrete trace: points [n,1,d] and centers [1,k,d] broadcast to [n,k,d].</figcaption></figure>

**Pattern:** Broadcasting.

**Simple idea:** Change points from shape `(n, d)` to `(n, 1, d)` and centers from `(k, d)`
to `(1, k, d)`. Their difference has shape `(n, k, d)`. Sum over the last axis.

```python
import numpy as np

def pairwise_squared_distances(
   points: np.ndarray, centers: np.ndarray
) -> np.ndarray:
   differences = points[:, None, :] - centers[None, :, :]
   return np.sum(differences * differences, axis=-1)
```

**Cost:** $O(nkd)$ time and $O(nkd)$ temporary space.
