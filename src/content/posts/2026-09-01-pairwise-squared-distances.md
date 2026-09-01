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
<figure class="learning-figure coding-visual-figure" aria-labelledby="pairwise-squared-distances-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="pairwise-squared-distances-state-title">Pairwise Squared Distances: Singleton axes create every point-center pair before reducing feature coordinates.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="pairwise-squared-distances" role="group" aria-label="Pairwise Squared Distances: Singleton axes create every point-center pair before reducing feature coordinates."><div class="coding-visual-example"><span>Input and goal</span><strong>Compute the squared distance from every point to every center without Python loops.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Add singleton axes"><div class="coding-trace-frame-heading"><span>Add singleton axes</span><strong>Points [n,d] become [n,1,d]; centers become [1,k,d].</strong></div><div class="coding-trace-shapes"><span class="coding-trace-shape is-input">points [n,1,d]</span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-shape is-output">centers [1,k,d]</span></div><div class="coding-trace-meta"><span><b>action</b>align singleton axes</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Broadcast pairs"><div class="coding-trace-frame-heading"><span>Broadcast pairs</span><strong>The difference tensor has one row for every point-center pair.</strong></div><div class="coding-trace-shapes"><span class="coding-trace-shape is-input">points [n,1,d]</span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-shape is-state">centers [1,k,d]</span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-shape is-output">difference [n,k,d]</span></div><div class="coding-trace-meta"><span><b>action</b>broadcast</span><span><b>focus</b>difference</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Reduce features"><div class="coding-trace-frame-heading"><span>Reduce features</span><strong>Summing squared differences over d yields [n,k] distances.</strong></div><div class="coding-trace-shapes"><span class="coding-trace-shape is-input">difference [n,k,d]</span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-shape is-state">sum over d</span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-shape is-output">distances [n,k]</span></div><div class="coding-trace-meta"><span><b>result</b>[n,k]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Add singleton axes</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Broadcast pairs</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Reduce features</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Singleton axes create every point-center pair before reducing feature coordinates.</p></div><figcaption><strong>Read it this way:</strong> Points [n,d] become [n,1,d]; centers become [1,k,d]. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
