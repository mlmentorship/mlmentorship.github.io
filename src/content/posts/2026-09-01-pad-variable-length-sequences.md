---
title: "Pad Variable-Length Sequences"
description: "Put integer sequences into one rectangular array and return a valid-token mask."
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

> Put integer sequences into one rectangular array and return a valid-token mask.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:pad-variable-length-sequences-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="pad-variable-length-sequences-state-title"><p class="visual-kicker">A rectangular batch</p><p class="visual-title" id="pad-variable-length-sequences-state-title">Pad Variable-Length Sequences: Pad values and validity together</p><div class="coding-visual coding-visual--padding" data-coding-visual data-coding-mode="padding" data-coding-slug="pad-variable-length-sequences" role="group" aria-label="Pad Variable-Length Sequences: [3,4] and [9] become tokens [[3,4],[9,0]] plus mask [[1,1],[1,0]]. Every padded token has a false mask, and every real token has a true mask at the same position."><div class="coding-visual-example"><span>Concrete trace</span><strong>[3,4] and [9] become tokens [[3,4],[9,0]] plus mask [[1,1],[1,0]]</strong></div><div class="coding-visual-sketch coding-visual-sketch--padding"><div class="coding-sketch-matrix"><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">token</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">token</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">pad / 0</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">token</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">pad / 0</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">pad / 0</span></div><p class="coding-sketch-note">the mask marks the same cells that contain real tokens</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Measure</span><strong>longest sequence</strong><small>Choose one width for the batch.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Fill</span><strong>pad value</strong><small>Initialize every unused position explicitly.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Copy</span><strong>real tokens</strong><small>Write each sequence into its prefix slice.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Mark</span><strong>boolean mask</strong><small>Keep computation aware of real versus padded cells.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Every padded token has a false mask, and every real token has a true mask at the same position.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Padding creates a rectangle; the mask preserves the original ragged boundary. The two arrays must be drawn as one contract. For this problem, hold onto the concrete trace: [3,4] and [9] become tokens [[3,4],[9,0]] plus mask [[1,1],[1,0]].</figcaption></figure>

**Pattern:** Allocate once, then fill slices.

**Simple idea:** Use the longest sequence as the width. Fill the token array with the pad
value. Copy each sequence into its row and mark the same positions as valid.

```python
from collections.abc import Sequence
import numpy as np

def pad_sequences(
   sequences: Sequence[Sequence[int]], pad_value: int = 0
) -> tuple[np.ndarray, np.ndarray]:
   width = max((len(sequence) for sequence in sequences), default=0)
   tokens = np.full((len(sequences), width), pad_value, dtype=int)
   mask = np.zeros((len(sequences), width), dtype=bool)

   for row, sequence in enumerate(sequences):
      tokens[row, : len(sequence)] = sequence
      mask[row, : len(sequence)] = True
   return tokens, mask
```

**Cost:** $O(batch \times width)$ time and space.
