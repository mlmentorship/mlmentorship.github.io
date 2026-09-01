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
<figure class="learning-figure coding-visual-figure" aria-labelledby="pad-variable-length-sequences-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="pad-variable-length-sequences-state-title">Pad Variable-Length Sequences: Padding creates a rectangle; the boolean mask preserves which cells were real.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="pad-variable-length-sequences" role="group" aria-label="Pad Variable-Length Sequences: Padding creates a rectangle; the boolean mask preserves which cells were real."><div class="coding-visual-example"><span>Input and goal</span><strong>Put integer sequences into one rectangular array and return a valid-token mask.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Start with ragged rows"><div class="coding-trace-frame-heading"><span>Start with ragged rows</span><strong>The sequences have lengths 2 and 1.</strong></div><div class="coding-trace-shapes"><span class="coding-trace-shape is-input">[3,4]</span><span class="coding-trace-link-arrow">&rarr;</span><span class="coding-trace-shape is-output">[9]</span></div><div class="coding-trace-meta"><span><b>action</b>ragged input</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Fill the rectangle"><div class="coding-trace-frame-heading"><span>Fill the rectangle</span><strong>Use the longest width and a pad value for unused cells.</strong></div><div class="coding-trace-grid" style="--trace-cols:2" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell"><span>3</span></span><span class="coding-trace-grid-cell"><span>4</span></span><span class="coding-trace-grid-cell"><span>9</span></span><span class="coding-trace-grid-cell trace-tone-state"><span>0</span><small>pad</small></span></div><div class="coding-trace-meta"><span><b>tensor</b>tokens [2,2]</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Carry the mask"><div class="coding-trace-frame-heading"><span>Carry the mask</span><strong>The same padded position is false in the validity mask.</strong></div><div class="coding-trace-grid" style="--trace-cols:2" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell"><span>1</span></span><span class="coding-trace-grid-cell trace-tone-output"><span>0</span><small>false</small></span></div><div class="coding-trace-meta"><span><b>tensor</b>mask [2,2]</span><span><b>result</b>tokens + mask</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start with ragged rows</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Fill the rectangle</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Carry the mask</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Padding creates a rectangle; the boolean mask preserves which cells were real.</p></div><figcaption><strong>Read it this way:</strong> The sequences have lengths 2 and 1. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
