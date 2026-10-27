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

Pack unequal-length token sequences into one rectangular array while distinguishing real tokens from padding.

## Problem

**Input.** A finite collection `sequences` of integer sequences and an integer `pad_value`, defaulting to zero.

**Output.** Return `(tokens, mask)`: an integer array and an equally shaped Boolean validity array.
Each row represents one sequence; the width is the longest sequence length.

**Rules.** Preserve token order, align sequences on the left, and pad on the right without truncation.
The mask is `True` exactly where a token came from input, even if its value equals `pad_value`.
Preserve inputs. Empty sequences form padded rows; an empty collection returns arrays shaped `(0, 0)`.

**Example.** With sequences `[[3, 4], [], [9]]` and padding zero, tokens are `[[3, 4], [0, 0], [9, 0]]`.
The mask is `[[True, True], [False, False], [True, False]]`.

## Picture

Lengths `2, 0, 1` require a `3 × 2` rectangle.
Initialize tokens to zero and validity to false; copy each row and mark the identical half-open slice.

<!-- visual:pad-variable-length-sequences-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="pad-variable-length-sequences-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="pad-variable-length-sequences-state-title">Pad Variable-Length Sequences: Allocate one batch-width rectangle filled with pad values, then copy each sequence and mark the identical half-open slice as valid.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="pad-variable-length-sequences" role="group" tabindex="0" aria-label="Pad Variable-Length Sequences: Allocate one batch-width rectangle filled with pad values, then copy each sequence and mark the identical half-open slice as valid."><div class="coding-visual-example"><span>Input and goal</span><strong>Put integer sequences into one rectangular array and return a valid-token mask.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="measure-width" role="group" aria-label="Find the rectangular width"><div class="coding-trace-frame-heading"><span>Find the rectangular width</span><strong>Lengths 2, 0 and 1 give width 2. Both output arrays therefore have shape (3,2).</strong></div><div class="coding-trace-grid" style="--trace-cols:2" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="fill-cursor"><span>row 0: 3</span><small>fill cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>row 0: 0</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>row 1: empty</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>row 1: empty</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>row 2: 9</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>row 2: empty</span></span></div><div class="coding-trace-meta"><span><b>input</b>sequences=[[3,0],[],[9]], pad_value=0</span><span><b>lengths</b>[2,0,1]</span><span><b>arithmetic</b>max(2,0,1) = 2</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="allocate" hidden role="group" aria-label="Allocate tokens and mask once"><div class="coding-trace-frame-heading"><span>Allocate tokens and mask once</span><strong>np.full creates three rows of [0,0]; np.zeros creates a matching all-false Boolean mask.</strong></div><div class="coding-trace-grid" style="--trace-cols:2" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="fill-cursor"><span>0 | false</span><small>fill cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>0 | false</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>0 | false</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>0 | false</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>0 | false</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>0 | false</span></span></div><div class="coding-trace-meta"><span><b>input</b>sequences=[[3,0],[],[9]], pad_value=0</span><span><b>shapes</b>tokens (3,2); mask (3,2)</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="fill-row-zero" hidden role="group" aria-label="Fill row 0 through slice [0:2]"><div class="coding-trace-frame-heading"><span>Fill row 0 through slice [0:2]</span><strong>Copy [3,0] into tokens[0,:2] and set mask[0,:2]=True. The input zero is valid, just like the input three.</strong></div><div class="coding-trace-grid" style="--trace-cols:2" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>3 | true</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="fill-cursor"><span>0 | true</span><small>fill cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>0 | false</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>0 | false</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>0 | false</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>0 | false</span></span></div><div class="coding-trace-meta"><span><b>input</b>sequences=[[3,0],[],[9]], pad_value=0</span><span><b>assignment</b>tokens[0,:2]=[3,0]; mask[0,:2]=True</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="keep-empty-row" hidden role="group" aria-label="Process the empty row"><div class="coding-trace-frame-heading"><span>Process the empty row</span><strong>For row 1, len(sequence)=0, so both [1,:0] assignments are empty slices and no cell changes.</strong></div><div class="coding-trace-grid" style="--trace-cols:2" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>3 | true</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>0 | true</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="fill-cursor"><span>0 | false</span><small>fill cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>0 | false</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>0 | false</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>0 | false</span></span></div><div class="coding-trace-meta"><span><b>input</b>sequences=[[3,0],[],[9]], pad_value=0</span><span><b>assignment</b>tokens[1,:0]=[]; mask[1,:0]=True</span><span><b>decision</b>empty slice changes nothing</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="fill-row-two" hidden role="group" aria-label="Fill row 2 through slice [0:1]"><div class="coding-trace-frame-heading"><span>Fill row 2 through slice [0:1]</span><strong>Copy [9] into the first token cell and mark only that position true; the second cell remains padding.</strong></div><div class="coding-trace-grid" style="--trace-cols:2" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>3 | true</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>0 | true</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>0 | false</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>0 | false</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="fill-cursor"><span>9 | true</span><small>fill cursor</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-1"><span>0 | false</span></span></div><div class="coding-trace-meta"><span><b>input</b>sequences=[[3,0],[],[9]], pad_value=0</span><span><b>assignment</b>tokens[2,:1]=[9]; mask[2,:1]=True</span></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="return" hidden role="group" aria-label="Return aligned token and mask rectangles"><div class="coding-trace-frame-heading"><span>Return aligned token and mask rectangles</span><strong>Compare row 0, column 1 with row 1, column 1: both tokens are zero, but only the first came from input. Validity depends on sequence length, not token value.</strong></div><div class="coding-trace-grid" style="--trace-cols:2" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>3 | true</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>0 | true</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>0 | false</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>0 | false</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-2-0"><span>9 | true</span></span><span class="coding-trace-grid-cell trace-tone-output" data-motion-key="fill-cursor"><span>0 | false</span><small>fill cursor</small></span></div><div class="coding-trace-meta"><span><b>input</b>sequences=[[3,0],[],[9]], pad_value=0</span><span><b>tokens</b>[[3,0],[0,0],[9,0]]</span><span><b>mask</b>[[T,T],[F,F],[T,F]]</span><span><b>result</b>tokens shape (3,2), mask shape (3,2)</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 6</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Find the rectangular width</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Allocate tokens and mask once</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Fill row 0 through slice [0:2]</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Process the empty row</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Fill row 2 through slice [0:1]</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Return aligned token and mask rectangles</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use padding plus a mask when variable-length sequences must enter rectangular tensor operations without treating synthetic pad cells as data.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">After processing row r, tokens[r,:len(sequence)] equals the original sequence and mask is true on exactly that slice; every remaining cell retains the pad value and false mask.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">For ragged-to-dense conversion, derive one shared extent, initialize safe defaults, and update data and validity metadata with the exact same slices.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Lengths 2, 0 and 1 give width 2. Both output arrays therefore have shape (3,2). Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

The result is `[[3, 4], [0, 0], [9, 0]]`, with mask `[[True, True], [False, False], [True, False]]`.
After each row, only its copied positions are valid; untouched cells remain padding.

## Code

Keep token and mask slices aligned, including the empty slice `:0`.

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

For batch size $B$ and width $W$, time is $O(B+BW)$ and output storage is $O(BW)$.

## Change alignment or width

### 1. Left-pad

**New task:** align sequence endings at the right edge.

**Picture and change:** start at `width - length`; move data and validity together.

```python
def left_pad_sequences(sequences, pad_value=0):
   width = max(map(len, sequences), default=0)
   tokens = np.full((len(sequences), width), pad_value, dtype=int)
   mask = np.zeros((len(sequences), width), dtype=bool)
   for row, sequence in enumerate(sequences):
      start = width - len(sequence)
      tokens[row, start:width] = sequence
      mask[row, start:width] = True
   return tokens, mask
```

**Check:** `[9]` becomes `[0, 9]`, with mask `[False, True]`.
An empty row remains invalid throughout; avoid the misleading slice `-0:`.

**Use when:** the consumer requires right-aligned endings.

### 2. Fix width and report truncation

**New task:** retain each sequence's prefix within a requested width.

**Picture and change:** copy at most `width` positions and report information loss per row.

```python
def pad_fixed_width(sequences, width, pad_value=0):
   if width < 0:
      raise ValueError("negative width")
   tokens = np.full((len(sequences), width), pad_value, dtype=int)
   mask = np.zeros((len(sequences), width), dtype=bool)
   truncated = []
   for row, sequence in enumerate(sequences):
      length = min(width, len(sequence))
      tokens[row, :length] = sequence[:length]
      mask[row, :length] = True
      truncated.append(len(sequence) > width)
   return tokens, mask, truncated
```

**Check:** `[[3, 4, 5], []]` at width `2` reports `[True, False]`.
Width zero still reports truncation of nonempty inputs.

**Use when:** a hard width is part of the caller contract.

## Failure and transfer

Why cannot `tokens != pad_value` recover validity?

<details>
<summary>Check your answer</summary>

Input `[[0], []]` produces two zero cells, but masks `[True]` and `[False]`.
Only original lengths identify which zero was copied.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Consumer contracts

No input rows produce shape `(0, 0)`; two empty rows produce `(2, 0)`.
Do not collapse batch cardinality because width is zero.

Inputs must be nested integer sequences within NumPy's machine-integer range.
This reference casts values; it does not validate a general token schema.
Fixed width must be a nonnegative integer.

Some attention APIs use `True` for blocked positions, unlike this validity mask.
Coordinate mask polarity, position IDs and padding conventions with the consumer.
Changing padding side alone does not settle those contracts.

</div>
