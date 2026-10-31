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

Compute a table containing the squared Euclidean distance from every point to every center.

## Problem

**Input.** Float64 arrays `points` with shape `(n, d)` and `centers` with shape `(k, d)`.

**Output.** Return an `(n, k)` array whose entry `[i, j]` is the sum of squared coordinate differences.
Do not take a square root: these are squared distances, not Euclidean distances themselves.

**Rules.** Feature counts must match. Repeated points and empty axes are allowed; preserve both inputs.
Differences, squares, and sums must remain representable in float64.
Integer inputs are outside this numeric contract because they may overflow before reduction.

**Example.** Points `[[0, 0], [1, 2]]` and centers `[[0, 0], [1, 0]]` return `[[0, 1], [5, 4]]`.
The lower-right entry is `0 * 0 + 2 * 2 = 4`.

## Picture

For points `(0,0)`, `(1,2)` and centers `(1,0)`, `(2,2)`, the output is `[[1,8],[4,1]]`.
Inserted axes form `(n,1,d)` and `(1,k,d)` views.
Their difference has shape `(n,k,d)`; summing only the feature axis retains both ownership axes.

<!-- visual:pairwise-squared-distances-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="pairwise-squared-distances-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="pairwise-squared-distances-state-title">Sum squared coordinate differences; do not take a square root</p><div class="visual-argument" data-visual-argument="pairwise-squared-distances-state" data-argument-kind="ledger" data-argument-composition="arithmetic"><svg viewBox="0 0 360 376" role="img" aria-label="Worked example: combine row norms with the cross-product, without a 3D difference tensor. For x=(1,2) and y=(3,2), squared distance is 5+13-2x7=4. The broadcasted identity produces that same scalar."><title>Worked example: combine row norms with the cross-product, without a 3D difference tensor</title><desc>For x=(1,2) and y=(3,2), squared distance is 5+13-2x7=4. The broadcasted identity produces that same scalar.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Squared row norms</tspan></text><g data-proof-token="5" data-proof-row="0" data-proof-position="0" data-proof-box="20,60,154,48"><text x="97" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="97" dy="0">5</tspan></text></g><g data-proof-token="13" data-proof-row="0" data-proof-position="1" data-proof-box="180,60,154,48"><text x="257" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="257" dy="0">13</tspan></text></g><text x="20" y="144" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Dot product</tspan></text><g data-proof-token="1x3" data-proof-row="1" data-proof-position="0" data-proof-box="20,176,58,48"><text x="49" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="49" dy="0">1x3</tspan></text></g><g data-proof-token="+" data-proof-row="1" data-proof-position="1" data-proof-box="84,176,58,48"><text x="113" y="203" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="113" dy="0">+</tspan></text></g><g data-proof-token="2x2" data-proof-row="1" data-proof-position="2" data-proof-box="148,176,58,48"><text x="177" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="177" dy="0">2x2</tspan></text></g><g data-proof-token="=" data-proof-row="1" data-proof-position="3" data-proof-box="212,176,58,48"><text x="241" y="203" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="241" dy="0">=</tspan></text></g><g data-proof-token="7" data-proof-row="1" data-proof-position="4" data-proof-box="276,176,58,48"><text x="305" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="305" dy="0">7</tspan></text></g><text x="20" y="260" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Squared-distance identity</tspan></text><g data-proof-token="5" data-proof-row="2" data-proof-position="0" data-proof-box="20,292,58,48"><text x="49" y="319" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="49" dy="0">5</tspan></text></g><g data-proof-token="+13" data-proof-row="2" data-proof-position="1" data-proof-box="84,292,58,48"><text x="113" y="319" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="113" dy="0">+13</tspan></text></g><g data-proof-token="-2x7" data-proof-row="2" data-proof-position="2" data-proof-box="148,292,58,48"><text x="177" y="319" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="177" dy="0">-2x7</tspan></text></g><g data-proof-token="=" data-proof-row="2" data-proof-position="3" data-proof-box="212,292,58,48"><text x="241" y="319" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="241" dy="0">=</tspan></text></g><g data-proof-token="4" data-proof-row="2" data-proof-position="4" data-proof-box="276,292,58,48"><text x="305" y="319" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="305" dy="0">4</tspan></text><path d="M280 337L330 337" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path></g></svg></div><details class="visual-derivation" open><summary>Execution trace</summary><div class="visual-derivation-content"><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="pairwise-squared-distances" role="group" tabindex="0" aria-label="Pairwise Squared Distances: Broadcast every point-center pair across a feature axis, square coordinate differences, then reduce that axis."><div class="coding-visual-example"><span>Input and goal</span><strong>Compute the squared distance from every point to every center without Python loops.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="inputs" role="group" aria-label="Lay out points and centers"><div class="coding-trace-frame-heading"><span>Lay out points and centers</span><strong>Use points P0=(0,0), P1=(1,2) and centers C0=(1,0), C1=(2,2). The output must contain all four P-C pairs.</strong></div><div class="coding-trace-grid" style="--trace-cols:2" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>P0 (0,0)</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>P0 (0,0)</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>P1 (1,2)</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>P1 (1,2)</span></span></div><div class="coding-trace-meta"><span><b>columns</b>C0 (1,0), C1 (2,2)</span><span><b>shape</b>pair grid [n=2,k=2]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>columns</dt><dd><strong class="coding-state-after">C0 (1,0), C1 (2,2)</strong></dd></div><div class="coding-state-change"><dt>shape</dt><dd><strong class="coding-state-after">pair grid [n=2,k=2]</strong></dd></div><div class="coding-state-change"><dt>C / row 0</dt><dd><strong class="coding-state-after">P0 (0,0)</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="differences" hidden role="group" aria-label="Broadcast coordinate differences"><div class="coding-trace-frame-heading"><span>Broadcast coordinate differences</span><strong>P[:,None,:] - C[None,:,:] produces one 2-vector per grid cell: P0-C0=(-1,0), P0-C1=(-2,-2), P1-C0=(0,2), P1-C1=(-1,0).</strong></div><div class="coding-trace-grid" style="--trace-cols:2" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="difference-cell"><span>(-1,0)</span><small>pair P0,C0</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>(-2,-2)</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>(0,2)</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>(-1,0)</span></span></div><div class="coding-trace-meta"><span><b>shape</b>differences [2,2,2]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>pair P0,C0</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(0, 0)</strong></dd></div><div class="coding-state-change"><dt>shape</dt><dd><span class="coding-state-before">pair grid [n=2,k=2]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">differences [2,2,2]</strong></dd></div><div class="coding-state-change"><dt>columns</dt><dd><span class="coding-state-before">C0 (1,0), C1 (2,2)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="squares" hidden role="group" aria-label="Square each feature"><div class="coding-trace-frame-heading"><span>Square each feature</span><strong>Elementwise multiplication differences*differences gives (1,0), (4,4), (0,4), and (1,0); signs disappear but the feature axis remains.</strong></div><div class="coding-trace-grid" style="--trace-cols:2" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>(1,0)</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="squared-cell"><span>(4,4)</span><small>4,4</small></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>(0,4)</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-1"><span>(1,0)</span></span></div><div class="coding-trace-meta"><span><b>operation</b>elementwise square</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>4,4</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(0, 1)</strong></dd></div><div class="coding-state-change"><dt>pair P0,C0</dt><dd><span class="coding-state-before">(0, 0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>operation</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">elementwise square</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="reduce" hidden role="group" aria-label="Reduce the feature axis"><div class="coding-trace-frame-heading"><span>Reduce the feature axis</span><strong>Sum each pair vector over d: 1+0=1, 4+4=8, 0+4=4, and 1+0=1. Rows are points and columns are centers.</strong></div><div class="coding-trace-grid" style="--trace-cols:2" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>1</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-0-1"><span>8</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>4</span></span><span class="coding-trace-grid-cell trace-tone-output" data-motion-key="distance-cell"><span>1</span><small>distance</small></span></div><div class="coding-trace-meta"><span><b>arithmetic</b>sum over final axis d</span><span><b>result</b>[[1,8],[4,1]]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>distance</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 1)</strong></dd></div><div class="coding-state-change"><dt>4,4</dt><dd><span class="coding-state-before">(0, 1)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>arithmetic</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">sum over final axis d</strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 4</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Lay out points and centers</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Broadcast coordinate differences</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Square each feature</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Reduce the feature axis</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when every item in one batch must pair with every item in another batch and the per-pair calculation is identical across a shared feature dimension.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Cell [i,j,:] always belongs to point i and center j; broadcasting changes alignment, not values, and reducing only the final feature axis preserves the pair grid.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">For all-pairs tensor operations, insert singleton axes where each operand should repeat, verify the broadcasted axis meaning, and reduce only the intended feature axes.</dd></div></dl></div><div class="visual-derivation-caption">Use points P0=(0,0), P1=(1,2) and centers C0=(1,0), C1=(2,2). The output must contain all four P-C pairs. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="pairwise-squared-distances-state"><strong>Check:</strong> Which axis disappears in the reduction? Why is the P0-C1 result eight rather than its square root?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="pairwise-squared-distances-state">For x=(1,2) and y=(3,2), squared distance is 5+13-2x7=4. The broadcasted identity produces that same scalar.</span></figcaption></figure>

Without singleton axes, subtraction may fail or compare corresponding rows instead of every point-center pair.

## Code

```python
import numpy as np

def pairwise_squared_distances(
   points: np.ndarray, centers: np.ndarray
) -> np.ndarray:
   differences = points[:, None, :] - centers[None, :, :]
   return np.sum(differences * differences, axis=-1)
```

**Cost:** $O(nkd)$ time and $O(nkd)$ temporary space.

The retained output additionally occupies $O(nk)$ space, including when the feature axis is empty.

## Change the Requirement

### Bound Pairwise Temporary Storage

**New task:** Emit point-row blocks rather than retaining every pairwise difference.

**Picture and change:** Keep center ownership fixed; restrict the active point axis to `block_size`.

**Code:** The reference still computes each complete block.

```python
def squared_distance_blocks(points, centers, block_size: int):
   if block_size <= 0:
      raise ValueError("block_size must be positive")
   for start in range(0, len(points), block_size):
      yield start, pairwise_squared_distances(points[start:start + block_size], centers)
```

**Check:** Two point rows with block size one yield offsets `0,1` and rows `[1,8]`, `[4,1]`.

**Use when:** The consumer processes blocks; retaining them all still costs $O(nk)$ output storage.

### Avoid the Three-Axis Difference

**New task:** Compute distances through norms and matrix multiplication.

**Picture and change:** Expand $\lVert p-c\rVert^2=\lVert p\rVert^2+\lVert c\rVert^2-2p^\top c$.

**Code:** Retain float64 shapes; require representable norm and product intermediates.

```python
def squared_distances_gram(points, centers):
   point_norms = np.einsum("nd,nd->n", points, points)[:, None]
   center_norms = np.einsum("kd,kd->k", centers, centers)[None, :]
   result = point_norms + center_norms - 2 * (points @ centers.T)
   return np.maximum(result, 0.0)
```

**Check:** The small example still gives `[[1,8],[4,1]]`.

**Use when:** Explicit array storage must scale as $O(nk+n+k)$, excluding backend workspace.
Cancellation can destroy small distances; clipping negatives does not restore lost accuracy.

<!-- learning:core-end -->
<div data-learning-extension>

## Reconstruct

Why can the Gram expression return zero for distinct points?

<details>
<summary>Reasoned answer</summary>

In float64, point `[[100000001.0]]` and center `[[100000000.0]]` have direct squared distance `1`.
The Gram form subtracts nearly equal values around $10^{16}$ and returns `0` for this example.
Subtracting nearby coordinates before squaring avoids that particular cancellation, but neither formulation prevents every overflow or rounding problem.
Check representative magnitudes rather than treating algebraic equivalence as identical floating-point behavior.

The blocked version needs $O(\min(n,block\_size)kd)$ pairwise temporary storage and corresponding block-output storage.
It does not reduce total $O(nkd)$ arithmetic or automatically make the consumer streaming.

</details>

</div>
