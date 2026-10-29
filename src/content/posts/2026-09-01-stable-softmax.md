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

Convert finite class scores into probabilities while avoiding overflow from large but valid score values.

## Problem

**Input.** A floating-point NumPy array `logits` and a valid normalization `axis`, defaulting to the last axis.

**Output.** Return softmax probabilities with the same shape, summing to one along each selected-axis slice.

**Rules.** Probabilities are proportional to exponentiated scores within each slice.
Require finite inputs, a nonempty selected axis, and representable score differences.
Preserve the input. Large common score offsets must not cause exponential overflow.

**Example.** Scores `[1000, 1001, 999]` return approximately `[0.244728, 0.665241, 0.090031]`.
Scores `[0, 0]` return `[0.5, 0.5]`, not raw exponentials `[1, 1]`.

## Picture

Subtract the row maximum $1001$: $[-1,0,-2]$.
Exponentials are approximately $[0.367879,1,0.135335]$, with sum $1.503215$.
Divide by that shared sum to obtain $[0.244728,0.665241,0.090031]$.

<!-- visual:stable-softmax-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="stable-softmax-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="stable-softmax-state-title">Subtracting a shared maximum changes exponentials without changing ideal probabilities</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="stable-softmax" role="group" tabindex="0" aria-label="Stable Softmax: Subtract the row maximum, exponentiate nonpositive shifts, and divide by their shared sum."><div class="coding-visual-example"><span>Input and goal</span><strong>Convert logits to probabilities without numeric overflow.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="maximum" role="group" aria-label="Find the maximum logit"><div class="coding-trace-frame-heading"><span>Find the maximum logit</span><strong>For logits [1000,1001,999], the row maximum is 1001. Direct exp(1001) can overflow.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1000</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="row-maximum">max = 1001</span><span class="coding-trace-array-cell">1001</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">999</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>max</dt><dd><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>index 0</dt><dd><strong class="coding-state-after">1000</strong></dd></div><div class="coding-state-change"><dt>index 1</dt><dd><strong class="coding-state-after">1001</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="shift" hidden role="group" aria-label="Shift without changing ratios"><div class="coding-trace-frame-heading"><span>Shift without changing ratios</span><strong>Subtract 1001 from every entry to get [-1,0,-2]. Pairwise gaps are unchanged, and every shifted logit is at most zero.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="row-maximum">largest = 0</span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">-2</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>arithmetic</b>[1000,1001,999] - 1001</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>arithmetic</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[1000,1001,999] - 1001</strong></dd></div><div class="coding-state-change"><dt>index 0</dt><dd><span class="coding-state-before">1000</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">-1</strong></dd></div><div class="coding-state-change"><dt>index 1</dt><dd><span class="coding-state-before">1001</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="exponentiate" hidden role="group" aria-label="Exponentiate safely"><div class="coding-trace-frame-heading"><span>Exponentiate safely</span><strong>Exponentials are [exp(-1),exp(0),exp(-2)] = [0.3679,1.0000,0.1353], whose sum is 1.5032.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.3679</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="largest-exponential">exp(0)</span><span class="coding-trace-array-cell">1.0000</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.1353</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>denominator</b>0.3679 + 1.0000 + 0.1353 = 1.5032</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>exp(0)</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>largest</dt><dd><span class="coding-state-before">1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>denominator</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0.3679 + 1.0000 + 0.1353 = 1.5032</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="normalize" hidden role="group" aria-label="Normalize the row"><div class="coding-trace-frame-heading"><span>Normalize the row</span><strong>Divide each exponential by 1.5032 to get [0.2447,0.6652,0.0900]. The rounded probabilities sum to 0.9999 (approximately 1).</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.2447</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="largest-probability">largest probability</span><span class="coding-trace-array-cell">0.6652</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0.0900</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>normalization</b>exponentials / 1.5032</span><span><b>result</b>[0.2447,0.6652,0.0900]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>largest probability</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>exp(0)</dt><dd><span class="coding-state-before">1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>normalization</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">exponentials / 1.5032</strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 4</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Find the maximum logit</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Shift without changing ratios</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Exponentiate safely</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Normalize the row</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when exponentials normalize scores into probabilities and large positive logits may overflow even though only relative differences should affect the result.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Subtracting one constant from every row element leaves every exponential ratio unchanged; after max shifting, the largest exponent is exp(0)=1 and none can overflow.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Before exponentiating normalized scores, exploit shift invariance to center at the maximum; preserve the reduced dimension so broadcasting back across the row is explicit.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> For logits [1000,1001,999], the row maximum is 1001. Direct exp(1001) can overflow. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="stable-softmax-state"><strong>Check:</strong> Why does subtracting 1001 change the exponentials but not the ideal softmax probabilities?</span></figcaption></figure>

## Code

A common shift cancels from normalized ratios:

$$
\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

Taking $m=\max_j z_j$ makes every exponent argument nonpositive, with at least one exponential equal to one.
Retain dimensions for row-wise broadcasting.

```python
import numpy as np

def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
   shifted = logits - np.max(logits, axis=axis, keepdims=True)
   exponentials = np.exp(shifted)
   return exponentials / np.sum(exponentials, axis=axis, keepdims=True)
```

Work, output and temporaries require $O(N)$ for $N$ entries.

## Change the output

### 1. Forbid selected alternatives

**New task:** forbidden entries must receive zero probability. Every row must contain an allowed entry.

**Picture and change:** validate the mask and allowed values before replacing forbidden scores by negative infinity.

**Code:**

```python
def masked_softmax(logits, allowed, axis=-1):
   logits = np.asarray(logits, dtype=float)
   allowed = np.asarray(allowed, dtype=bool)
   if logits.shape != allowed.shape or not np.all(allowed.any(axis=axis)):
      raise ValueError("invalid visibility")
   if not np.all(np.isfinite(logits[allowed])):
      raise ValueError("nonfinite visible score")
   return stable_softmax(np.where(allowed, logits, -np.inf), axis=axis)
```

**Check:** mask $[1,0,1]$ gives approximately $[0.731059,0,0.268941]$.
A completely masked row raises an error rather than inventing probabilities.

**Use when:** attention or classification explicitly forbids alternatives.

### 2. Return log probabilities

**New task:** preserve useful log values when tiny probabilities underflow.

**Picture and change:** subtract the log normalizer directly, rather than taking logs of computed probabilities.

**Code:**

```python
def stable_log_softmax(logits, axis=-1):
   logits = np.asarray(logits, dtype=float)
   shifted = logits - logits.max(axis=axis, keepdims=True)
   normalizer = np.exp(shifted).sum(axis=axis, keepdims=True)
   return shifted - np.log(normalizer)
```

**Check:** $[0,-1000]$ gives log probabilities $[0,-1000]$ in float64, although $e^{-1000}$ underflows.

**Use when:** losses or sequence scores consume log probabilities.

## Check the axis and dtype

The wrong axis normalizes the wrong groups.
Integer subtraction can overflow before exponentiation. Max shifting does not repair NaNs or a completely masked row.

## Try subtracting the mean

Why does mean subtraction preserve exact ratios but offer less overflow protection?

<details>
<summary>Check your answer</summary>

Any common shift cancels algebraically.
After mean subtraction, the largest exponent argument can remain positive.
Maximum subtraction guarantees nonpositive arguments for valid finite arithmetic.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Numerical limits

The four-decimal trace probabilities sum to $0.9999$ because of rounding.
Test the full-precision result with a tolerance, not exact equality to rounded values.
For a batch, verify every intended row sums to one and compare a column-normalized case separately.

Very large input differences can overflow subtraction even when both original values are finite.
Low precision can erase small differences before this function receives the logits.
Underflow of very small exponentials can still produce zero probabilities.
The log variant avoids first forming those probabilities, but does not remove finite-range arithmetic limits.

Production libraries may fuse softmax or log-softmax with downstream operations.
The NumPy temporary arrays establish no fused-kernel memory or speed claim.

</div>
