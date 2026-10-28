---
title: "Find Minimum in Rotated Sorted Array"
description: "Retain a minimum inside an inclusive candidate interval, keep a possible midpoint answer, and handle duplicate ambiguity without false logarithmic guarantees."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Foundation"
priority: "Core"
aliases: []
prerequisites: []
---

Find the smallest value in a distinct sorted array whose starting position may have been rotated.

## Problem

**Input.** A nonempty integer list `nums` obtained by rotating a strictly increasing list at some position.

**Output.** Return the smallest value, not its index or the amount of rotation.

**Rules.** Rotation moves a prefix to the end without changing the relative order within either part.
Zero rotation and singleton input are valid. Values must be distinct for this reference.
Do not modify the list. Empty input is outside the contract; duplicates are a changed requirement below.

**Example.** `[3, 4, 5, 1, 2]` returns `1`.
The unrotated list `[1, 2, 3]` also returns `1`.

## Picture

Use `[6,7,8,1,2,3,4,5]`. The minimum's position remains in inclusive `[left,right]`.

Compare `nums[middle]` with `nums[right]` while at least two candidates remain:

| Comparison | What it proves | Update |
| --- | --- | --- |
| `nums[middle] > nums[right]` | The rotation drop lies strictly after `middle` | `left = middle + 1` |
| `nums[middle] < nums[right]` | The suffix through `right` is increasing; `middle` may be minimum | `right = middle` |

The source's `else` covers the second row because distinct values make equality impossible inside the loop.

| Interval | `middle` | Compared values | Next interval |
| --- | --- | --- | --- |
| `[0,7]` | 3 | `1 < 5` | `[0,3]` |
| `[0,3]` | 1 | `7 > 1` | `[2,3]` |
| `[2,3]` | 2 | `8 > 1` | `[3,3]` |

Return `nums[3] = 1`. The first midpoint was already the answer, so replacing `right = middle` with `right = middle - 1` fails.

This reuses [candidate intervals](/questions/binary-search/), but retains a minimum position rather than a target match.
Trace `L`, `m`, and `R` mean `left`, `middle`, and `right`.

<!-- visual:find-minimum-in-rotated-sorted-array-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="find-minimum-in-rotated-sorted-array-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="find-minimum-in-rotated-sorted-array-state-title">Find Minimum in Rotated Sorted Array: Retain the minimum by comparing middle with the current right boundary.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="find-minimum-in-rotated-sorted-array" role="group" tabindex="0" aria-label="Find Minimum in Rotated Sorted Array: Retain the minimum by comparing middle with the current right boundary."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the smallest value in a sorted array that was rotated once.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="retain-middle" role="group" aria-label="Keep the midpoint"><div class="coding-trace-frame-heading"><span>Keep the midpoint</span><strong>left=0, right=7, middle=3. The comparison 1 &lt; 5 permits a midpoint minimum, so set right=3.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:8" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="pointer-left">L</span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="pointer-middle">m</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">6</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-7"><span class="coding-trace-array-pointer" data-motion-key="pointer-right">R</span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">7</small></span></div><div class="coding-trace-meta"><span><b>interval</b>[0, 7]</span><span><b>comparison</b>1 &lt;= 5</span><span><b>decision</b>right = 3</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>pointer-left</dt><dd><strong class="coding-state-after">0</strong></dd></div><div class="coding-state-change"><dt>pointer-middle</dt><dd><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>pointer-right</dt><dd><strong class="coding-state-after">7</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="discard-left-run" hidden role="group" aria-label="Discard the left part"><div class="coding-trace-frame-heading"><span>Discard the left part</span><strong>Now left=0, right=3, middle=1. Since 7 &gt; 1, the drop follows middle, so set left=2.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:8" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="pointer-left">L</span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="pointer-middle">m</span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="pointer-right">R</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">6</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-7"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">7</small></span></div><div class="coding-trace-meta"><span><b>interval</b>[0, 3]</span><span><b>comparison</b>7 &gt; 1</span><span><b>decision</b>left = 2</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>pointer-middle</dt><dd><span class="coding-state-before">3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>pointer-right</dt><dd><span class="coding-state-before">7</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>interval</dt><dd><span class="coding-state-before">[0, 7]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[0, 3]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="discard-eight" hidden role="group" aria-label="Discard midpoint 2"><div class="coding-trace-frame-heading"><span>Discard midpoint 2</span><strong>Now left=2, right=3, middle=2. Since 8 &gt; 1, index 2 cannot be minimum, so set left=3.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:8" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="pointer-middle">L=m</span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="pointer-right">R</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">6</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-7"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">7</small></span></div><div class="coding-trace-meta"><span><b>interval</b>[2, 3]</span><span><b>comparison</b>8 &gt; 1</span><span><b>decision</b>left = 3</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>pointer-middle</dt><dd><span class="coding-state-before">1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2</strong></dd></div><div class="coding-state-change"><dt>pointer-left</dt><dd><span class="coding-state-before">0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>interval</dt><dd><span class="coding-state-before">[0, 3]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[2, 3]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="return-minimum" hidden role="group" aria-label="Return the converged value"><div class="coding-trace-frame-heading"><span>Return the converged value</span><strong>left and right meet at index 3. The invariant leaves nums[3]=1 as the only possible minimum.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:8" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="pointer-middle">min</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">6</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-7"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">7</small></span></div><div class="coding-trace-meta"><span><b>interval</b>[3, 3]</span><span><b>result</b>1</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>pointer-middle</dt><dd><span class="coding-state-before">2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>pointer-right</dt><dd><span class="coding-state-before">3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>interval</dt><dd><span class="coding-state-before">[2, 3]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[3, 3]</strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 4</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Keep the midpoint</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Discard the left part</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Discard midpoint 2</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Return the converged value</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Distinct values form two increasing runs after one rotation, and the task asks for the smallest value rather than a particular target.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">The minimum is always inside inclusive [left, right]. If nums[middle] exceeds nums[right], middle is on the high run; otherwise middle may be the minimum and must be retained.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Pivot searches depend on choosing an endpoint reference and preserving the candidate when equality or ordering does not prove it impossible; this same boundary discipline applies to first-occurrence searches.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> left=0, right=7, middle=3. The comparison 1 &lt; 5 permits a midpoint minimum, so set right=3. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="find-minimum-in-rotated-sorted-array-state"><strong>Check:</strong> Why must the first update use right=middle rather than middle-1?</span></figcaption></figure>

## Code

```python
def find_min_rotated(nums: list[int]) -> int:
   left, right = 0, len(nums) - 1

   while left < right:
      middle = (left + right) // 2
      if nums[middle] > nums[right]:
         left = middle + 1
      else:
         right = middle
   return nums[left]
```

Both branches shrink the interval while preserving the answer. Distinct input needs $O(\log n)$ comparisons and $O(1)$ auxiliary space.

An empty array has no minimum. The reference raises `IndexError` on it.
With duplicates, `[1,1,1,0,1]` incorrectly returns one because equality does not locate the drop.

<!-- learning:core-end -->

## Change the Result or Distinctness

Start from `find_min_rotated`. Two adjustable parts are the returned evidence and the comparison's equality case.

1. **Return the rotation boundary.**

   **New task:** Find the minimum's index rather than its value.

   **Picture and change:** Keep the distinct-array interval and both updates. Replace the final lookup with `return left`.

   **Code:** Complete replacement with an explicit nonempty check:

   ```python
   def rotation_index(nums):
      if not nums:
         raise ValueError("nonempty input required")
      left, right = 0, len(nums) - 1
      while left < right:
         middle = (left + right) // 2
         if nums[middle] > nums[right]:
            left = middle + 1
         else:
            right = middle
      return left
   ```

   **Check:** The pictured input returns three. `[1,2,3]` returns zero because its minimum already starts the array.

   **Use when:** The caller needs a boundary position. Distinctness, $O(\log n)$ comparisons, and $O(1)$ auxiliary space remain unchanged.

2. **Permit duplicate values.**

   **New task:** Find a minimum in a nonempty rotation of a nondecreasing array.

   **Picture and change:** Separate equality from strict comparison. On equality, discard only `right`; `middle` retains an equal witness.

   **Code:** Complete replacement:

   ```python
   def min_with_duplicates(nums):
      if not nums:
         raise ValueError("nonempty input required")
      left, right = 0, len(nums) - 1
      while left < right:
         middle = (left + right) // 2
         if nums[middle] > nums[right]:
            left = middle + 1
         elif nums[middle] < nums[right]:
            right = middle
         else:
            right -= 1
      return nums[left]
   ```

   **Check:** `[1,1,1,0,1]` returns zero. At its first equality, removing only the last one preserves the hidden zero.

   All-equal input can remove only one candidate per iteration. Worst-case work becomes $O(n)$, with $O(1)$ auxiliary space.

   **Use when:** Equality obscures the rotation drop. A surviving minimum is not necessarily a unique rotation boundary.

## Rebuild the Discard

Why is discarding one equal endpoint safe, while discarding its entire half can fail?

**Answer:** If `right` was a minimum, the equal `middle` remains one. Equality says nothing about smaller values between them.
