---
title: "Search in Rotated Sorted Array"
description: "Identify one sorted half, use exact endpoint bounds to retain a target, and handle duplicate ambiguity without promising logarithmic progress."
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

Find a target's original position in a distinct sorted array whose starting position may have been rotated.

## Problem

**Input.** An integer list `nums` obtained by rotating a strictly increasing list, and an integer `target`.

**Output.** Return the target's zero-based index in the supplied rotated list, or `-1` if absent.

**Rules.** Values are distinct, and zero rotation is valid.
Do not sort or otherwise modify the input; returned positions refer to its current order.
Empty input returns `-1`. Supporting duplicate values is a changed contract below.

**Example.** For `[4, 5, 6, 7, 0, 1, 2]` and target `0`, return `4`.
Target `3` in that list returns `-1`.

## Picture

Reuse the [inclusive candidate interval](/questions/binary-search/): if the target exists, its index remains in `[left,right]`.

First compare `nums[middle]` with `target`. Equality returns immediately, so later tests can exclude the midpoint.

At most one rotation drop exists. If `nums[left] <= nums[middle]`, the left half is sorted; otherwise, the right half is sorted.

| Known sorted half | Target fits when | Retain if it fits |
| --- | --- | --- |
| `left..middle` | `nums[left] <= target < nums[middle]` | `[left,middle-1]` |
| `middle..right` | `nums[middle] < target <= nums[right]` | `[middle+1,right]` |

If the target does not fit the sorted half, retain the other half. Neither update can discard a possible occurrence.

For `[4,5,6,7,0,1,2]` and target zero:

| Candidate interval | Probe | Decision |
| --- | --- | --- |
| `[0,6]` | Index 3 has 7 | Sorted values 4 through 7 exclude zero; retain `[4,6]` |
| `[4,6]` | Index 5 has 1 | Sorted values 0 through 1 include zero; retain `[4,4]` |
| `[4,4]` | Index 4 has 0 | Return 4 |

Trace `L`, `m`, and `R` mean `left`, `middle`, and `right`.

<!-- visual:search-in-rotated-sorted-array-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="search-in-rotated-sorted-array-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="search-in-rotated-sorted-array-state-title">Search in Rotated Sorted Array: Keep the interval containing target 0 by identifying one sorted half at each probe.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="search-in-rotated-sorted-array" role="group" tabindex="0" aria-label="Search in Rotated Sorted Array: Keep the interval containing target 0 by identifying one sorted half at each probe."><div class="coding-visual-example"><span>Input and goal</span><strong>Find a target in a sorted array that was rotated once.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="discard-sorted-left" role="group" aria-label="Reject the sorted half"><div class="coding-trace-frame-heading"><span>Reject the sorted half</span><strong>left=0, right=6, middle=3. The left half is sorted, but zero is outside [4,7), so set left=4.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:7" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="pointer-left">L</span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="pointer-middle">m</span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-pointer" data-motion-key="pointer-right">R</span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>target</b>0</span><span><b>sorted</b>indices 0..3</span><span><b>decision</b>0 not in [4, 7): left = 4</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>pointer-left</dt><dd><strong class="coding-state-after">0</strong></dd></div><div class="coding-state-change"><dt>pointer-middle</dt><dd><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>pointer-right</dt><dd><strong class="coding-state-after">6</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="keep-sorted-left" hidden role="group" aria-label="Keep the sorted half"><div class="coding-trace-frame-heading"><span>Keep the sorted half</span><strong>Now left=4, right=6, middle=5. The target fits [0,1), so set right=4 and keep index 4.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:7" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-pointer" data-motion-key="pointer-left">L</span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-pointer" data-motion-key="pointer-middle">m</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-pointer" data-motion-key="pointer-right">R</span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>target</b>0</span><span><b>sorted</b>indices 4..5</span><span><b>decision</b>0 in [0, 1): right = 4</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>pointer-left</dt><dd><span class="coding-state-before">0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">4</strong></dd></div><div class="coding-state-change"><dt>pointer-middle</dt><dd><span class="coding-state-before">3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">5</strong></dd></div><div class="coding-state-change"><dt>sorted</dt><dd><span class="coding-state-before">indices 0..3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">indices 4..5</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="return-target" hidden role="group" aria-label="Return the target"><div class="coding-trace-frame-heading"><span>Return the target</span><strong>left=right=middle=4 and nums[4]=0, so the search returns index 4.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:7" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-pointer" data-motion-key="pointer-middle">found</span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>target</b>0</span><span><b>comparison</b>0 = 0</span><span><b>result</b>index 4</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>pointer-middle</dt><dd><span class="coding-state-before">5</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">4</strong></dd></div><div class="coding-state-change"><dt>pointer-left</dt><dd><span class="coding-state-before">4</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>pointer-right</dt><dd><span class="coding-state-before">6</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Reject the sorted half</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Keep the sorted half</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the target</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">The array was sorted and rotated once with distinct values, so although the whole interval may cross the pivot, at least one side of middle remains sorted.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">If target 0 exists, it remains in inclusive [left, right]. A half is discarded only after its sorted value range proves whether target can occur there.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">When a monotone ordering is disrupted once, find a locally ordered region and use its endpoint values to make the same safe-elimination argument as ordinary binary search.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> left=0, right=6, middle=3. The left half is sorted, but zero is outside [4,7), so set left=4. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="search-in-rotated-sorted-array-state"><strong>Check:</strong> Why may indices 0 through 3 be discarded even though they are the sorted half?</span></figcaption></figure>

## Code

```python
def search_rotated(nums: list[int], target: int) -> int:
   left, right = 0, len(nums) - 1

   while left <= right:
      middle = (left + right) // 2
      if nums[middle] == target:
         return middle

      if nums[left] <= nums[middle]:
         if nums[left] <= target < nums[middle]:
            right = middle - 1
         else:
            left = middle + 1
      elif nums[middle] < target <= nums[right]:
         left = middle + 1
      else:
         right = middle - 1

   return -1
```

Each unsuccessful probe removes its midpoint and roughly half the candidates. Distinct input needs $O(\log n)$ comparisons and $O(1)$ auxiliary space.

With duplicates, `[1,0,1,1,1]` and target zero incorrectly return `-1`.
Equal left/middle values no longer prove the left half is sorted.

Unlike [minimum search](/questions/find-minimum-in-rotated-sorted-array/), this loop checks midpoint equality before discarding it.

<!-- learning:core-end -->

## Change Ambiguity or Output

Start from `search_rotated`. Two adjustable parts are the ordering evidence and the result representation.

1. **Permit duplicates.**

   **New task:** Return an occurrence in a rotated nondecreasing array, or `-1`.

   **Picture and change:** Keep the first equality check. If left, middle, and right values agree, the endpoint values are also known nonmatches.

   Discard only those endpoints; their equality does not locate the rotation drop.

   **Code:** Complete replacement:

   ```python
   def search_rotated_duplicates(nums, target):
      left, right = 0, len(nums) - 1
      while left <= right:
         middle = (left + right) // 2
         if nums[middle] == target:
            return middle
         if nums[left] == nums[middle] == nums[right]:
            left += 1
            right -= 1
         elif nums[left] <= nums[middle]:
            if nums[left] <= target < nums[middle]:
               right = middle - 1
            else:
               left = middle + 1
         elif nums[middle] < target <= nums[right]:
            left = middle + 1
         else:
            right = middle - 1
      return -1
   ```

   **Check:** `[1,0,1,1,1]`, target zero, returns index one. `[1,1,1]`, target two, returns `-1`.

   No first-occurrence guarantee is added. All-equal absent input can require $O(n)$ work; auxiliary space remains $O(1)$.

   **Use when:** Duplicate values remove the evidence needed to discard a complete half.

2. **Return membership only.**

   **New task:** Report a Boolean rather than an index, under the original distinctness contract.

   **Picture and change:** Keep every search decision. Convert the returned missing sentinel explicitly.

   **Code:** Place this complete wrapper after the retained `search_rotated` definition.

   ```python
   def contains_rotated(nums, target):
      return search_rotated(nums, target) != -1
   ```

   **Check:** `[1,2,3]`, target one, returns `True` despite its zero index. Empty input returns `False`.

   Using `bool(index)` would treat index zero as false and missing `-1` as true.

   **Use when:** Only the output representation changes. The reference's assumptions and complexity remain unchanged.

## Rebuild the Evidence

Why does the duplicate variant lose logarithmic worst-case progress?

**Answer:** Equal boundary and midpoint values may justify removing only two positions, not a half. Correctness survives; halving does not.
