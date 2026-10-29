---
title: "Keep the best k values with the weakest retained value at the root"
description: "Evict the smallest retained occurrence on overflow, then change distinctness or report streaming thresholds."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Intermediate"
priority: "Core"
aliases: []
prerequisites: []
---

Return the value at a requested descending rank, counting repeated array entries separately.

## Problem

**Input.** An integer list `nums` and a one-based integer rank `k` satisfying `1 <= k <= len(nums)`.

**Output.** Return the value in position `k` when all occurrences are ordered from largest to smallest.

**Rules.** Equal values occupy separate ranks. Return a value, not its original index or the `k` largest values.
Preserve the input. The reference raises `ValueError` for invalid ranks, including empty input.

**Example.** For `[3, 2, 1, 5, 6, 4]` and `k = 2`, return `5`.
For `[5, 5, 4]` and `k = 2`, return `5`, not the second distinct value 4.

## Picture

Protect `k` places, with the weakest protected value easiest to remove.
After each input, the retained multiset contains the largest `min(k, processed)` occurrences seen.
Push the new occurrence; if there are too many, evict the minimum.
Removing the smallest of `k+1` candidates leaves the best `k`.

| After input | Retained heap values | Root threshold |
| --- | --- | --- |
| `3,2` | `[2,3]` | `2` |
| `3,2,1,5` | `[3,5]` | `3` |
| `3,2,1,5,6,4` | `[5,6]` | `5` |

<!-- visual:kth-largest-element-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="kth-largest-element-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="kth-largest-element-state-title">The smallest of the largest k values is the kth largest</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="kth-largest-element" role="group" tabindex="0" aria-label="Kth Largest Element: A size-k min-heap retains the k largest values seen, so its root is the kth largest."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the `k`th largest value in an unsorted array.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="push-3" role="group" aria-label="Seed the candidate heap"><div class="coding-trace-frame-heading"><span>Seed the candidate heap</span><strong>For nums = [3,2,1,5,6,4] and k = 2, push 3. The heap has room, so nothing is removed.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root is-current" data-motion-key="heap-value-3-0">3</span></li></ul></section></div><div class="coding-trace-meta"><span><b>input</b>[3,2,1,5,6,4]</span><span><b>current</b>3</span><span><b>processed</b>1 of 6</span><span><b>size</b>1 &lt;= k</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>input</dt><dd><strong class="coding-state-after">[3,2,1,5,6,4]</strong></dd></div><div class="coding-state-change"><dt>current</dt><dd><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>processed</dt><dd><strong class="coding-state-after">1 of 6</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="push-2" hidden role="group" aria-label="Fill the second slot"><div class="coding-trace-frame-heading"><span>Fill the second slot</span><strong>Push 2. The min-heap orders the two protected candidates as [2,3], with the weaker candidate 2 at the root.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root is-current" data-motion-key="heap-value-2-0">2</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-3-0">3</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>current</b>2</span><span><b>processed</b>2 of 6</span><span><b>size</b>2 = k</span><span><b>action</b>keep both</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>current</dt><dd><span class="coding-state-before">3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2</strong></dd></div><div class="coding-state-change"><dt>processed</dt><dd><span class="coding-state-before">1 of 6</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2 of 6</strong></dd></div><div class="coding-state-change"><dt>size</dt><dd><span class="coding-state-before">1 &lt;= k</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2 = k</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="reject-1" hidden role="group" aria-label="Discard an undersized candidate"><div class="coding-trace-frame-heading"><span>Discard an undersized candidate</span><strong>Push 1 to make [1,3,2]. Size 3 exceeds k, so pop root 1; [2,3] still holds the largest two of [3,2,1].</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-2-0">2</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-3-0">3</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>current</b>1</span><span><b>processed</b>3 of 6</span><span><b>overflow</b>[1,3,2]</span><span><b>action</b>pop 1</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>current</dt><dd><span class="coding-state-before">2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>processed</dt><dd><span class="coding-state-before">2 of 6</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3 of 6</strong></dd></div><div class="coding-state-change"><dt>overflow</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[1,3,2]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="accept-5" hidden role="group" aria-label="Raise the protected floor"><div class="coding-trace-frame-heading"><span>Raise the protected floor</span><strong>Push 5 to make [2,3,5], then pop 2. The retained candidates become [3,5], so the root threshold rises to 3.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-3-0">3</span><ul><li><span class="coding-trace-heap-value is-current" data-motion-key="heap-value-5-0">5</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>current</b>5</span><span><b>processed</b>4 of 6</span><span><b>overflow</b>[2,3,5]</span><span><b>action</b>pop 2</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>current</dt><dd><span class="coding-state-before">1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">5</strong></dd></div><div class="coding-state-change"><dt>processed</dt><dd><span class="coding-state-before">3 of 6</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">4 of 6</strong></dd></div><div class="coding-state-change"><dt>overflow</dt><dd><span class="coding-state-before">[1,3,2]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[2,3,5]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="accept-6" hidden role="group" aria-label="Keep the next large value"><div class="coding-trace-frame-heading"><span>Keep the next large value</span><strong>Push 6 to make [3,5,6], then pop 3. The heap [5,6] is exactly the largest two values seen so far.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-5-0">5</span><ul><li><span class="coding-trace-heap-value is-current" data-motion-key="heap-value-6-0">6</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>current</b>6</span><span><b>processed</b>5 of 6</span><span><b>overflow</b>[3,5,6]</span><span><b>action</b>pop 3</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>current</dt><dd><span class="coding-state-before">5</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">6</strong></dd></div><div class="coding-state-change"><dt>processed</dt><dd><span class="coding-state-before">4 of 6</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">5 of 6</strong></dd></div><div class="coding-state-change"><dt>overflow</dt><dd><span class="coding-state-before">[2,3,5]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[3,5,6]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="reject-4" hidden role="group" aria-label="Reject the last weaker value"><div class="coding-trace-frame-heading"><span>Reject the last weaker value</span><strong>Push 4 to make [4,6,5], then pop 4. The protected set remains [5,6] after all six inputs.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-5-0">5</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-6-0">6</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>current</b>4</span><span><b>processed</b>6 of 6</span><span><b>overflow</b>[4,6,5]</span><span><b>action</b>pop 4</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>current</dt><dd><span class="coding-state-before">6</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">4</strong></dd></div><div class="coding-state-change"><dt>processed</dt><dd><span class="coding-state-before">5 of 6</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">6 of 6</strong></dd></div><div class="coding-state-change"><dt>overflow</dt><dd><span class="coding-state-before">[3,5,6]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[4,6,5]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="6" data-frame-key="return-root" hidden role="group" aria-label="Read the kth largest"><div class="coding-trace-frame-heading"><span>Read the kth largest</span><strong>The root is the smallest of the two retained values: min([5,6]) = 5, so the second largest is 5.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-5-0">5</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-6-0">6</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>k</b>2</span><span><b>arithmetic</b>min([5,6]) = 5</span><span><b>result</b>5</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>k</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2</strong></dd></div><div class="coding-state-change"><dt>arithmetic</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">min([5,6]) = 5</strong></dd></div><div class="coding-state-change"><dt>result</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">5</strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 7</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Seed the candidate heap</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Fill the second slot</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Discard an undersized candidate</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Raise the protected floor</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Keep the next large value</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Reject the last weaker value</strong></button><button type="button" data-coding-frame-button="6"><span>7</span><strong>Read the kth largest</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when an unsorted or streaming input asks for the kth largest item or the largest k items, while sorting every value would retain more order than the answer needs.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">After each input is pushed and any size-(k+1) overflow root is popped, the heap contains exactly the largest min(k, processed) values seen; its root is the weakest retained candidate.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">For a bounded best-k set, choose the opposite heap polarity: a min-heap protects the largest k and a max-heap protects the smallest k; the root is always the next item to evict.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> For nums = [3,2,1,5,6,4] and k = 2, push 3. The heap has room, so nothing is removed. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="kth-largest-element-state"><strong>Check:</strong> Why is the root 5, rather than the largest retained value 6, the answer for k=2?</span></figcaption></figure>

Heap order guarantees each parent is no larger than its children.
It does not sort every array position.
The root becomes the kth largest only after at least `k` values arrive.

## Code

```python
import heapq

def find_kth_largest(nums: list[int], k: int) -> int:
   if not 1 <= k <= len(nums):
      raise ValueError("k must name an item in nums")

   heap: list[int] = []
   for num in nums:
      heapq.heappush(heap, num)
      if len(heap) > k:
         heapq.heappop(heap)
   return heap[0]
```

Time is $O(n\log(k+1))$; auxiliary heap space is $O(k)$.
The heap briefly holds `k+1` candidates before eviction.
Writing `log(k+1)` also covers `k=1` without suggesting zero work.
The input list is not modified.

### Failure Check

Evicting the largest retained value protects the wrong end of the ordering.
Silently deduplicating changes the requested rank: `[5,5,4]` would incorrectly return `4` for this contract.

<!-- learning:core-end -->

## Change the Rank

**New task:** Count distinct values only.

**Picture and change:** Deduplicate before using the retained reference.
Its range guard now applies to the distinct count.

```python
def kth_largest_distinct(nums, k):
   return find_kth_largest(list(set(nums)), k)
```

**Check:** `[5,5,4]`, rank `2`, returns `4`.
Rank `3` now raises `ValueError`.
The set and list add $O(d)$ storage for $d$ distinct values, plus expected linear hashing work.

**Use when:** The specification explicitly asks for a distinct-value rank.

## Change to Prefix Answers

**New task:** Report the kth largest after every stream prefix.

**Picture and change:** Preserve the bounded heap across arrivals.
Return `None` until `k` occurrences exist.
Require a positive integer `k`; total stream length need not be known.

```python
def running_kth(values, k):
   if k <= 0:
      raise ValueError("positive k required")
   heap, answer = [], []
   for value in values:
      heapq.heappush(heap, value)
      if len(heap) > k:
         heapq.heappop(heap)
      answer.append(heap[0] if len(heap) == k else None)
   return answer
```

**Check:** The source stream returns `[None,2,2,3,5,5]`.
Once defined, the threshold never decreases as more values arrive.
Keep the `heapq` import above.
Time is $O(n\log(k+1))$, retained state is $O(k)$, and the returned history adds $O(n)$ space.

**Use when:** Answers are required before the complete input arrives.

### Transfer Check

Why use a min-heap when the requested rank is largest-first?

### Answer

The root must expose the worst protected candidate for eviction.
The desired output direction does not determine the useful eviction direction.
Compare [frequency ranking](/questions/top-k-frequent-elements/) when priority comes from counts rather than raw values.
