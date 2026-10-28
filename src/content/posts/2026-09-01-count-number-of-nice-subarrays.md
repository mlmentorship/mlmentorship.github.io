---
title: "Count Number of Nice Subarrays"
description: "Count subarrays that contain exactly `k` odd numbers."
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

Count contiguous input ranges containing exactly a requested number of odd integers.

## Problem

**Input.** An integer list `nums` and a nonnegative integer `odd_count`.

**Output.** Return the number of nonempty contiguous subarrays containing exactly `odd_count` odd values.

**Rules.** Count ranges by their starting and ending positions, even when their values are identical.
Values may be negative, zero, or repeated. Exactly zero odd values means an all-even range.
Do not modify the input. An empty list contains zero qualifying ranges.

**Example.** For `nums = [1, 1, 2, 1, 1]` and `odd_count = 3`, return `2`.
The zero-based inclusive ranges are `[0..3]` and `[1..4]`; both contain three odd values.

## Picture: Count Valid Starts at Each Ending

<!-- visual:count-number-of-nice-subarrays-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="count-number-of-nice-subarrays-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="count-number-of-nice-subarrays-state-title">Count Number of Nice Subarrays: Count endings with at most 3 odds and at most 2 odds at every R, then subtract 14 - 12.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="count-number-of-nice-subarrays" role="group" tabindex="0" aria-label="Count Number of Nice Subarrays: Count endings with at most 3 odds and at most 2 odds at every R, then subtract 14 - 12."><div class="coding-visual-example"><span>Input and goal</span><strong>Count subarrays that contain exactly `k` odd numbers.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="right-0" role="group" aria-label="Read the first odd at R = 0"><div class="coding-trace-frame-heading"><span>Read the first odd at R = 0</span><strong>Both windows start at 0. Each has one valid ending at R, so totals become at_most(3) = 1 and at_most(2) = 1.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state trace-tone-state trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="left-at-most-3">L&lt;=3</span><span class="coding-trace-array-pointer" data-motion-key="left-at-most-2">L&lt;=2</span><span class="coding-trace-array-pointer" data-motion-key="right">R</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>ranges</b>&lt;=3 [0..0]; &lt;=2 [0..0]</span><span><b>additions</b>1 and 1</span><span><b>totals</b>1 and 1</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>left-at-most-3</dt><dd><strong class="coding-state-after">0</strong></dd></div><div class="coding-state-change"><dt>left-at-most-2</dt><dd><strong class="coding-state-after">0</strong></dd></div><div class="coding-state-change"><dt>right</dt><dd><strong class="coding-state-after">0</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="right-1" hidden role="group" aria-label="Read the second odd at R = 1"><div class="coding-trace-frame-heading"><span>Read the second odd at R = 1</span><strong>Both budgets still fit. Each left boundary stays 0, adding R - L + 1 = 2 endings; totals become 3 and 3.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="left-at-most-3">L&lt;=3</span><span class="coding-trace-array-pointer" data-motion-key="left-at-most-2">L&lt;=2</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="right">R</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>ranges</b>&lt;=3 [0..1]; &lt;=2 [0..1]</span><span><b>additions</b>2 and 2</span><span><b>totals</b>3 and 3</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>right</dt><dd><span class="coding-state-before">0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>ranges</dt><dd><span class="coding-state-before">&lt;=3 [0..0]; &lt;=2 [0..0]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">&lt;=3 [0..1]; &lt;=2 [0..1]</strong></dd></div><div class="coding-state-change"><dt>additions</dt><dd><span class="coding-state-before">1 and 1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2 and 2</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="right-2" hidden role="group" aria-label="Read the even value at R = 2"><div class="coding-trace-frame-heading"><span>Read the even value at R = 2</span><strong>The even value spends no odd budget. Both L pointers stay 0 and each window adds 2 - 0 + 1 = 3 endings; totals become 6 and 6.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="left-at-most-3">L&lt;=3</span><span class="coding-trace-array-pointer" data-motion-key="left-at-most-2">L&lt;=2</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="right">R</span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>ranges</b>&lt;=3 [0..2]; &lt;=2 [0..2]</span><span><b>additions</b>3 and 3</span><span><b>totals</b>6 and 6</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>right</dt><dd><span class="coding-state-before">1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2</strong></dd></div><div class="coding-state-change"><dt>ranges</dt><dd><span class="coding-state-before">&lt;=3 [0..1]; &lt;=2 [0..1]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">&lt;=3 [0..2]; &lt;=2 [0..2]</strong></dd></div><div class="coding-state-change"><dt>additions</dt><dd><span class="coding-state-before">2 and 2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3 and 3</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="right-3" hidden role="group" aria-label="Third odd forces only L&lt;=2 right"><div class="coding-trace-frame-heading"><span>Third odd forces only L&lt;=2 right</span><strong>At R = 3 the &lt;=3 window still starts at 0 and adds 4. The &lt;=2 budget goes negative, so remove nums[0] = 1 and move L&lt;=2 to 1, adding 3.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="left-at-most-3">L&lt;=3</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="left-at-most-2">L&lt;=2</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="right">R</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>ranges</b>&lt;=3 [0..3]; &lt;=2 [1..3]</span><span><b>direction</b>L&lt;=3 stays 0; L&lt;=2: 0 -&gt; 1</span><span><b>additions</b>4 and 3</span><span><b>totals</b>10 and 9</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>left-at-most-2</dt><dd><span class="coding-state-before">0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>right</dt><dd><span class="coding-state-before">2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>ranges</dt><dd><span class="coding-state-before">&lt;=3 [0..2]; &lt;=2 [0..2]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">&lt;=3 [0..3]; &lt;=2 [1..3]</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="right-4-result" hidden role="group" aria-label="Fourth odd advances both boundaries"><div class="coding-trace-frame-heading"><span>Fourth odd advances both boundaries</span><strong>At R = 4, remove nums[0] = 1 for &lt;=3 and nums[1] = 1 for &lt;=2. The windows add 4 and 3 endings, producing 14 - 12 = 2 exact matches.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="left-at-most-3">L&lt;=3</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="left-at-most-2">L&lt;=2</span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-pointer" data-motion-key="right">R</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>ranges</b>&lt;=3 [1..4]; &lt;=2 [2..4]</span><span><b>direction</b>L&lt;=3: 0 -&gt; 1; L&lt;=2: 1 -&gt; 2</span><span><b>additions</b>4 and 3</span><span><b>totals</b>14 and 12</span><span><b>result</b>14 - 12 = 2</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>left-at-most-3</dt><dd><span class="coding-state-before">0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>left-at-most-2</dt><dd><span class="coding-state-before">1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2</strong></dd></div><div class="coding-state-change"><dt>right</dt><dd><span class="coding-state-before">3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">4</strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Read the first odd at R = 0</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Read the second odd at R = 1</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Read the even value at R = 2</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Third odd forces only L&lt;=2 right</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Fourth odd advances both boundaries</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when subarrays must contain exactly k nonnegative events, while an at-most-k window can be repaired monotonically by moving its left boundary right.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">For each R, every start from L through R satisfies the relevant at-most limit, so that pass adds exactly R - L + 1 valid endings. The running total includes all right endpoints processed so far.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Convert exact monotone counts into at_most(k) - at_most(k - 1); at each right endpoint, the difference between the two valid-start ranges counts exactly-k subarrays ending there.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Both windows start at 0. Each has one valid ending at R, so totals become at_most(3) = 1 and at_most(2) = 1. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="count-number-of-nice-subarrays-state"><strong>Check:</strong> Which start remains after subtracting the at-most-two starts at the final right endpoint?</span></figcaption></figure>

Only `num % 2` is accumulated. Each contribution is 0 or 1, so extending a range cannot decrease its event count.

After repair, `left` is the earliest valid start for this `right`.
Every start from `left` through `right` fits; add `right - left + 1`, not just one longest window.
Earlier discarded starts stay invalid as `right` advances.

| right | At-most-3 addition | At-most-2 addition | Exactly-3 addition |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 0 |
| 1 | 2 | 2 | 0 |
| 2 | 3 | 3 | 0 |
| 3 | 4 | 3 | 1 |
| 4 | 4 | 3 | 1 |
| Total | 14 | 12 | 2 |

The at-most-2 ranges are a subset of the at-most-3 ranges. Their difference leaves exactly three odds.
Generally, `exactly(k) = at_most(k) - at_most(k - 1)`.

## Code

```python
def number_of_nice_subarrays(nums: list[int], odd_count: int) -> int:
   def at_most(limit: int) -> int:
      if limit < 0:
         return 0

      left = 0
      total = 0
      for right, num in enumerate(nums):
         limit -= num % 2
         while limit < 0:
            limit += nums[left] % 2
            left += 1
         total += right - left + 1
      return total

   return at_most(odd_count) - at_most(odd_count - 1)
```

`limit` is the remaining budget: initial limit minus the current window's odd count.
Each boundary advances at most $n$ positions per pass: total $O(n)$ time and $O(1)$ auxiliary space under unit-cost arithmetic.

The negative-limit guard makes `at_most(-1)` zero. Thus `[2, 4]`, count 0, returns three nonempty ranges without overrunning the array.

## Change 1: Compute With Prefix Frequencies

**New task.** Use boundary counts instead of window repair.

**Picture and change.** Count earlier odd-prefix totals equal to `prefix - odd_count`.

**Code.**

```python
def nice_by_prefix(nums, odd_count):
   counts = {0: 1}
   prefix = answer = 0
   for value in nums:
      prefix += value % 2
      answer += counts.get(prefix - odd_count, 0)
      counts[prefix] = counts.get(prefix, 0) + 1
   return answer
```

**Check.** The main input returns 2; `[2, 4]`, count 0, returns 3.

**Use when.** A boundary-frequency derivation is easier to extend. This alternate uses $O(n)$ state, not constant space.

## Change 2: Count a Target Character

**New task.** Count ranges with exactly `count` occurrences of one target character; require `count >= 0`.

**Picture and change.** Replace parity with a 0/1 equality indicator.

**Code.**

```python
def exact_character_ranges(text, target, count):
   counts = {0: 1}
   prefix = answer = 0
   for character in text:
      prefix += character == target
      answer += counts.get(prefix - count, 0)
      counts[prefix] = counts.get(prefix, 0) + 1
   return answer
```

**Check.** For `text="aba"`, target `"a"`, count 1 returns 4; count 0 returns 1 for `"b"`.

**Use when.** The event changes but remains a nonnegative indicator.

## Failure: Shrinking an Arbitrary Signed Sum

Signed values can decrease sums when appended; that invalidates this window-repair proof. Counting one start also misses valid ranges.

## Transfer Check

Why are negative integers safe here?

### Reasoned Answer

Parity contributes only 0 or 1. The original sign never becomes a negative event count.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Same Count, Different Proof and State

The trace's `R` is code `right`; its two left pointers correspond to separate `at_most` calls.
The visual synchronizes those calls for comparison; the implementation performs two linear passes.

For `[-3, 2, -1]`, parity is `[1, 0, 1]`; exactly one odd occurs in four ranges.
Empty input contributes no nonempty ranges. The public contract excludes negative requested counts.

Prefix frequency counts earlier boundaries, including the initial zero boundary.
Querying before recording the current boundary excludes empty ranges, including when the requested count is zero.
See [signed prefix counting](/questions/subarray-sum-equals-k/) for the boundary-difference proof; it is not this window proof.

</div>
