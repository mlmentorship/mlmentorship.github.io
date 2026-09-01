---
title: "Split Array Largest Sum"
description: "Split an array into `k` nonempty continuous parts. Make the largest part sum as small as possible."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Advanced"
priority: "Specialist"
aliases: []
prerequisites: []
---

> Split an array into `k` nonempty continuous parts. Make the largest part sum as small as possible.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:split-array-largest-sum-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="split-array-largest-sum-state-title"><p class="visual-kicker">A shrinking answer space</p><p class="visual-title" id="split-array-largest-sum-state-title">Split Array Largest Sum: Discard a half only after a yes-or-no test</p><div class="coding-visual coding-visual--binary" data-coding-visual data-coding-mode="binary" data-coding-slug="split-array-largest-sum" role="group" aria-label="Split Array Largest Sum: [7,2,5,10,8], k=2 -&gt; test a limit and count how many parts it forces. The answer never leaves the current low-to-high interval."><div class="coding-visual-example"><span>Concrete trace</span><strong>[7,2,5,10,8], k=2 -&gt; test a limit and count how many parts it forces</strong></div><div class="coding-visual-sketch coding-visual-sketch--binary"><div class="coding-sketch-array"><span class="coding-sketch-pointer">lo</span><span class="coding-sketch-cell">left</span><span class="coding-sketch-cell">left</span><span class="coding-sketch-cell coding-sketch-cell--active">mid</span><span class="coding-sketch-cell">right</span><span class="coding-sketch-cell">right</span><span class="coding-sketch-pointer">hi</span></div><p class="coding-sketch-note">probe the middle, then discard the side the predicate rules out</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Bound</span><strong>lo ... hi</strong><small>Every possible answer is inside this interval.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Probe</span><strong>mid</strong><small>Test the middle value or candidate answer.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Decide</span><strong>predicate</strong><small>The monotone result says which side can survive.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Keep</span><strong>one half</strong><small>Move one boundary and preserve the answer.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The answer never leaves the current low-to-high interval.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Read the interval as a promise: everything outside it is already impossible. The midpoint is useful only because the predicate is monotone. For this problem, hold onto the concrete trace: [7,2,5,10,8], k=2 -&gt; test a limit and count how many parts it forces.</figcaption></figure>

**Pattern:** Binary search on the answer.

**Simple idea:** Guess the largest allowed sum. Start a new part when adding the next value
would pass the guess. If this needs at most `k` parts, the guess works.

```python
def split_array_largest_sum(nums: list[int], parts: int) -> int:
   def parts_needed(limit: int) -> int:
      used = 1
      total = 0
      for num in nums:
         if total + num > limit:
            used += 1
            total = 0
         total += num
      return used

   left, right = max(nums), sum(nums)
   while left < right:
      middle = (left + right) // 2
      if parts_needed(middle) <= parts:
         right = middle
      else:
         left = middle + 1
   return left
```

**Cost:** $O(n\log s)$ time and $O(1)$ space, where $s$ is the sum range searched.
