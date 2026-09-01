---
title: "Missing Number"
description: "Values come from 0 through `n`, with one missing. Return the missing value."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Mixed"
priority: "Core"
aliases: []
prerequisites: []
---

> Values come from 0 through `n`, with one missing. Return the missing value.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:missing-number-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="missing-number-state-title"><p class="visual-kicker">Bits as visible state</p><p class="visual-title" id="missing-number-state-title">Missing Number: Use one local bit identity to remove or cancel work</p><div class="coding-visual coding-visual--bit" data-coding-visual data-coding-mode="bit" data-coding-slug="missing-number" role="group" aria-label="Missing Number: [3,0,1] -&gt; XOR expected and actual values; unmatched 2 remains. Each step preserves the numerical meaning of the bits not yet processed."><div class="coding-visual-example"><span>Concrete trace</span><strong>[3,0,1] -&gt; XOR expected and actual values; unmatched 2 remains</strong></div><div class="coding-visual-sketch coding-visual-sketch--bit"><div class="coding-sketch-bits"><span>1</span><span>0</span><span>1</span><span class="coding-sketch-bit--active">1</span><span>0</span><span>1</span><span>0</span><span>0</span></div><p class="coding-sketch-note">read, cancel, or carry one bit position at a time</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Read</span><strong>lowest bit</strong><small>Inspect the bit at the edge of the word.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Combine</span><strong>XOR / AND</strong><small>Separate information from carry or cancellation.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Shift</span><strong>move one place</strong><small>Bring the next bit into position.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Finish</span><strong>zero or fixed width</strong><small>Stop when the represented state is complete.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Each step preserves the numerical meaning of the bits not yet processed.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Watch bits move from one position to the next. XOR keeps non-carrying differences, AND identifies shared one-bits, and shifts expose the next position. For this problem, hold onto the concrete trace: [3,0,1] -&gt; XOR expected and actual values; unmatched 2 remains.</figcaption></figure>

**Pattern:** XOR cancellation.

**Simple idea:** XOR every expected index and every actual value. Matching values cancel,
leaving only the missing value.

```python
def missing_number(nums: list[int]) -> int:
   missing = len(nums)
   for index, num in enumerate(nums):
      missing ^= index ^ num
   return missing
```

**Cost:** $O(n)$ time and $O(1)$ space.
