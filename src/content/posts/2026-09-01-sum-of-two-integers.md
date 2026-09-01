---
title: "Sum of Two Integers"
description: "Add two integers without `+` or `-`."
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

> Add two integers without `+` or `-`.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:sum-of-two-integers-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="sum-of-two-integers-state-title"><p class="visual-kicker">Bits as visible state</p><p class="visual-title" id="sum-of-two-integers-state-title">Sum of Two Integers: Use one local bit identity to remove or cancel work</p><div class="coding-visual coding-visual--bit" data-coding-visual data-coding-mode="bit" data-coding-slug="sum-of-two-integers" role="group" aria-label="Sum of Two Integers: XOR gives provisional sum; AND shifted left gives the carry to add next. Each step preserves the numerical meaning of the bits not yet processed."><div class="coding-visual-example"><span>Concrete trace</span><strong>XOR gives provisional sum; AND shifted left gives the carry to add next</strong></div><div class="coding-visual-sketch coding-visual-sketch--bit"><div class="coding-sketch-bits"><span>1</span><span>0</span><span>1</span><span class="coding-sketch-bit--active">1</span><span>0</span><span>1</span><span>0</span><span>0</span></div><p class="coding-sketch-note">read, cancel, or carry one bit position at a time</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Read</span><strong>lowest bit</strong><small>Inspect the bit at the edge of the word.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Combine</span><strong>XOR / AND</strong><small>Separate information from carry or cancellation.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Shift</span><strong>move one place</strong><small>Bring the next bit into position.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Finish</span><strong>zero or fixed width</strong><small>Stop when the represented state is complete.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Each step preserves the numerical meaning of the bits not yet processed.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Watch bits move from one position to the next. XOR keeps non-carrying differences, AND identifies shared one-bits, and shifts expose the next position. For this problem, hold onto the concrete trace: XOR gives provisional sum; AND shifted left gives the carry to add next.</figcaption></figure>

**Pattern:** XOR gives sum bits and AND gives carry bits.

**Simple idea:** XOR adds without carrying. AND finds positions that need a carry. Shift the
carry left and repeat until no carry remains. The mask keeps Python within 32 bits.

```python
def get_sum(first: int, second: int) -> int:
   mask = 0xFFFFFFFF
   while second:
      first, second = (first ^ second) & mask, ((first & second) << 1) & mask
   return first if first < 0x80000000 else ~(first ^ mask)
```

**Cost:** $O(1)$ time and space for 32-bit integers.
