---
title: "Counting Bits"
description: "Return the set-bit count for every value from 0 through `n`."
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

> Return the set-bit count for every value from 0 through `n`.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:counting-bits-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="counting-bits-state-title"><p class="visual-kicker">Bits as visible state</p><p class="visual-title" id="counting-bits-state-title">Counting Bits: Use one local bit identity to remove or cancel work</p><div class="coding-visual coding-visual--bit" data-coding-visual data-coding-mode="bit" data-coding-slug="counting-bits" role="group" aria-label="Counting Bits: bits[6] = bits[3] + 0 because 6 &gt;&gt; 1 is 3. Each step preserves the numerical meaning of the bits not yet processed."><div class="coding-visual-example"><span>Concrete trace</span><strong>bits[6] = bits[3] + 0 because 6 &gt;&gt; 1 is 3</strong></div><div class="coding-visual-sketch coding-visual-sketch--bit"><div class="coding-sketch-bits"><span>1</span><span>0</span><span>1</span><span class="coding-sketch-bit--active">1</span><span>0</span><span>1</span><span>0</span><span>0</span></div><p class="coding-sketch-note">read, cancel, or carry one bit position at a time</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Read</span><strong>lowest bit</strong><small>Inspect the bit at the edge of the word.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Combine</span><strong>XOR / AND</strong><small>Separate information from carry or cancellation.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Shift</span><strong>move one place</strong><small>Bring the next bit into position.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Finish</span><strong>zero or fixed width</strong><small>Stop when the represented state is complete.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Each step preserves the numerical meaning of the bits not yet processed.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Watch bits move from one position to the next. XOR keeps non-carrying differences, AND identifies shared one-bits, and shifts expose the next position. For this problem, hold onto the concrete trace: bits[6] = bits[3] + 0 because 6 &gt;&gt; 1 is 3.</figcaption></figure>

**Pattern:** DP from a number with its last bit removed.

**Simple idea:** `value >> 1` is the same number without its last bit. Add that last bit to
the saved answer.

```python
def count_bits(limit: int) -> list[int]:
   answer = [0] * (limit + 1)
   for value in range(1, limit + 1):
      answer[value] = answer[value >> 1] + (value & 1)
   return answer
```

**Cost:** $O(n)$ time and $O(n)$ answer space.
