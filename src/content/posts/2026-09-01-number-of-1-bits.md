---
title: "Number of 1 Bits"
description: "Count the set bits in an integer."
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

> Count the set bits in an integer.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:number-of-1-bits-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="number-of-1-bits-state-title"><p class="visual-kicker">Bits as visible state</p><p class="visual-title" id="number-of-1-bits-state-title">Number of 1 Bits: Use one local bit identity to remove or cancel work</p><div class="coding-visual coding-visual--bit" data-coding-visual data-coding-mode="bit" data-coding-slug="number-of-1-bits" role="group" aria-label="Number of 1 Bits: 1011 -&gt; clear the lowest 1 three times. Each step preserves the numerical meaning of the bits not yet processed."><div class="coding-visual-example"><span>Concrete trace</span><strong>1011 -&gt; clear the lowest 1 three times</strong></div><div class="coding-visual-sketch coding-visual-sketch--bit"><div class="coding-sketch-bits"><span>1</span><span>0</span><span>1</span><span class="coding-sketch-bit--active">1</span><span>0</span><span>1</span><span>0</span><span>0</span></div><p class="coding-sketch-note">read, cancel, or carry one bit position at a time</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Read</span><strong>lowest bit</strong><small>Inspect the bit at the edge of the word.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Combine</span><strong>XOR / AND</strong><small>Separate information from carry or cancellation.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Shift</span><strong>move one place</strong><small>Bring the next bit into position.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Finish</span><strong>zero or fixed width</strong><small>Stop when the represented state is complete.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Each step preserves the numerical meaning of the bits not yet processed.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Watch bits move from one position to the next. XOR keeps non-carrying differences, AND identifies shared one-bits, and shifts expose the next position. For this problem, hold onto the concrete trace: 1011 -&gt; clear the lowest 1 three times.</figcaption></figure>

**Pattern:** Remove one set bit at a time.

**Simple idea:** `value & (value - 1)` changes the lowest `1` bit to `0`. Count how many
times this can happen.

```python
def hamming_weight(value: int) -> int:
   count = 0
   while value:
      value &= value - 1
      count += 1
   return count
```

**Cost:** $O(b)$ time and $O(1)$ space, where $b$ is the number of set bits.
