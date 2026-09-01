---
title: "Reverse Bits"
description: "Reverse the 32 bits of an unsigned integer."
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

> Reverse the 32 bits of an unsigned integer.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:reverse-bits-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="reverse-bits-state-title"><p class="visual-kicker">Bits as visible state</p><p class="visual-title" id="reverse-bits-state-title">Reverse Bits: Use one local bit identity to remove or cancel work</p><div class="coding-visual coding-visual--bit" data-coding-visual data-coding-mode="bit" data-coding-slug="reverse-bits" role="group" aria-label="Reverse Bits: read the input from right to left while appending each bit to the answer. Each step preserves the numerical meaning of the bits not yet processed."><div class="coding-visual-example"><span>Concrete trace</span><strong>read the input from right to left while appending each bit to the answer</strong></div><div class="coding-visual-sketch coding-visual-sketch--bit"><div class="coding-sketch-bits"><span>1</span><span>0</span><span>1</span><span class="coding-sketch-bit--active">1</span><span>0</span><span>1</span><span>0</span><span>0</span></div><p class="coding-sketch-note">read, cancel, or carry one bit position at a time</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Read</span><strong>lowest bit</strong><small>Inspect the bit at the edge of the word.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Combine</span><strong>XOR / AND</strong><small>Separate information from carry or cancellation.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Shift</span><strong>move one place</strong><small>Bring the next bit into position.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Finish</span><strong>zero or fixed width</strong><small>Stop when the represented state is complete.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Each step preserves the numerical meaning of the bits not yet processed.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Watch bits move from one position to the next. XOR keeps non-carrying differences, AND identifies shared one-bits, and shifts expose the next position. For this problem, hold onto the concrete trace: read the input from right to left while appending each bit to the answer.</figcaption></figure>

**Pattern:** Read from one end and build at the other.

**Simple idea:** Read the last input bit, append it to the answer, then shift the input
right. Repeat exactly 32 times.

```python
def reverse_bits(value: int) -> int:
   answer = 0
   for _ in range(32):
      answer = (answer << 1) | (value & 1)
      value >>= 1
   return answer
```

**Cost:** $O(1)$ time and space for 32 bits.
