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
<figure class="learning-figure coding-visual-figure" aria-labelledby="number-of-1-bits-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="number-of-1-bits-state-title">Number of 1 Bits: The operation x &amp; (x-1) removes exactly the lowest set bit.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="number-of-1-bits" role="group" aria-label="Number of 1 Bits: The operation x &amp; (x-1) removes exactly the lowest set bit."><div class="coding-visual-example"><span>Input and goal</span><strong>Count the set bits in an integer.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Read the bits"><div class="coding-trace-frame-heading"><span>Read the bits</span><strong>11 is binary 1011 and has three set bits.</strong></div><div class="coding-trace-bits"><span class="coding-trace-bit"><b>1</b></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>1</b></span><span class="coding-trace-bit trace-tone-focus"><b>1</b><small>lowest 1</small></span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Clear one bit"><div class="coding-trace-frame-heading"><span>Clear one bit</span><strong>1011 becomes 1010; two more applications produce 1000 and then 0000.</strong></div><div class="coding-trace-bits"><span class="coding-trace-bit"><b>1</b></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>1</b></span><span class="coding-trace-bit trace-tone-state"><b>0</b><small>cleared</small></span></div><div class="coding-trace-meta"><span><b>action</b>count = 1; next 1000 -&gt; 0000</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Stop at zero"><div class="coding-trace-frame-heading"><span>Stop at zero</span><strong>Three bit removals means Hamming weight 3.</strong></div><div class="coding-trace-bits"><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>0</b></span></div><div class="coding-trace-meta"><span><b>result</b>3</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Read the bits</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Clear one bit</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Stop at zero</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>The operation x &amp; (x-1) removes exactly the lowest set bit.</p></div><figcaption><strong>Read it this way:</strong> 11 is binary 1011 and has three set bits. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
