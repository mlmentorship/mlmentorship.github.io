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
<figure class="learning-figure coding-visual-figure" aria-labelledby="sum-of-two-integers-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="sum-of-two-integers-state-title">Sum of Two Integers: XOR supplies sum bits without carry; AND shifted left supplies the carry.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="sum-of-two-integers" role="group" aria-label="Sum of Two Integers: XOR supplies sum bits without carry; AND shifted left supplies the carry."><div class="coding-visual-example"><span>Input and goal</span><strong>Add two integers without `+` or `-`.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Separate sum and carry"><div class="coding-trace-frame-heading"><span>Separate sum and carry</span><strong>For 3 (0011) and 1 (0001), XOR gives 0010 and the carry is 0010.</strong></div><div class="coding-trace-bits"><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit trace-tone-state"><b>1</b><small>xor</small></span><span class="coding-trace-bit trace-tone-state"><b>0</b><small>xor</small></span></div><div class="coding-trace-meta"><span><b>sum</b>0010</span><span><b>carry</b>0010</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Move the carry left"><div class="coding-trace-frame-heading"><span>Move the carry left</span><strong>The next pass combines 0010 with 0010, producing no sum bits and carry 0100.</strong></div><div class="coding-trace-bits"><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit trace-tone-focus"><b>0</b><small>carry</small></span><span class="coding-trace-bit"><b>0</b></span></div><div class="coding-trace-meta"><span><b>sum</b>0000</span><span><b>carry</b>0100</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Stop when carry is zero"><div class="coding-trace-frame-heading"><span>Stop when carry is zero</span><strong>A final pass produces 0100, the sum of 3 and 1.</strong></div><div class="coding-trace-bits"><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit trace-tone-output"><b>1</b><small>4</small></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>0</b></span></div><div class="coding-trace-meta"><span><b>result</b>4</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Separate sum and carry</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Move the carry left</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Stop when carry is zero</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>XOR supplies sum bits without carry; AND shifted left supplies the carry.</p></div><figcaption><strong>Read it this way:</strong> For 3 (0011) and 1 (0001), XOR gives 0010 and the carry is 0010. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
