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
<figure class="learning-figure coding-visual-figure" aria-labelledby="reverse-bits-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="reverse-bits-state-title">Reverse Bits: Read one input bit from the right and append it to the answer on the left.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="reverse-bits" role="group" tabindex="0" aria-label="Reverse Bits: Read one input bit from the right and append it to the answer on the left."><div class="coding-visual-example"><span>Input and goal</span><strong>Reverse the 32 bits of an unsigned integer.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Read the low bit"><div class="coding-trace-frame-heading"><span>Read the low bit</span><strong>The input cursor starts at the least-significant bit. The drawing shows an 8-bit slice; the implementation repeats the same move 32 times.</strong></div><div class="coding-trace-bits"><span class="coding-trace-bit trace-tone-focus"><b>1</b><small>read</small></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>1</b></span><span class="coding-trace-bit"><b>1</b></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>1</b></span><span class="coding-trace-bit"><b>0</b></span></div><div class="coding-trace-meta"><span><b>input</b>right -&gt; left</span><span><b>output</b>empty</span><span><b>width</b>8-bit illustration</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Append to output"><div class="coding-trace-frame-heading"><span>Append to output</span><strong>Shift the output left and place the read bit at its low end.</strong></div><div class="coding-trace-bits"><span class="coding-trace-bit trace-tone-state"><b>1</b><small>read</small></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>1</b></span><span class="coding-trace-bit"><b>1</b></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>1</b></span><span class="coding-trace-bit trace-tone-focus"><b>0</b><small>write</small></span></div><div class="coding-trace-meta"><span><b>output</b>1</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Repeat 32 times"><div class="coding-trace-frame-heading"><span>Repeat 32 times</span><strong>After fixed-width processing, the bit order is reversed.</strong></div><div class="coding-trace-bits"><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>1</b></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>1</b></span><span class="coding-trace-bit"><b>1</b></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>1</b></span></div><div class="coding-trace-meta"><span><b>result</b>reversed 32-bit word</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Read the low bit</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Append to output</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Repeat 32 times</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Read one input bit from the right and append it to the answer on the left.</p></div><figcaption><strong>Read it this way:</strong> The input cursor starts at the least-significant bit. The drawing shows an 8-bit slice; the implementation repeats the same move 32 times. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
