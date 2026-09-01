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
<figure class="learning-figure coding-visual-figure" aria-labelledby="missing-number-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="missing-number-state-title">Missing Number: XOR cancels every value that appears in both the expected and actual sets.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="missing-number" role="group" tabindex="0" aria-label="Missing Number: XOR cancels every value that appears in both the expected and actual sets."><div class="coding-visual-example"><span>Input and goal</span><strong>Values come from 0 through `n`, with one missing. Return the missing value.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Pair expected and actual"><div class="coding-trace-frame-heading"><span>Pair expected and actual</span><strong>Expected values are 0,1,2,3; actual values are 3,0,1.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">expected</th><th scope="col">actual</th><th scope="col">xor</th></tr></thead><tbody><tr><td class="">0</td><td class="">3</td><td class="">0 xor 3</td></tr><tr><td class="">1</td><td class="">0</td><td class="">1 xor 0</td></tr><tr><td class="is-active">2</td><td class="">-</td><td class="">2 remains</td></tr><tr><td class="">3</td><td class="">1</td><td class="">3 xor 1</td></tr></tbody></table></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Cancel matches"><div class="coding-trace-frame-heading"><span>Cancel matches</span><strong>0, 1, and 3 cancel in pairs; only 2 remains.</strong></div><div class="coding-trace-bits"><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit trace-tone-focus"><b>1</b><small>uncancelled 2</small></span><span class="coding-trace-bit"><b>0</b></span></div><div class="coding-trace-meta"><span><b>detail</b>XOR result = 2</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Return the survivor"><div class="coding-trace-frame-heading"><span>Return the survivor</span><strong>The missing value is 2.</strong></div><div class="coding-trace-bits"><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit"><b>0</b></span><span class="coding-trace-bit trace-tone-output"><b>1</b><small>missing</small></span><span class="coding-trace-bit"><b>0</b></span></div><div class="coding-trace-meta"><span><b>result</b>2</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Pair expected and actual</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Cancel matches</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the survivor</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>XOR cancels every value that appears in both the expected and actual sets.</p></div><figcaption><strong>Read it this way:</strong> Expected values are 0,1,2,3; actual values are 3,0,1. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
