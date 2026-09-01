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
<figure class="learning-figure coding-visual-figure" aria-labelledby="counting-bits-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="counting-bits-state-title">Counting Bits: Remove the lowest bit and reuse the answer for the shifted number.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="counting-bits" role="group" tabindex="0" aria-label="Counting Bits: Remove the lowest bit and reuse the answer for the shifted number."><div class="coding-visual-example"><span>Input and goal</span><strong>Return the set-bit count for every value from 0 through `n`.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Use a smaller number"><div class="coding-trace-frame-heading"><span>Use a smaller number</span><strong>For 6, shift right to 3 and inspect the low bit 0.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">value</th><th scope="col">value &gt;&gt; 1</th><th scope="col">value &amp; 1</th><th scope="col">count</th></tr></thead><tbody><tr><td class="is-active">6</td><td class="">3</td><td class="">0</td><td class="">?</td></tr><tr><td class="">3</td><td class="">1</td><td class="">1</td><td class="">2</td></tr></tbody></table></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Apply the recurrence"><div class="coding-trace-frame-heading"><span>Apply the recurrence</span><strong>count[6] = count[3] + 0 = 2.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">value</th><th scope="col">shifted count</th><th scope="col">last bit</th><th scope="col">answer</th></tr></thead><tbody><tr><td class="is-active">6</td><td class="">2</td><td class="">0</td><td class="">2</td></tr><tr><td class="">5</td><td class="">2</td><td class="">1</td><td class="">2</td></tr></tbody></table></div><div class="coding-trace-meta"><span><b>action</b>reuse DP</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Fill the line"><div class="coding-trace-frame-heading"><span>Fill the line</span><strong>Every value reuses a previously solved value.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-pointer" data-motion-key="marker-count-5-2">count(5)=2</span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">5</small></span></div><div class="coding-trace-meta"><span><b>result</b>counts 0..5</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Use a smaller number</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Apply the recurrence</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Fill the line</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Remove the lowest bit and reuse the answer for the shifted number.</p></div><figcaption><strong>Read it this way:</strong> For 6, shift right to 3 and inspect the low bit 0. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
