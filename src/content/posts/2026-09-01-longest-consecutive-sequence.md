---
title: "Longest Consecutive Sequence"
description: "Find the length of the longest run of consecutive values in an unsorted array."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Foundation"
priority: "Core"
aliases: []
prerequisites: []
---

> Find the length of the longest run of consecutive values in an unsorted array.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:longest-consecutive-sequence-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="longest-consecutive-sequence-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="longest-consecutive-sequence-state-title">Longest Consecutive Sequence: Start a run only at a value with no predecessor.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="longest-consecutive-sequence" role="group" tabindex="0" aria-label="Longest Consecutive Sequence: Start a run only at a value with no predecessor."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the length of the longest run of consecutive values in an unsorted array.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Find a run start"><div class="coding-trace-frame-heading"><span>Find a run start</span><strong>4 is skipped because 3 exists. 1 has no predecessor, so it starts a run.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">100</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">200</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="marker-start">start</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">5</small></span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>1</b><span>start</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Walk forward"><div class="coding-trace-frame-heading"><span>Walk forward</span><strong>The set answers 2, 3, and 4 in constant-time average lookups.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-start">start</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="marker-end">end</span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">3</small></span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>1</b><span>run length 4</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Keep the longest"><div class="coding-trace-frame-heading"><span>Keep the longest</span><strong>Every other value either starts a shorter run or belongs to this one.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-best">best</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="marker-best">best</span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="marker-best">best</span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="marker-best">best</span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">3</small></span></div><div class="coding-trace-meta"><span><b>result</b>4</span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>1</b><span>best = 4</span></span></div></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Find a run start</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Walk forward</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Keep the longest</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Start a run only at a value with no predecessor.</p></div><figcaption><strong>Read it this way:</strong> 4 is skipped because 3 exists. 1 has no predecessor, so it starts a run. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Set.

**Simple idea:** Start counting only when `value - 1` is missing. That means the value is the
start of a run. Each run is counted once.

```python
def longest_consecutive(nums: list[int]) -> int:
   values = set(nums)
   best = 0

   for start in values:
      if start - 1 in values:
         continue

      end = start
      while end + 1 in values:
         end += 1
      best = max(best, end - start + 1)

   return best
```

**Cost:** $O(n)$ average time and $O(n)$ space.
