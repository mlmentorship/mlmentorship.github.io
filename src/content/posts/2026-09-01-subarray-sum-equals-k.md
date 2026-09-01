---
title: "Subarray Sum Equals K"
description: "Count continuous subarrays whose sum equals the target."
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

> Count continuous subarrays whose sum equals the target.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:subarray-sum-equals-k-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="subarray-sum-equals-k-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="subarray-sum-equals-k-state-title">Subarray Sum Equals K: Turn a target subarray into a lookup between two prefix sums.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="subarray-sum-equals-k" role="group" tabindex="0" aria-label="Subarray Sum Equals K: Turn a target subarray into a lookup between two prefix sums."><div class="coding-visual-example"><span>Input and goal</span><strong>Count continuous subarrays whose sum equals the target.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Record the empty prefix"><div class="coding-trace-frame-heading"><span>Record the empty prefix</span><strong>Before reading values, prefix sum 0 has appeared once.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-prefix-0">prefix 0</span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">3</small></span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>0</b><span>count 1</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Reach prefix 2"><div class="coding-trace-frame-heading"><span>Reach prefix 2</span><strong>After the second 1, current prefix is 2. It needs an earlier prefix 0.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="marker-prefix-2">prefix 2</span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">3</small></span></div><div class="coding-trace-meta"><span><b>query</b>2 - k = 0</span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>0</b><span>count 1</span></span><span class="coding-trace-map-entry"><b>1</b><span>count 1</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Count every match"><div class="coding-trace-frame-heading"><span>Count every match</span><strong>The prefix-2 query finds prefix 0; prefix 3 later finds prefix 1.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="marker-match">match</span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="marker-match">match</span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">3</small></span></div><div class="coding-trace-meta"><span><b>result</b>2 subarrays</span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>0</b><span>count 1</span></span><span class="coding-trace-map-entry"><b>1</b><span>count 1</span></span><span class="coding-trace-map-entry"><b>2</b><span>count 1</span></span><span class="coding-trace-map-entry"><b>3</b><span>count 1</span></span></div></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Record the empty prefix</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Reach prefix 2</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Count every match</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Turn a target subarray into a lookup between two prefix sums.</p></div><figcaption><strong>Read it this way:</strong> Before reading values, prefix sum 0 has appeared once. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Prefix sum plus hash map.

**Simple idea:** If the current prefix sum is `p`, a wanted subarray starts after an earlier
prefix sum of `p - target`. Store how many times each earlier prefix sum appeared.

This is Two Sum on prefix sums.

```python
def subarray_sum(nums: list[int], target: int) -> int:
   prefix_count = {0: 1}
   prefix = 0
   answer = 0

   for num in nums:
      prefix += num
      answer += prefix_count.get(prefix - target, 0)
      prefix_count[prefix] = prefix_count.get(prefix, 0) + 1

   return answer
```

**Cost:** $O(n)$ time and $O(n)$ space.

**Why the map starts with `{0: 1}`:** A prefix that already equals the target forms a valid
subarray starting at index 0.
