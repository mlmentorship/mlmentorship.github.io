---
title: "Contains Duplicate"
description: "Check whether any value appears more than once."
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

> Check whether any value appears more than once.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:contains-duplicate-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="contains-duplicate-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="contains-duplicate-state-title">Contains Duplicate: The first repeated value is visible when it is already in the seen set.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="contains-duplicate" role="group" aria-label="Contains Duplicate: The first repeated value is visible when it is already in the seen set."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether any value appears more than once.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Save new values"><div class="coding-trace-frame-heading"><span>Save new values</span><strong>1, 2, and 3 have not appeared before.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">current</span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span></span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>1</b><span>seen</span></span><span class="coding-trace-map-entry"><b>2</b><span>seen</span></span><span class="coding-trace-map-entry"><b>3</b><span>seen</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Detect the repeat"><div class="coding-trace-frame-heading"><span>Detect the repeat</span><strong>The final 1 is already in the set.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">same value</span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">repeat</span><span class="coding-trace-array-cell">1</span></span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>1</b><span>seen</span></span><span class="coding-trace-map-entry"><b>2</b><span>seen</span></span><span class="coding-trace-map-entry"><b>3</b><span>seen</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Return true"><div class="coding-trace-frame-heading"><span>Return true</span><strong>A set membership hit proves a duplicate exists.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">duplicate</span><span class="coding-trace-array-cell">1</span></span></div><div class="coding-trace-meta"><span><b>result</b>true</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Save new values</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Detect the repeat</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return true</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>The first repeated value is visible when it is already in the seen set.</p></div><figcaption><strong>Read it this way:</strong> 1, 2, and 3 have not appeared before. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Set.

**Simple idea:** A set removes repeated values. A duplicate exists when the set is shorter
than the input.

```python
def contains_duplicate(nums: list[int]) -> bool:
   return len(nums) != len(set(nums))
```

**Cost:** $O(n)$ average time and $O(n)$ space.
