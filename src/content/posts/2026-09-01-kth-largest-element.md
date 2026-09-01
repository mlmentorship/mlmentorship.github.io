---
title: "Kth Largest Element"
description: "Find the `k`th largest value in an unsorted array."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Intermediate"
priority: "Core"
aliases: []
prerequisites: []
---

> Find the `k`th largest value in an unsorted array.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:kth-largest-element-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="kth-largest-element-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="kth-largest-element-state-title">Kth Largest Element: Keep only the largest k values; the smallest of those is the kth largest.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="kth-largest-element" role="group" tabindex="0" aria-label="Kth Largest Element: Keep only the largest k values; the smallest of those is the kth largest."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the `k`th largest value in an unsorted array.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Fill a size-2 heap"><div class="coding-trace-frame-heading"><span>Fill a size-2 heap</span><strong>Read 3 and 2. Both remain candidates for the top two.</strong></div><div class="coding-trace-heap"><svg viewBox="0 0 480 144" role="img" aria-label="Complete binary heap topology"><line class="coding-trace-heap-edge coding-trace-edge-line" x1="240" y1="32" x2="160" y2="104" /><g class="coding-trace-heap-node is-root" data-motion-key="heap-value-2-0"><circle cx="240" cy="32" r="21" /><text x="240" y="36">2</text></g><g class="coding-trace-heap-node" data-motion-key="heap-value-3-0"><circle cx="160" cy="104" r="21" /><text x="160" y="108">3</text></g></svg></div><div class="coding-trace-meta"><span><b>root</b>2</span><span><b>detail</b>size 2</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Replace the weak root"><div class="coding-trace-frame-heading"><span>Replace the weak root</span><strong>5 arrives and evicts 2. The heap now protects 3 and 5.</strong></div><div class="coding-trace-heap"><svg viewBox="0 0 480 144" role="img" aria-label="Complete binary heap topology"><line class="coding-trace-heap-edge coding-trace-edge-line" x1="240" y1="32" x2="160" y2="104" /><g class="coding-trace-heap-node is-root" data-motion-key="heap-value-3-0"><circle cx="240" cy="32" r="21" /><text x="240" y="36">3</text></g><g class="coding-trace-heap-node" data-motion-key="heap-value-5-0"><circle cx="160" cy="104" r="21" /><text x="160" y="108">5</text></g></svg></div><div class="coding-trace-meta"><span><b>root</b>3</span><span><b>detail</b>2 discarded</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Return the root"><div class="coding-trace-frame-heading"><span>Return the root</span><strong>After all values, heap [5,6] has root 5, the second largest.</strong></div><div class="coding-trace-heap"><svg viewBox="0 0 480 144" role="img" aria-label="Complete binary heap topology"><line class="coding-trace-heap-edge coding-trace-edge-line" x1="240" y1="32" x2="160" y2="104" /><g class="coding-trace-heap-node is-root" data-motion-key="heap-value-5-0"><circle cx="240" cy="32" r="21" /><text x="240" y="36">5</text></g><g class="coding-trace-heap-node" data-motion-key="heap-value-6-0"><circle cx="160" cy="104" r="21" /><text x="160" y="108">6</text></g></svg></div><div class="coding-trace-meta"><span><b>root</b>5</span><span><b>detail</b>kth largest</span><span><b>result</b>5</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Fill a size-2 heap</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Replace the weak root</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the root</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Keep only the largest k values; the smallest of those is the kth largest.</p></div><figcaption><strong>Read it this way:</strong> Read 3 and 2. Both remain candidates for the top two. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Min-heap of size `k`.

**Simple idea:** Keep only the largest `k` values seen. The smallest value in that group is
the `k`th largest overall.

```python
import heapq

def find_kth_largest(nums: list[int], k: int) -> int:
   if not 1 <= k <= len(nums):
      raise ValueError("k must name an item in nums")

   heap: list[int] = []
   for num in nums:
      heapq.heappush(heap, num)
      if len(heap) > k:
         heapq.heappop(heap)
   return heap[0]
```

**Cost:** $O(n\log k)$ time and $O(k)$ space.
