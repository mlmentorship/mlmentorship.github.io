---
title: "Find Median From Data Stream"
description: "Add numbers one at a time and return the current median."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Advanced"
priority: "Specialist"
aliases: []
prerequisites: []
---

> Add numbers one at a time and return the current median.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:find-median-from-data-stream-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="find-median-from-data-stream-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="find-median-from-data-stream-state-title">Find Median From Data Stream: Keep the lower half in a max-heap and the upper half in a min-heap.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="find-median-from-data-stream" role="group" tabindex="0" aria-label="Find Median From Data Stream: Keep the lower half in a max-heap and the upper half in a min-heap."><div class="coding-visual-example"><span>Input and goal</span><strong>Add numbers one at a time and return the current median.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Add 1"><div class="coding-trace-frame-heading"><span>Add 1</span><strong>The lower half contains 1; the upper half is empty.</strong></div><div class="coding-trace-heap"><svg viewBox="0 0 480 144" role="img" aria-label="Complete binary heap topology"><line class="coding-trace-heap-edge coding-trace-edge-line" x1="240" y1="32" x2="160" y2="104" /><g class="coding-trace-heap-node is-root" data-motion-key="heap-value-lower max:1-0"><circle cx="240" cy="32" r="21" /><text x="240" y="36">lower max:1</text></g><g class="coding-trace-heap-node" data-motion-key="heap-value-upper min:--0"><circle cx="160" cy="104" r="21" /><text x="160" y="108">upper min:-</text></g></svg></div><div class="coding-trace-meta"><span><b>root</b>lower 1</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Add 2"><div class="coding-trace-frame-heading"><span>Add 2</span><strong>Balance the halves: lower has 1 and upper has 2.</strong></div><div class="coding-trace-heap"><svg viewBox="0 0 480 144" role="img" aria-label="Complete binary heap topology"><line class="coding-trace-heap-edge coding-trace-edge-line" x1="240" y1="32" x2="160" y2="104" /><g class="coding-trace-heap-node is-root" data-motion-key="heap-value-lower max:1-0"><circle cx="240" cy="32" r="21" /><text x="240" y="36">lower max:1</text></g><g class="coding-trace-heap-node" data-motion-key="heap-value-upper min:2-0"><circle cx="160" cy="104" r="21" /><text x="160" y="108">upper min:2</text></g></svg></div><div class="coding-trace-meta"><span><b>detail</b>two roots bracket the median</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Read the middle"><div class="coding-trace-frame-heading"><span>Read the middle</span><strong>With two values, median is (1+2)/2 = 1.5.</strong></div><div class="coding-trace-heap"><svg viewBox="0 0 480 144" role="img" aria-label="Complete binary heap topology"><line class="coding-trace-heap-edge coding-trace-edge-line" x1="240" y1="32" x2="160" y2="104" /><g class="coding-trace-heap-node is-root" data-motion-key="heap-value-lower max:1-0"><circle cx="240" cy="32" r="21" /><text x="240" y="36">lower max:1</text></g><g class="coding-trace-heap-node" data-motion-key="heap-value-upper min:2-0"><circle cx="160" cy="104" r="21" /><text x="160" y="108">upper min:2</text></g></svg></div><div class="coding-trace-meta"><span><b>result</b>1.5</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Add 1</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Add 2</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Read the middle</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Keep the lower half in a max-heap and the upper half in a min-heap.</p></div><figcaption><strong>Read it this way:</strong> The lower half contains 1; the upper half is empty. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Two heaps.

**Simple idea:** A max-heap stores the lower half and a min-heap stores the upper half. Keep
the lower half the same size as the upper half or one item larger. The middle value or
values
are then at the heap roots.

Python has only a min-heap, so negative values create the max-heap.

```python
import heapq

class MedianFinder:
   def __init__(self) -> None:
      self.lower: list[int] = []
      self.upper: list[int] = []

   def add_num(self, num: int) -> None:
      heapq.heappush(self.lower, -num)
      heapq.heappush(self.upper, -heapq.heappop(self.lower))
      if len(self.upper) > len(self.lower):
         heapq.heappush(self.lower, -heapq.heappop(self.upper))

   def find_median(self) -> float:
      if len(self.lower) > len(self.upper):
         return float(-self.lower[0])
      return (-self.lower[0] + self.upper[0]) / 2
```

**Cost:** Adding takes $O(\log n)$ time. Finding the median takes $O(1)$ time. Space is
$O(n)$.
