---
title: "Meeting Rooms II"
description: "Find the smallest number of rooms needed for all meetings."
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

> Find the smallest number of rooms needed for all meetings.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:meeting-rooms-ii-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="meeting-rooms-ii-state-title"><p class="visual-kicker">A frontier ordered by value</p><p class="visual-title" id="meeting-rooms-ii-state-title">Meeting Rooms II: Keep the candidates that can still win</p><div class="coding-visual coding-visual--heap" data-coding-visual data-coding-mode="heap" data-coding-slug="meeting-rooms-ii" role="group" aria-label="Meeting Rooms II: start 1, start 2, end 3 -&gt; two active meetings need two rooms. The heap root is the next candidate whose priority is safe to process."><div class="coding-visual-example"><span>Concrete trace</span><strong>start 1, start 2, end 3 -&gt; two active meetings need two rooms</strong></div><div class="coding-visual-sketch coding-visual-sketch--heap"><div class="coding-sketch-tree"><span class="coding-sketch-node coding-sketch-node--active">root: next best</span><div class="coding-sketch-branch"><span class="coding-sketch-node">candidate</span><span class="coding-sketch-node">candidate</span></div></div><p class="coding-sketch-note">the root is exposed while the rest stays as a frontier</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Offer</span><strong>candidate set</strong><small>Put a new value into the frontier.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Expose</span><strong>root = next best</strong><small>The heap makes the smallest or largest current item visible.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Trim</span><strong>keep k</strong><small>Discard a candidate that cannot enter the answer.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Advance</span><strong>next candidate</strong><small>Replace the used item and continue the stream.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The heap root is the next candidate whose priority is safe to process.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The heap is not a sorted list. It exposes only the next useful item, while preserving enough frontier state to continue without sorting everything. For this problem, hold onto the concrete trace: start 1, start 2, end 3 -&gt; two active meetings need two rooms.</figcaption></figure>

**Pattern:** Sort starts and keep active end times in a heap.

**Simple idea:** Before starting a meeting, remove every meeting that already ended. Add the
new end time. The largest active heap size is the room count.

```python
import heapq

def min_meeting_rooms(intervals: list[list[int]]) -> int:
   end_times: list[int] = []
   most_rooms = 0

   for start, end in sorted(intervals):
      while end_times and end_times[0] <= start:
         heapq.heappop(end_times)
      heapq.heappush(end_times, end)
      most_rooms = max(most_rooms, len(end_times))
   return most_rooms
```

**Cost:** $O(n\log n)$ time and $O(n)$ space.
