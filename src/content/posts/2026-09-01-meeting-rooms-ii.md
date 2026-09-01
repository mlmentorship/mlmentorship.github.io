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
<figure class="learning-figure coding-visual-figure" aria-labelledby="meeting-rooms-ii-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="meeting-rooms-ii-state-title">Meeting Rooms II: At each start, remove rooms whose meetings have already ended.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="meeting-rooms-ii" role="group" aria-label="Meeting Rooms II: At each start, remove rooms whose meetings have already ended."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the smallest number of rooms needed for all meetings.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="First meeting"><div class="coding-trace-frame-heading"><span>First meeting</span><strong>Meeting [0,30] occupies one room.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row"><span>[0,30]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:0%;--trace-width:100%"></i></div></div></div><div class="coding-trace-meta"><span><b>rooms</b>1 active room</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Overlap needs another room"><div class="coding-trace-frame-heading"><span>Overlap needs another room</span><strong>At start 5, [0,30] is still active, so [5,10] uses room 2.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row"><span>[0,30]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:0%;--trace-width:100%"></i></div></div><div class="coding-trace-interval-row"><span>[5,10]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:16.666666666666664%;--trace-width:16.666666666666664%"></i></div></div></div><div class="coding-trace-meta"><span><b>rooms</b>2 active rooms</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Reuse after an end"><div class="coding-trace-frame-heading"><span>Reuse after an end</span><strong>At start 15, [5,10] is gone; the maximum active count was 2.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row"><span>[0,30]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:0%;--trace-width:100%"></i></div></div><div class="coding-trace-interval-row"><span>[15,20]</span><div class="coding-trace-track"><i class="trace-tone-output" style="--trace-start:50%;--trace-width:16.666666666666664%"></i></div></div></div><div class="coding-trace-meta"><span><b>result</b>2 rooms</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>First meeting</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Overlap needs another room</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Reuse after an end</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>At each start, remove rooms whose meetings have already ended.</p></div><figcaption><strong>Read it this way:</strong> Meeting [0,30] occupies one room. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
