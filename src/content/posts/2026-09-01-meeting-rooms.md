---
title: "Meeting Rooms"
description: "Check whether one person can attend every meeting."
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

> Check whether one person can attend every meeting.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:meeting-rooms-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="meeting-rooms-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="meeting-rooms-state-title">Meeting Rooms: After sorting by start time, only the previous end can overlap the next meeting.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="meeting-rooms" role="group" tabindex="0" aria-label="Meeting Rooms: After sorting by start time, only the previous end can overlap the next meeting."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether one person can attend every meeting.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Sort meetings"><div class="coding-trace-frame-heading"><span>Sort meetings</span><strong>The starts are 0, 5, and 15.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row"><span>[0,30]</span><div class="coding-trace-track"><i class="trace-tone-state" style="--trace-start:0%;--trace-width:100%"></i></div></div><div class="coding-trace-interval-row"><span>[5,10]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:16.666666666666664%;--trace-width:16.666666666666664%"></i></div></div><div class="coding-trace-interval-row"><span>[15,20]</span><div class="coding-trace-track"><i class="" style="--trace-start:50%;--trace-width:16.666666666666664%"></i></div></div></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Find the overlap"><div class="coding-trace-frame-heading"><span>Find the overlap</span><strong>The next start 5 is before previous end 30.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row"><span>[0,30]</span><div class="coding-trace-track"><i class="trace-tone-warning" style="--trace-start:0%;--trace-width:100%"></i></div></div><div class="coding-trace-interval-row"><span>[5,10]</span><div class="coding-trace-track"><i class="trace-tone-focus" style="--trace-start:16.666666666666664%;--trace-width:16.666666666666664%"></i></div></div></div><div class="coding-trace-meta"><span><b>detail</b>5 &lt; 30</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Return false"><div class="coding-trace-frame-heading"><span>Return false</span><strong>One person cannot attend overlapping meetings.</strong></div><div class="coding-trace-intervals"><div class="coding-trace-interval-row"><span>[0,30]</span><div class="coding-trace-track"><i class="trace-tone-warning" style="--trace-start:0%;--trace-width:100%"></i></div></div><div class="coding-trace-interval-row"><span>[5,10]</span><div class="coding-trace-track"><i class="trace-tone-warning" style="--trace-start:16.666666666666664%;--trace-width:16.666666666666664%"></i></div></div></div><div class="coding-trace-meta"><span><b>result</b>false</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Sort meetings</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Find the overlap</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return false</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>After sorting by start time, only the previous end can overlap the next meeting.</p></div><figcaption><strong>Read it this way:</strong> The starts are 0, 5, and 15. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Sort intervals by start.

**Simple idea:** After sorting, only neighboring meetings can reveal the first overlap. Each
meeting must start at or after the previous meeting ends.

```python
def can_attend_meetings(intervals: list[list[int]]) -> bool:
   intervals.sort()
   return all(
      intervals[index - 1][1] <= intervals[index][0]
      for index in range(1, len(intervals))
   )
```

**Cost:** $O(n\log n)$ time and $O(1)$ extra space, not counting sorting.
