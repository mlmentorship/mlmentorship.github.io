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
<figure class="learning-figure coding-visual-figure" aria-labelledby="meeting-rooms-state-title"><p class="visual-kicker">Time ranges on one line</p><p class="visual-title" id="meeting-rooms-state-title">Meeting Rooms: Sorting makes the next possible conflict visible</p><div class="coding-visual coding-visual--interval" data-coding-visual data-coding-mode="interval" data-coding-slug="meeting-rooms" role="group" aria-label="Meeting Rooms: sort starts; if the next start is before the previous end, overlap exists. The saved boundary summarizes every interval that can still affect the next one."><div class="coding-visual-example"><span>Concrete trace</span><strong>sort starts; if the next start is before the previous end, overlap exists</strong></div><div class="coding-visual-sketch coding-visual-sketch--interval"><div class="coding-sketch-timeline"><span class="coding-sketch-tick">time</span><span class="coding-sketch-bar coding-sketch-bar--state">kept range</span><span class="coding-sketch-bar coding-sketch-bar--active">next range</span></div><p class="coding-sketch-note">sort first; carry the boundary that preserves future room</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Order</span><strong>start or end</strong><small>Put ranges in the order the proof needs.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Compare</span><strong>current boundary</strong><small>Check the next range against the active boundary.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Decide</span><strong>overlap?</strong><small>Merge, remove, or allocate a room.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Advance</span><strong>last safe end</strong><small>Carry the boundary that preserves future room.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The saved boundary summarizes every interval that can still affect the next one.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> See intervals as occupied segments, not pairs of unrelated numbers. The sort order turns a global overlap question into a local boundary comparison. For this problem, hold onto the concrete trace: sort starts; if the next start is before the previous end, overlap exists.</figcaption></figure>

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
