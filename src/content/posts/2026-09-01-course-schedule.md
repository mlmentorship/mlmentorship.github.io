---
title: "Course Schedule"
description: "Check whether all courses can be completed."
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

> Check whether all courses can be completed.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:course-schedule-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="course-schedule-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="course-schedule-state-title">Course Schedule: A course becomes ready when its remaining prerequisite count reaches zero.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="course-schedule" role="group" tabindex="0" aria-label="Course Schedule: A course becomes ready when its remaining prerequisite count reaches zero."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether all courses can be completed.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Count prerequisites"><div class="coding-trace-frame-heading"><span>Count prerequisites</span><strong>Course 0 is ready; course 1 has one incoming edge.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><g data-motion-key="edge-0 -&gt; 1-0"><line class="coding-trace-edge-line" x1="240" y1="37" x2="240" y2="193" /><text x="240" y="109">0 -&gt; 1</text></g><g class="coding-trace-graph-node" data-motion-key="node-course 0-0"><circle cx="240" cy="37" r="23" /><text x="240" y="41">course 0</text></g><g class="coding-trace-graph-node" data-motion-key="node-course 1-1"><circle cx="240" cy="193" r="23" /><text x="240" y="197">course 1</text></g></svg><div class="coding-trace-meta"><span><b>ready</b>0</span><span><b>indegree</b>0:0, 1:1</span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Complete a ready course"><div class="coding-trace-frame-heading"><span>Complete a ready course</span><strong>Removing course 0 decrements course 1 from 1 to 0.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><g data-motion-key="edge-0 -&gt; 1-0"><line class="coding-trace-edge-line" x1="240" y1="37" x2="240" y2="193" /><text x="240" y="109">0 -&gt; 1</text></g><g class="coding-trace-graph-node" data-motion-key="node-course 0-0"><circle cx="240" cy="37" r="23" /><text x="240" y="41">course 0</text></g><g class="coding-trace-graph-node" data-motion-key="node-course 1-1"><circle cx="240" cy="193" r="23" /><text x="240" y="197">course 1</text></g></svg><div class="coding-trace-meta"><span><b>ready</b>1</span><span><b>indegree</b>0:done, 1:0</span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Finish all nodes"><div class="coding-trace-frame-heading"><span>Finish all nodes</span><strong>Every course entered the ready queue, so no cycle remains.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><g data-motion-key="edge-0 -&gt; 1-0"><line class="coding-trace-edge-line" x1="240" y1="37" x2="240" y2="193" /><text x="240" y="109">0 -&gt; 1</text></g><g class="coding-trace-graph-node" data-motion-key="node-course 0-0"><circle cx="240" cy="37" r="23" /><text x="240" y="41">course 0</text></g><g class="coding-trace-graph-node" data-motion-key="node-course 1-1"><circle cx="240" cy="193" r="23" /><text x="240" y="197">course 1</text></g></svg><div class="coding-trace-meta"><span><b>order</b>0, 1</span></div></div><div class="coding-trace-meta"><span><b>result</b>true</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Count prerequisites</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Complete a ready course</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Finish all nodes</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>A course becomes ready when its remaining prerequisite count reaches zero.</p></div><figcaption><strong>Read it this way:</strong> Course 0 is ready; course 1 has one incoming edge. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Topological sort with requirement counts.

**Simple idea:** Start with courses that need nothing. Completing one course removes one
requirement from every course that follows it.

```python
from collections import deque

def can_finish(course_count: int, prerequisites: list[list[int]]) -> bool:
   graph = [[] for _ in range(course_count)]
   indegree = [0] * course_count

   for course, prerequisite in prerequisites:
      graph[prerequisite].append(course)
      indegree[course] += 1

   ready = deque(course for course in range(course_count) if indegree[course] == 0)
   completed = 0

   while ready:
      course = ready.popleft()
      completed += 1
      for next_course in graph[course]:
         indegree[next_course] -= 1
         if indegree[next_course] == 0:
            ready.append(next_course)

   return completed == course_count
```

**Cost:** $O(V + E)$ time and space.
