---
title: "Course Schedule II"
description: "Return one valid order for completing every course."
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

> Return one valid order for completing every course.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:course-schedule-ii-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="course-schedule-ii-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="course-schedule-ii-state-title">Course Schedule II: The topological queue is also the feasible course order.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="course-schedule-ii" role="group" tabindex="0" aria-label="Course Schedule II: The topological queue is also the feasible course order."><div class="coding-visual-example"><span>Input and goal</span><strong>Return one valid order for completing every course.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Seed zero-indegree courses"><div class="coding-trace-frame-heading"><span>Seed zero-indegree courses</span><strong>Only course 0 has no unmet prerequisite.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><g data-motion-key="edge-0 -&gt; 1-0"><line class="coding-trace-edge-line" x1="240" y1="37" x2="356.9134295108992" y2="154" /><text x="298.45671475544964" y="89.5">0 -&gt; 1</text></g><g data-motion-key="edge-1 -&gt; 2-1"><line class="coding-trace-edge-line" x1="356.9134295108992" y1="154" x2="123.08657048910081" y2="154.00000000000003" /><text x="240" y="148">1 -&gt; 2</text></g><g class="coding-trace-graph-node" data-motion-key="node-0-0"><circle cx="240" cy="37" r="23" /><text x="240" y="41">0</text></g><g class="coding-trace-graph-node" data-motion-key="node-1-1"><circle cx="356.9134295108992" cy="154" r="23" /><text x="356.9134295108992" y="158">1</text></g><g class="coding-trace-graph-node" data-motion-key="node-2-2"><circle cx="123.08657048910081" cy="154.00000000000003" r="23" /><text x="123.08657048910081" y="158.00000000000003">2</text></g></svg><div class="coding-trace-meta"><span><b>ready</b>0</span><span><b>indegree</b>0:0, 1:1, 2:1</span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Append and decrement"><div class="coding-trace-frame-heading"><span>Append and decrement</span><strong>Taking 0 makes 1 ready; taking 1 makes 2 ready.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><g data-motion-key="edge-0 -&gt; 1-0"><line class="coding-trace-edge-line" x1="240" y1="37" x2="356.9134295108992" y2="154" /><text x="298.45671475544964" y="89.5">0 -&gt; 1</text></g><g data-motion-key="edge-1 -&gt; 2-1"><line class="coding-trace-edge-line" x1="356.9134295108992" y1="154" x2="123.08657048910081" y2="154.00000000000003" /><text x="240" y="148">1 -&gt; 2</text></g><g class="coding-trace-graph-node" data-motion-key="node-0-0"><circle cx="240" cy="37" r="23" /><text x="240" y="41">0</text></g><g class="coding-trace-graph-node" data-motion-key="node-1-1"><circle cx="356.9134295108992" cy="154" r="23" /><text x="356.9134295108992" y="158">1</text></g><g class="coding-trace-graph-node" data-motion-key="node-2-2"><circle cx="123.08657048910081" cy="154.00000000000003" r="23" /><text x="123.08657048910081" y="158.00000000000003">2</text></g></svg><div class="coding-trace-meta"><span><b>ready</b>2</span><span><b>order</b>0, 1</span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Return the order"><div class="coding-trace-frame-heading"><span>Return the order</span><strong>The queue emitted a valid prerequisite-respecting sequence.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 480 230" role="img" aria-label="Connected graph topology"><g data-motion-key="edge-0 -&gt; 1-0"><line class="coding-trace-edge-line" x1="240" y1="37" x2="356.9134295108992" y2="154" /><text x="298.45671475544964" y="89.5">0 -&gt; 1</text></g><g data-motion-key="edge-1 -&gt; 2-1"><line class="coding-trace-edge-line" x1="356.9134295108992" y1="154" x2="123.08657048910081" y2="154.00000000000003" /><text x="240" y="148">1 -&gt; 2</text></g><g class="coding-trace-graph-node" data-motion-key="node-0-0"><circle cx="240" cy="37" r="23" /><text x="240" y="41">0</text></g><g class="coding-trace-graph-node" data-motion-key="node-1-1"><circle cx="356.9134295108992" cy="154" r="23" /><text x="356.9134295108992" y="158">1</text></g><g class="coding-trace-graph-node" data-motion-key="node-2-2"><circle cx="123.08657048910081" cy="154.00000000000003" r="23" /><text x="123.08657048910081" y="158.00000000000003">2</text></g></svg><div class="coding-trace-meta"><span><b>order</b>0, 1, 2</span></div></div><div class="coding-trace-meta"><span><b>result</b>[0,1,2]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Seed zero-indegree courses</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Append and decrement</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the order</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>The topological queue is also the feasible course order.</p></div><figcaption><strong>Read it this way:</strong> Only course 0 has no unmet prerequisite. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Topological sort that saves the order.

**Simple idea:** This is Course Schedule with one extra action. Append each ready course to
the answer. If a cycle prevents some courses from becoming ready, return an empty list.

```python
from collections import deque

def find_order(course_count: int, prerequisites: list[list[int]]) -> list[int]:
   graph = [[] for _ in range(course_count)]
   indegree = [0] * course_count

   for course, prerequisite in prerequisites:
      graph[prerequisite].append(course)
      indegree[course] += 1

   ready = deque(course for course in range(course_count) if indegree[course] == 0)
   order = []
   while ready:
      course = ready.popleft()
      order.append(course)
      for next_course in graph[course]:
         indegree[next_course] -= 1
         if indegree[next_course] == 0:
            ready.append(next_course)

   return order if len(order) == course_count else []
```

**Cost:** $O(V + E)$ time and space.
