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
<figure class="learning-figure coding-visual-figure" aria-labelledby="course-schedule-ii-state-title"><p class="visual-kicker">Dependencies becoming ready</p><p class="visual-title" id="course-schedule-ii-state-title">Course Schedule II: Remove prerequisites until the next zero-indegree item appears</p><div class="coding-visual coding-visual--topology" data-coding-visual data-coding-mode="topology" data-coding-slug="course-schedule-ii" role="group" aria-label="Course Schedule II: queue zero-indegree courses and append each one to the feasible order. The ready queue contains exactly the nodes whose prerequisites are complete."><div class="coding-visual-example"><span>Concrete trace</span><strong>queue zero-indegree courses and append each one to the feasible order</strong></div><div class="coding-visual-sketch coding-visual-sketch--topology"><div class="coding-sketch-row"><span class="coding-sketch-pill coding-sketch-pill--state">0 unmet</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--active">ready</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill">next is ready</span></div><p class="coding-sketch-note">remove incoming requirements until a node becomes ready</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Count</span><strong>incoming edges</strong><small>Record how many requirements each node still has.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Ready</span><strong>indegree = 0</strong><small>Only nodes with no unmet prerequisite can start.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Remove</span><strong>complete one</strong><small>Subtract its edge from every dependent node.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Detect</span><strong>cycle or order</strong><small>Unfinished nodes reveal a dependency cycle.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The ready queue contains exactly the nodes whose prerequisites are complete.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Imagine removing foundation blocks from a dependency wall. A block becomes available only when every incoming requirement has disappeared. For this problem, hold onto the concrete trace: queue zero-indegree courses and append each one to the feasible order.</figcaption></figure>

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
