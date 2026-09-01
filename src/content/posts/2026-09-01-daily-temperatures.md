---
title: "Daily Temperatures"
description: "For each day, find how many days pass before a warmer temperature."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Foundation"
priority: "Core"
aliases: []
prerequisites: []
---

> For each day, find how many days pass before a warmer temperature.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:daily-temperatures-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="daily-temperatures-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="daily-temperatures-state-title">Daily Temperatures: Keep colder days waiting until a warmer day resolves them.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="daily-temperatures" role="group" tabindex="0" aria-label="Daily Temperatures: Keep colder days waiting until a warmer day resolves them."><div class="coding-visual-example"><span>Input and goal</span><strong>For each day, find how many days pass before a warmer temperature.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Hold unresolved days"><div class="coding-trace-frame-heading"><span>Hold unresolved days</span><strong>After scanning through 69, days 2, 3, and 4 are still waiting for a warmer temperature.</strong></div><div class="coding-trace-stack-layout"><div class="coding-trace-stack-input"><span class="coding-trace-label">input</span><strong>73 74 75 71 69 72</strong></div><div class="coding-trace-stack-column"><span class="coding-trace-label">top</span><span class="coding-trace-stack-item">day 2: 75</span><span class="coding-trace-stack-item">day 3: 71</span><span class="coding-trace-stack-item is-top">day 4: 69</span></div></div><div class="coding-trace-meta"><span><b>current</b>72</span><span><b>action</b>wait</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Resolve from the top"><div class="coding-trace-frame-heading"><span>Resolve from the top</span><strong>72 is warmer than 69 and 71, so both waiting days receive distances. Day 2 remains.</strong></div><div class="coding-trace-stack-layout"><div class="coding-trace-stack-input"><span class="coding-trace-label">input</span><strong>73 74 75 71 69 72</strong></div><div class="coding-trace-stack-column"><span class="coding-trace-label">top</span><span class="coding-trace-stack-item is-top">day 2: 75</span></div></div><div class="coding-trace-meta"><span><b>current</b>72</span><span><b>action</b>resolve 69 -&gt; 1, 71 -&gt; 2</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Leave no warmer day as zero"><div class="coding-trace-frame-heading"><span>Leave no warmer day as zero</span><strong>75 stays in the stack because no later value is warmer.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="marker-none">none</span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-pointer" data-motion-key="marker-none">none</span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">5</small></span></div><div class="coding-trace-meta"><span><b>result</b>[1,1,0,2,1,0]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Hold unresolved days</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Resolve from the top</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Leave no warmer day as zero</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Keep colder days waiting until a warmer day resolves them.</p></div><figcaption><strong>Read it this way:</strong> After scanning through 69, days 2, 3, and 4 are still waiting for a warmer temperature. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Decreasing stack.

**Simple idea:** The stack holds days still waiting for something warmer. A warmer new day
finishes every colder day at the top.

```python
def daily_temperatures(temperatures: list[int]) -> list[int]:
   answer = [0] * len(temperatures)
   waiting: list[int] = []

   for day, temperature in enumerate(temperatures):
      while waiting and temperatures[waiting[-1]] < temperature:
         earlier_day = waiting.pop()
         answer[earlier_day] = day - earlier_day
      waiting.append(day)

   return answer
```

**Cost:** $O(n)$ time and $O(n)$ space. Each index enters and leaves the stack once.
