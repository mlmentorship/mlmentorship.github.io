---
title: "Permutations"
description: "Return every possible ordering of the input values."
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

> Return every possible ordering of the input values.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:permutations-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="permutations-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="permutations-state-title">Permutations: Fill one position with each unused value, then undo it for the next branch.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="permutations" role="group" tabindex="0" aria-label="Permutations: Fill one position with each unused value, then undo it for the next branch."><div class="coding-visual-example"><span>Input and goal</span><strong>Return every possible ordering of the input values.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Choose the first position"><div class="coding-trace-frame-heading"><span>Choose the first position</span><strong>For [1,2,3], any of the three values can start the path.</strong></div><div class="coding-trace-choices"><div class="coding-trace-choice-path"><span class="coding-trace-label">path</span><strong>empty</strong></div><div class="coding-trace-choice-branches"><span>1__</span><span>2__</span><span>3__</span></div></div><div class="coding-trace-meta"><span><b>used</b>none</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Choose below 1"><div class="coding-trace-frame-heading"><span>Choose below 1</span><strong>After choosing 1, only 2 and 3 remain for the next position.</strong></div><div class="coding-trace-choices"><div class="coding-trace-choice-path"><span class="coding-trace-label">path</span><strong>1</strong></div><div class="coding-trace-choice-branches"><span>12_</span><span>13_</span></div></div><div class="coding-trace-meta"><span><b>used</b>1</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Reach complete leaves"><div class="coding-trace-frame-heading"><span>Reach complete leaves</span><strong>The tree ends at all six orderings.</strong></div><div class="coding-trace-choices"><div class="coding-trace-choice-path"><span class="coding-trace-label">path</span><strong>empty</strong></div><div class="coding-trace-choice-branches"><span>123</span><span>132</span><span>213</span><span>231</span><span>312</span><span>321</span></div></div><div class="coding-trace-meta"><span><b>result</b>6 permutations</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Choose the first position</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Choose below 1</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Reach complete leaves</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Fill one position with each unused value, then undo it for the next branch.</p></div><figcaption><strong>Read it this way:</strong> For [1,2,3], any of the three values can start the path. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Backtracking with a used list.

**Simple idea:** At each position, try every value that is not already in the path. Remove it
after that branch finishes.

```python
def permutations(nums: list[int]) -> list[list[int]]:
   answer: list[list[int]] = []
   path: list[int] = []
   used = [False] * len(nums)

   def choose() -> None:
      if len(path) == len(nums):
         answer.append(path.copy())
         return

      for index, num in enumerate(nums):
         if not used[index]:
            used[index] = True
            path.append(num)
            choose()
            path.pop()
            used[index] = False

   choose()
   return answer
```

**Cost:** $O(n \times n!)$ time and $O(n)$ working space.
