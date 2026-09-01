---
title: "Combination Sum"
description: "Return combinations that add to a target. A value may be used more than once."
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

> Return combinations that add to a target. A value may be used more than once.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:combination-sum-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="combination-sum-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="combination-sum-state-title">Combination Sum: Choose in nondecreasing index order and carry the remaining target.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="combination-sum" role="group" aria-label="Combination Sum: Choose in nondecreasing index order and carry the remaining target."><div class="coding-visual-example"><span>Input and goal</span><strong>Return combinations that add to a target. A value may be used more than once.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Start with target 7"><div class="coding-trace-frame-heading"><span>Start with target 7</span><strong>The first choices are 2, 3, 6, or 7.</strong></div><div class="coding-trace-choices"><div class="coding-trace-choice-path"><span class="coding-trace-label">path</span><strong>empty</strong></div><div class="coding-trace-choice-branches"><span>2 (remain 5)</span><span>3 (remain 4)</span><span>6 (remain 1)</span><span>7 (remain 0)</span></div></div><div class="coding-trace-meta"><span><b>target</b>7</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Reuse a choice"><div class="coding-trace-frame-heading"><span>Reuse a choice</span><strong>From remainder 5, choosing 2 again leaves 3; [2,2,3] reaches zero.</strong></div><div class="coding-trace-choices"><div class="coding-trace-choice-path"><span class="coding-trace-label">path</span><strong>2 -&gt; 2</strong></div><div class="coding-trace-choice-branches"><span>choose 3 -&gt; remain 0</span><span>choose 6 -&gt; too large</span></div></div><div class="coding-trace-meta"><span><b>target</b>3</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Collect complete paths"><div class="coding-trace-frame-heading"><span>Collect complete paths</span><strong>The valid combinations are [2,2,3] and [7].</strong></div><div class="coding-trace-choices"><div class="coding-trace-choice-path"><span class="coding-trace-label">path</span><strong>empty</strong></div><div class="coding-trace-choice-branches"><span>[2,2,3] = 7</span><span>[7] = 7</span></div></div><div class="coding-trace-meta"><span><b>result</b>2 combinations</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Start with target 7</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Reuse a choice</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Collect complete paths</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Choose in nondecreasing index order and carry the remaining target.</p></div><figcaption><strong>Read it this way:</strong> The first choices are 2, 3, 6, or 7. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Backtracking with a remaining target.

**Simple idea:** Choose values in sorted index order. Reuse the same index when repeats are
allowed. Stop when the next value is larger than the remaining target.

```python
def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
   choices = sorted({num for num in candidates if num > 0})
   answer: list[list[int]] = []
   path: list[int] = []

   def choose(start: int, remaining: int) -> None:
      if remaining == 0:
         answer.append(path.copy())
         return

      for index in range(start, len(choices)):
         num = choices[index]
         if num > remaining:
            break
         path.append(num)
         choose(index, remaining - num)
         path.pop()

   choose(0, target)
   return answer
```

**Cost:** Exponential time in the worst case and $O(target)$ call-stack space when the
smallest choice is 1.
