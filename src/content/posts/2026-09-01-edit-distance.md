---
title: "Edit Distance"
description: "Find the fewest insert, delete, or replace steps needed to change one string into another."
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

> Find the fewest insert, delete, or replace steps needed to change one string into another.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:edit-distance-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="edit-distance-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="edit-distance-state-title">Edit Distance: Each mismatch chooses the cheapest of insert, delete, and replace.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="edit-distance" role="group" tabindex="0" aria-label="Edit Distance: Each mismatch chooses the cheapest of insert, delete, and replace."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the fewest insert, delete, or replace steps needed to change one string into another.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Initialize empty prefixes"><div class="coding-trace-frame-heading"><span>Initialize empty prefixes</span><strong>The first row and column count edits against an empty string; the interior is not solved yet.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col"></th><th scope="col">0</th><th scope="col">r</th><th scope="col">o</th><th scope="col">s</th></tr></thead><tbody><tr><td class="is-active">0</td><td class="">0</td><td class="">1</td><td class="">2</td><td class="">3</td></tr><tr><td class="">h</td><td class="">1</td><td class="">?</td><td class="">?</td><td class="">?</td></tr><tr><td class="">o</td><td class="">2</td><td class="">?</td><td class="">?</td><td class="">?</td></tr><tr><td class="">r</td><td class="">3</td><td class="">?</td><td class="">?</td><td class="">?</td></tr><tr><td class="">s</td><td class="">4</td><td class="">?</td><td class="">?</td><td class="">?</td></tr><tr><td class="">e</td><td class="">5</td><td class="">?</td><td class="">?</td><td class="">?</td></tr></tbody></table></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Choose a local operation"><div class="coding-trace-frame-heading"><span>Choose a local operation</span><strong>At the final e/s mismatch, the cell is 1 plus the smallest neighbor.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col"></th><th scope="col">0</th><th scope="col">r</th><th scope="col">o</th><th scope="col">s</th></tr></thead><tbody><tr><td class="">0</td><td class="">0</td><td class="">1</td><td class="">2</td><td class="">3</td></tr><tr><td class="">h</td><td class="">1</td><td class="">1</td><td class="">2</td><td class="">3</td></tr><tr><td class="">o</td><td class="">2</td><td class="">2</td><td class="">1</td><td class="">2</td></tr><tr><td class="">r</td><td class="">3</td><td class="">2</td><td class="">2</td><td class="">2</td></tr><tr><td class="">s</td><td class="">4</td><td class="">3</td><td class="">3</td><td class="">2</td></tr><tr><td class="">e</td><td class="">5</td><td class="">4</td><td class="">4</td><td class="is-active">3</td></tr></tbody></table></div><div class="coding-trace-meta"><span><b>action</b>min(insert, delete, replace) + 1</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Read the final cost"><div class="coding-trace-frame-heading"><span>Read the final cost</span><strong>The bottom-right cell gives the distance from horse to ros.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col"></th><th scope="col">0</th><th scope="col">r</th><th scope="col">o</th><th scope="col">s</th></tr></thead><tbody><tr><td class="">0</td><td class="">0</td><td class="">1</td><td class="">2</td><td class="">3</td></tr><tr><td class="">h</td><td class="">1</td><td class="">1</td><td class="">2</td><td class="">3</td></tr><tr><td class="">o</td><td class="">2</td><td class="">2</td><td class="">1</td><td class="">2</td></tr><tr><td class="">r</td><td class="">3</td><td class="">2</td><td class="">2</td><td class="">2</td></tr><tr><td class="">s</td><td class="">4</td><td class="">3</td><td class="">3</td><td class="">2</td></tr><tr><td class="">e</td><td class="">5</td><td class="">4</td><td class="">4</td><td class="is-active">3</td></tr></tbody></table></div><div class="coding-trace-meta"><span><b>result</b>3</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize empty prefixes</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Choose a local operation</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Read the final cost</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Each mismatch chooses the cheapest of insert, delete, and replace.</p></div><figcaption><strong>Read it this way:</strong> The first row and column count edits against an empty string; the interior is not solved yet. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Grid DP stored as two rows.

**State:** The fewest edits between two prefixes.

**Simple idea:** Equal characters need no new edit. Different characters try insert, delete,
and replace, then add one to the smallest earlier answer.

```python
def edit_distance(first: str, second: str) -> int:
   previous = list(range(len(second) + 1))

   for first_index, first_char in enumerate(first, 1):
      current = [first_index]
      for second_index, second_char in enumerate(second, 1):
         if first_char == second_char:
            current.append(previous[second_index - 1])
         else:
            insert = current[-1]
            delete = previous[second_index]
            replace = previous[second_index - 1]
            current.append(1 + min(insert, delete, replace))
      previous = current

   return previous[-1]
```

**Cost:** $O(mn)$ time and $O(n)$ space.
