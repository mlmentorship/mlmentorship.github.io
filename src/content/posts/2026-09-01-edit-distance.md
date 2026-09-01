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
<figure class="learning-figure coding-visual-figure" aria-labelledby="edit-distance-state-title"><p class="visual-kicker">A small state graph</p><p class="visual-title" id="edit-distance-state-title">Edit Distance: Keep the complete answer for each smaller state</p><div class="coding-visual coding-visual--dp" data-coding-visual data-coding-mode="dp" data-coding-slug="edit-distance" role="group" aria-label="Edit Distance: horse -&gt; ros; each grid cell chooses insert, delete, or replace. Each saved state is the complete answer for its prefix, amount, cell, or pair of prefixes."><div class="coding-visual-example"><span>Concrete trace</span><strong>horse -&gt; ros; each grid cell chooses insert, delete, or replace</strong></div><div class="coding-visual-sketch coding-visual-sketch--dp"><div class="coding-sketch-dp-grid"><span class="coding-sketch-cell coding-sketch-cell--state">base</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell">smaller</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell coding-sketch-cell--active">current</span></div><p class="coding-sketch-note">each cell is a complete answer to one smaller question</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Base</span><strong>known state</strong><small>Initialize the smallest solvable problem.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Read</span><strong>earlier answers</strong><small>Look only at states the transition depends on.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Build</span><strong>current state</strong><small>Choose, count, or combine those answers.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Compress</span><strong>rolling memory</strong><small>Discard old states that no future step needs.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Each saved state is the complete answer for its prefix, amount, cell, or pair of prefixes.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Treat the table as a map of smaller questions. The recurrence is the arrow between states; space optimization is safe only after the dependencies are visible. For this problem, hold onto the concrete trace: horse -&gt; ros; each grid cell chooses insert, delete, or replace.</figcaption></figure>

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
