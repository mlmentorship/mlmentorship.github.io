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
<figure class="learning-figure coding-visual-figure" aria-labelledby="permutations-state-title"><p class="visual-kicker">A tree of choices</p><p class="visual-title" id="permutations-state-title">Permutations: Choose, explore, then undo the exact choice</p><div class="coding-visual coding-visual--backtrack" data-coding-visual data-coding-mode="backtrack" data-coding-slug="permutations" role="group" aria-label="Permutations: for [1,2,3], choose one unused value for each next position. At every call, the path contains exactly the choices on the route from the root."><div class="coding-visual-example"><span>Concrete trace</span><strong>for [1,2,3], choose one unused value for each next position</strong></div><div class="coding-visual-sketch coding-visual-sketch--backtrack"><div class="coding-sketch-choice-tree"><span class="coding-sketch-node coding-sketch-node--active">partial path</span><div class="coding-sketch-choice-branches"><span class="coding-sketch-node">choose A</span><span class="coding-sketch-node">choose B</span><span class="coding-sketch-node">choose C</span></div></div><p class="coding-sketch-note">add one choice, explore below it, then restore the parent path</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Path</span><strong>partial answer</strong><small>The current path is a valid unfinished choice.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Choose</span><strong>one branch</strong><small>Add one available value, cell, or letter.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Recurse</span><strong>smaller problem</strong><small>Explore everything below that choice.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Undo</span><strong>restore state</strong><small>Remove the same choice before the next branch.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>At every call, the path contains exactly the choices on the route from the root.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The visual is a choice tree, not a list of magic loops. Backtracking works because every branch starts from the same restored parent state. For this problem, hold onto the concrete trace: for [1,2,3], choose one unused value for each next position.</figcaption></figure>

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
