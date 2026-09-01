---
title: "Palindromic Substrings"
description: "Count every continuous palindrome in a string."
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

> Count every continuous palindrome in a string.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:palindromic-substrings-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="palindromic-substrings-state-title"><p class="visual-kicker">A moving range</p><p class="visual-title" id="palindromic-substrings-state-title">Palindromic Substrings: Grow until valid, then shrink until necessary</p><div class="coding-visual coding-visual--window" data-coding-visual data-coding-mode="window" data-coding-slug="palindromic-substrings" role="group" aria-label="Palindromic Substrings: each center contributes one count per successful outward expansion. The current window has exactly the state needed to decide whether it is valid."><div class="coding-visual-example"><span>Concrete trace</span><strong>each center contributes one count per successful outward expansion</strong></div><div class="coding-visual-sketch coding-visual-sketch--window"><div class="coding-sketch-array"><span class="coding-sketch-pointer">L</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-cell coding-sketch-cell--active">active</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-pointer">R</span></div><p class="coding-sketch-note">the active bracket grows for evidence and shrinks when its state is sufficient</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Extend</span><strong>L ... R</strong><small>Move the right edge to include new evidence.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Measure</span><strong>window state</strong><small>Update counts, sum, or the required matches.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Tighten</span><strong>advance L</strong><small>Remove the oldest item while validity survives.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Record</span><strong>best valid range</strong><small>Save the shortest, longest, or counted window.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The current window has exactly the state needed to decide whether it is valid.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The two edges are not guesses. The right edge gathers enough evidence; the left edge removes anything no longer needed, so each item enters and leaves once. For this problem, hold onto the concrete trace: each center contributes one count per successful outward expansion.</figcaption></figure>

**Pattern:** Expand from every center.

**Simple idea:** This is the same center rule as Longest Palindromic Substring. Count each
valid expansion instead of saving the longest one.

```python
def count_palindromic_substrings(text: str) -> int:
   def expand(left: int, right: int) -> int:
      count = 0
      while left >= 0 and right < len(text) and text[left] == text[right]:
         count += 1
         left -= 1
         right += 1
      return count

   return sum(
      expand(middle, middle) + expand(middle, middle + 1)
      for middle in range(len(text))
   )
```

**Cost:** $O(n^2)$ time and $O(1)$ space.
