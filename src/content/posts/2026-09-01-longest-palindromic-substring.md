---
title: "Longest Palindromic Substring"
description: "Return the longest continuous palindrome in a string."
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

> Return the longest continuous palindrome in a string.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:longest-palindromic-substring-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="longest-palindromic-substring-state-title"><p class="visual-kicker">A moving range</p><p class="visual-title" id="longest-palindromic-substring-state-title">Longest Palindromic Substring: Grow until valid, then shrink until necessary</p><div class="coding-visual coding-visual--window" data-coding-visual data-coding-mode="window" data-coding-slug="longest-palindromic-substring" role="group" aria-label="Longest Palindromic Substring: expand around one letter and one gap; keep the widest match. The current window has exactly the state needed to decide whether it is valid."><div class="coding-visual-example"><span>Concrete trace</span><strong>expand around one letter and one gap; keep the widest match</strong></div><div class="coding-visual-sketch coding-visual-sketch--window"><div class="coding-sketch-array"><span class="coding-sketch-pointer">L</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-cell coding-sketch-cell--active">active</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-pointer">R</span></div><p class="coding-sketch-note">the active bracket grows for evidence and shrinks when its state is sufficient</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Extend</span><strong>L ... R</strong><small>Move the right edge to include new evidence.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Measure</span><strong>window state</strong><small>Update counts, sum, or the required matches.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Tighten</span><strong>advance L</strong><small>Remove the oldest item while validity survives.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Record</span><strong>best valid range</strong><small>Save the shortest, longest, or counted window.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The current window has exactly the state needed to decide whether it is valid.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The two edges are not guesses. The right edge gathers enough evidence; the left edge removes anything no longer needed, so each item enters and leaves once. For this problem, hold onto the concrete trace: expand around one letter and one gap; keep the widest match.</figcaption></figure>

**Pattern:** Expand from every center.

**Simple idea:** Every palindrome has one center character or a gap between two center
characters. Expand both forms at every position and save the longest range.

```python
def longest_palindrome(text: str) -> str:
   best_left = best_right = 0

   def expand(left: int, right: int) -> None:
      nonlocal best_left, best_right
      while left >= 0 and right < len(text) and text[left] == text[right]:
         if right - left > best_right - best_left:
            best_left, best_right = left, right
         left -= 1
         right += 1

   for middle in range(len(text)):
      expand(middle, middle)
      expand(middle, middle + 1)
   return text[best_left : best_right + 1]
```

**Cost:** $O(n^2)$ time and $O(1)$ space.
