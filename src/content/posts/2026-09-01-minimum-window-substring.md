---
title: "Minimum Window Substring"
description: "Find the shortest substring that contains all required characters and counts."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Advanced"
priority: "Specialist"
aliases: []
prerequisites: []
---

> Find the shortest substring that contains all required characters and counts.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:minimum-window-substring-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="minimum-window-substring-state-title"><p class="visual-kicker">A moving range</p><p class="visual-title" id="minimum-window-substring-state-title">Minimum Window Substring: Grow until valid, then shrink until necessary</p><div class="coding-visual coding-visual--window" data-coding-visual data-coding-mode="window" data-coding-slug="minimum-window-substring" role="group" aria-label="Minimum Window Substring: ADOBECODEBANC needs ABC -&gt; expand to BANC, then shrink from the left. The current window has exactly the state needed to decide whether it is valid."><div class="coding-visual-example"><span>Concrete trace</span><strong>ADOBECODEBANC needs ABC -&gt; expand to BANC, then shrink from the left</strong></div><div class="coding-visual-sketch coding-visual-sketch--window"><div class="coding-sketch-array"><span class="coding-sketch-pointer">L</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-cell coding-sketch-cell--active">active</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-pointer">R</span></div><p class="coding-sketch-note">the active bracket grows for evidence and shrinks when its state is sufficient</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Extend</span><strong>L ... R</strong><small>Move the right edge to include new evidence.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Measure</span><strong>window state</strong><small>Update counts, sum, or the required matches.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Tighten</span><strong>advance L</strong><small>Remove the oldest item while validity survives.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Record</span><strong>best valid range</strong><small>Save the shortest, longest, or counted window.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The current window has exactly the state needed to decide whether it is valid.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The two edges are not guesses. The right edge gathers enough evidence; the left edge removes anything no longer needed, so each item enters and leaves once. For this problem, hold onto the concrete trace: ADOBECODEBANC needs ABC -&gt; expand to BANC, then shrink from the left.</figcaption></figure>

**Pattern:** Grow, then shrink a sliding window.

**Simple idea:** Move the right side until every required character is present. Then move
the left side while the window remains valid. Save the shortest valid window.

```python
from collections import Counter

def min_window(text: str, required: str) -> str:
   if not required:
      return ""

   need = Counter(required)
   missing = len(required)
   left = 0
   best_start = 0
   best_length = len(text) + 1

   for right, char in enumerate(text):
      if need[char] > 0:
         missing -= 1
      need[char] -= 1

      while missing == 0:
         length = right - left + 1
         if length < best_length:
            best_start, best_length = left, length

         left_char = text[left]
         need[left_char] += 1
         if need[left_char] > 0:
            missing += 1
         left += 1

   if best_length > len(text):
      return ""
   return text[best_start : best_start + best_length]
```

**Cost:** $O(n)$ time and $O(k)$ space.
