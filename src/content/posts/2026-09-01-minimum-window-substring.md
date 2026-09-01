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
<figure class="learning-figure coding-visual-figure" aria-labelledby="minimum-window-substring-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="minimum-window-substring-state-title">Minimum Window Substring: Grow until all required characters are present, then shrink while the window remains valid.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="minimum-window-substring" role="group" aria-label="Minimum Window Substring: Grow until all required characters are present, then shrink while the window remains valid."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the shortest substring that contains all required characters and counts.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Gather ABC"><div class="coding-trace-frame-heading"><span>Gather ABC</span><strong>ADOBEC contains A, B, and C, so the first valid window ends at C.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">L</span><span class="coding-trace-array-cell">A</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">D</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">O</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">E</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">R</span><span class="coding-trace-array-cell">C</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">O</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">D</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">E</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">N</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">C</span></span></div><div class="coding-trace-meta"><span><b>range</b>ADOBEC</span><span><b>need</b>A,B,C</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Shrink from the left"><div class="coding-trace-frame-heading"><span>Shrink from the left</span><strong>Dropping A breaks validity, so grow again until the new valid window is BANC.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">D</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">O</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">E</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">old valid</span><span class="coding-trace-array-cell">C</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">O</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">D</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">E</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">L</span><span class="coding-trace-array-cell">B</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">N</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">R</span><span class="coding-trace-array-cell">C</span></span></div><div class="coding-trace-meta"><span><b>range</b>BANC</span><span><b>action</b>shrink then regrow</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Keep the shortest"><div class="coding-trace-frame-heading"><span>Keep the shortest</span><strong>BANC is the shortest window containing A, B, and C.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">D</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">O</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">E</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">C</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">O</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">D</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">E</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">B</span><span class="coding-trace-array-cell">B</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">A</span><span class="coding-trace-array-cell">A</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">N</span><span class="coding-trace-array-cell">N</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">C</span><span class="coding-trace-array-cell">C</span></span></div><div class="coding-trace-meta"><span><b>result</b>BANC</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Gather ABC</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Shrink from the left</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Keep the shortest</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Grow until all required characters are present, then shrink while the window remains valid.</p></div><figcaption><strong>Read it this way:</strong> ADOBEC contains A, B, and C, so the first valid window ends at C. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
