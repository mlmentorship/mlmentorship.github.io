---
title: "Longest Repeating Character Replacement"
description: "Replace at most `k` letters so the longest possible substring has one repeated letter."
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

> Replace at most `k` letters so the longest possible substring has one repeated letter.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:longest-repeating-character-replacement-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="longest-repeating-character-replacement-state-title"><p class="visual-kicker">A moving range</p><p class="visual-title" id="longest-repeating-character-replacement-state-title">Longest Repeating Character Replacement: Grow until valid, then shrink until necessary</p><div class="coding-visual coding-visual--window" data-coding-visual data-coding-mode="window" data-coding-slug="longest-repeating-character-replacement" role="group" aria-label="Longest Repeating Character Replacement: AABABBA, k=1 -&gt; window is valid when length - max_count &lt;= 1. The current window has exactly the state needed to decide whether it is valid."><div class="coding-visual-example"><span>Concrete trace</span><strong>AABABBA, k=1 -&gt; window is valid when length - max_count &lt;= 1</strong></div><div class="coding-visual-sketch coding-visual-sketch--window"><div class="coding-sketch-array"><span class="coding-sketch-pointer">L</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-cell coding-sketch-cell--active">active</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-pointer">R</span></div><p class="coding-sketch-note">the active bracket grows for evidence and shrinks when its state is sufficient</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Extend</span><strong>L ... R</strong><small>Move the right edge to include new evidence.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Measure</span><strong>window state</strong><small>Update counts, sum, or the required matches.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Tighten</span><strong>advance L</strong><small>Remove the oldest item while validity survives.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Record</span><strong>best valid range</strong><small>Save the shortest, longest, or counted window.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The current window has exactly the state needed to decide whether it is valid.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The two edges are not guesses. The right edge gathers enough evidence; the left edge removes anything no longer needed, so each item enters and leaves once. For this problem, hold onto the concrete trace: AABABBA, k=1 -&gt; window is valid when length - max_count &lt;= 1.</figcaption></figure>

**Pattern:** Sliding window with counts.

**Simple idea:** Keep the most common letter. Every other letter in the window needs one
replacement. The window is valid when:

`window length - largest letter count <= k`

```python
from collections import defaultdict

def character_replacement(text: str, replacements: int) -> int:
   counts: dict[str, int] = defaultdict(int)
   left = 0
   largest_count = 0
   best = 0

   for right, char in enumerate(text):
      counts[char] += 1
      largest_count = max(largest_count, counts[char])

      while right - left + 1 - largest_count > replacements:
         counts[text[left]] -= 1
         left += 1

      best = max(best, right - left + 1)

   return best
```

**Cost:** $O(n)$ time and $O(k)$ space.
