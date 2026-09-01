---
title: "Longest Substring Without Repeating Characters"
description: "Find the longest substring with no repeated character."
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

> Find the longest substring with no repeated character.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:longest-substring-without-repeating-characters-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="longest-substring-without-repeating-characters-state-title"><p class="visual-kicker">A moving range</p><p class="visual-title" id="longest-substring-without-repeating-characters-state-title">Longest Substring Without Repeating Characters: Grow until valid, then shrink until necessary</p><div class="coding-visual coding-visual--window" data-coding-visual data-coding-mode="window" data-coding-slug="longest-substring-without-repeating-characters" role="group" aria-label="Longest Substring Without Repeating Characters: abcabcbb -&gt; window abc, then move left past the old a. The current window has exactly the state needed to decide whether it is valid."><div class="coding-visual-example"><span>Concrete trace</span><strong>abcabcbb -&gt; window abc, then move left past the old a</strong></div><div class="coding-visual-sketch coding-visual-sketch--window"><div class="coding-sketch-array"><span class="coding-sketch-pointer">L</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-cell coding-sketch-cell--active">active</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-pointer">R</span></div><p class="coding-sketch-note">the active bracket grows for evidence and shrinks when its state is sufficient</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Extend</span><strong>L ... R</strong><small>Move the right edge to include new evidence.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Measure</span><strong>window state</strong><small>Update counts, sum, or the required matches.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Tighten</span><strong>advance L</strong><small>Remove the oldest item while validity survives.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Record</span><strong>best valid range</strong><small>Save the shortest, longest, or counted window.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The current window has exactly the state needed to decide whether it is valid.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The two edges are not guesses. The right edge gathers enough evidence; the left edge removes anything no longer needed, so each item enters and leaves once. For this problem, hold onto the concrete trace: abcabcbb -&gt; window abc, then move left past the old a.</figcaption></figure>

**Pattern:** Sliding window with last positions.

**Simple idea:** When a repeated character is inside the current window, move `left` to one
position after its last copy. Never move `left` backward.

```python
def length_of_longest_substring(text: str) -> int:
   last_seen: dict[str, int] = {}
   left = 0
   best = 0

   for right, char in enumerate(text):
      if char in last_seen:
         left = max(left, last_seen[char] + 1)
      last_seen[char] = right
      best = max(best, right - left + 1)

   return best
```

**Cost:** $O(n)$ time and $O(k)$ space.
