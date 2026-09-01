---
title: "Permutation in String"
description: "Check whether any substring has the same letter counts as the pattern."
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

> Check whether any substring has the same letter counts as the pattern.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:permutation-in-string-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="permutation-in-string-state-title"><p class="visual-kicker">A moving range</p><p class="visual-title" id="permutation-in-string-state-title">Permutation in String: Grow until valid, then shrink until necessary</p><div class="coding-visual coding-visual--window" data-coding-visual data-coding-mode="window" data-coding-slug="permutation-in-string" role="group" aria-label="Permutation in String: s1=ab, s2=eidbaooo -&gt; compare each width-2 frequency window. The current window has exactly the state needed to decide whether it is valid."><div class="coding-visual-example"><span>Concrete trace</span><strong>s1=ab, s2=eidbaooo -&gt; compare each width-2 frequency window</strong></div><div class="coding-visual-sketch coding-visual-sketch--window"><div class="coding-sketch-array"><span class="coding-sketch-pointer">L</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-cell coding-sketch-cell--active">active</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-pointer">R</span></div><p class="coding-sketch-note">the active bracket grows for evidence and shrinks when its state is sufficient</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Extend</span><strong>L ... R</strong><small>Move the right edge to include new evidence.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Measure</span><strong>window state</strong><small>Update counts, sum, or the required matches.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Tighten</span><strong>advance L</strong><small>Remove the oldest item while validity survives.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Record</span><strong>best valid range</strong><small>Save the shortest, longest, or counted window.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The current window has exactly the state needed to decide whether it is valid.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The two edges are not guesses. The right edge gathers enough evidence; the left edge removes anything no longer needed, so each item enters and leaves once. For this problem, hold onto the concrete trace: s1=ab, s2=eidbaooo -&gt; compare each width-2 frequency window.</figcaption></figure>

**Pattern:** Fixed-size sliding window.

**Simple idea:** A matching substring must have the same length as the pattern. Keep letter
counts for one window of that size. Add the new right character and remove the old left
character.

```python
from collections import Counter

def check_inclusion(pattern: str, text: str) -> bool:
   if len(pattern) > len(text):
      return False

   need = Counter(pattern)
   window = Counter(text[: len(pattern)])
   if window == need:
      return True

   for right in range(len(pattern), len(text)):
      window[text[right]] += 1
      left_char = text[right - len(pattern)]
      window[left_char] -= 1
      if window[left_char] == 0:
         del window[left_char]
      if window == need:
         return True
   return False
```

**Cost:** $O(n)$ time for a fixed alphabet and $O(k)$ space.
