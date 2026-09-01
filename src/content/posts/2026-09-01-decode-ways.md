---
title: "Decode Ways"
description: "Count ways to decode digits where `1` through `26` map to letters."
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

> Count ways to decode digits where `1` through `26` map to letters.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:decode-ways-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="decode-ways-state-title"><p class="visual-kicker">A small state graph</p><p class="visual-title" id="decode-ways-state-title">Decode Ways: Keep the complete answer for each smaller state</p><div class="coding-visual coding-visual--dp" data-coding-visual data-coding-mode="dp" data-coding-slug="decode-ways" role="group" aria-label="Decode Ways: 226 -&gt; 2|2|6, 22|6, and 2|26 are the valid paths. Each saved state is the complete answer for its prefix, amount, cell, or pair of prefixes."><div class="coding-visual-example"><span>Concrete trace</span><strong>226 -&gt; 2|2|6, 22|6, and 2|26 are the valid paths</strong></div><div class="coding-visual-sketch coding-visual-sketch--dp"><div class="coding-sketch-dp-grid"><span class="coding-sketch-cell coding-sketch-cell--state">base</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell">smaller</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell coding-sketch-cell--active">current</span></div><p class="coding-sketch-note">each cell is a complete answer to one smaller question</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Base</span><strong>known state</strong><small>Initialize the smallest solvable problem.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Read</span><strong>earlier answers</strong><small>Look only at states the transition depends on.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Build</span><strong>current state</strong><small>Choose, count, or combine those answers.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Compress</span><strong>rolling memory</strong><small>Discard old states that no future step needs.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Each saved state is the complete answer for its prefix, amount, cell, or pair of prefixes.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Treat the table as a map of smaller questions. The recurrence is the arrow between states; space optimization is safe only after the dependencies are visible. For this problem, hold onto the concrete trace: 226 -&gt; 2|2|6, 22|6, and 2|26 are the valid paths.</figcaption></figure>

**Pattern:** DP with one-digit and two-digit choices.

**Simple idea:** A nonzero current digit can extend every decoding from one position back.
A valid two-digit number from 10 through 26 can extend every decoding from two positions
back.

```python
def num_decodings(text: str) -> int:
   if not text or text[0] == "0":
      return 0

   two_back = one_back = 1
   for index in range(1, len(text)):
      current = one_back if text[index] != "0" else 0
      if 10 <= int(text[index - 1 : index + 1]) <= 26:
         current += two_back
      two_back, one_back = one_back, current
   return one_back
```

**Cost:** $O(n)$ time and $O(1)$ space.
