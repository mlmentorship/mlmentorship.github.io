---
title: "Valid Anagram"
description: "Check whether two strings contain the same letters with the same counts."
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

> Check whether two strings contain the same letters with the same counts.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:valid-anagram-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="valid-anagram-state-title"><p class="visual-kicker">Memory as a shortcut</p><p class="visual-title" id="valid-anagram-state-title">Valid Anagram: Save the fact that makes the next item cheap</p><div class="coding-visual coding-visual--hash" data-coding-visual data-coding-mode="hash" data-coding-slug="valid-anagram" role="group" aria-label="Valid Anagram: eat and tea -&gt; both reduce to a:1, e:1, t:1. The state contains every useful fact from the prefix already processed."><div class="coding-visual-example"><span>Concrete trace</span><strong>eat and tea -&gt; both reduce to a:1, e:1, t:1</strong></div><div class="coding-visual-sketch coding-visual-sketch--hash"><div class="coding-sketch-row"><span class="coding-sketch-label">current</span><span class="coding-sketch-pill coding-sketch-pill--input">item</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-label">ask</span><span class="coding-sketch-pill coding-sketch-pill--focus">needed fact</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--state">saved state</span></div><p class="coding-sketch-note">read the concrete example above as the values flowing through this lookup</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Read</span><strong>one item</strong><small>The scan has a current value and a position.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Remember</span><strong>small state</strong><small>Store the fact a future item may need.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Ask</span><strong>lookup or difference</strong><small>Turn the target into a question about saved state.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Commit</span><strong>answer or update</strong><small>A hit completes the answer; otherwise save this item.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The state contains every useful fact from the prefix already processed.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Follow the scan from left to right. The data structure is a compressed memory of the past, so the current item never needs to rescan earlier items. For this problem, hold onto the concrete trace: eat and tea -&gt; both reduce to a:1, e:1, t:1.</figcaption></figure>

**Pattern:** Frequency map.

**Simple idea:** Anagrams have the same letter count. `Counter` builds that count map.

```python
from collections import Counter

def is_anagram(first: str, second: str) -> bool:
   return Counter(first) == Counter(second)
```

**Cost:** $O(n)$ time and $O(k)$ space, where $k$ is the number of different characters.
