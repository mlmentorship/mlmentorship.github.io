---
title: "Group Anagrams"
description: "Put words with the same letters into the same group."
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

> Put words with the same letters into the same group.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:group-anagrams-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="group-anagrams-state-title"><p class="visual-kicker">Memory as a shortcut</p><p class="visual-title" id="group-anagrams-state-title">Group Anagrams: Save the fact that makes the next item cheap</p><div class="coding-visual coding-visual--hash" data-coding-visual data-coding-mode="hash" data-coding-slug="group-anagrams" role="group" aria-label="Group Anagrams: eat, tea, tan -&gt; [a,e,t] shares one bucket; tan uses [a,n,t]. The state contains every useful fact from the prefix already processed."><div class="coding-visual-example"><span>Concrete trace</span><strong>eat, tea, tan -&gt; [a,e,t] shares one bucket; tan uses [a,n,t]</strong></div><div class="coding-visual-sketch coding-visual-sketch--hash"><div class="coding-sketch-row"><span class="coding-sketch-label">current</span><span class="coding-sketch-pill coding-sketch-pill--input">item</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-label">ask</span><span class="coding-sketch-pill coding-sketch-pill--focus">needed fact</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--state">saved state</span></div><p class="coding-sketch-note">read the concrete example above as the values flowing through this lookup</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Read</span><strong>one item</strong><small>The scan has a current value and a position.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Remember</span><strong>small state</strong><small>Store the fact a future item may need.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Ask</span><strong>lookup or difference</strong><small>Turn the target into a question about saved state.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Commit</span><strong>answer or update</strong><small>A hit completes the answer; otherwise save this item.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The state contains every useful fact from the prefix already processed.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Follow the scan from left to right. The data structure is a compressed memory of the past, so the current item never needs to rescan earlier items. For this problem, hold onto the concrete trace: eat, tea, tan -&gt; [a,e,t] shares one bucket; tan uses [a,n,t].</figcaption></figure>

**Pattern:** Hash map with a shared key.

**Simple idea:** Count each lowercase letter. Anagrams have the same 26 counts, so they use
the same tuple as a map key.

```python
from collections import defaultdict

def group_anagrams(words: list[str]) -> list[list[str]]:
   groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
   for word in words:
      counts = [0] * 26
      for char in word:
         counts[ord(char) - ord("a")] += 1
      groups[tuple(counts)].append(word)
   return list(groups.values())
```

**Cost:** $O(nm)$ time and $O(nm)$ space for $n$ words of average length $m$.
