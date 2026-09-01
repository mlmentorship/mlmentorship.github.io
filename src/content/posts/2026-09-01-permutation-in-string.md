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
<figure class="learning-figure coding-visual-figure" aria-labelledby="permutation-in-string-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="permutation-in-string-state-title">Permutation in String: Compare each fixed-width window with the pattern counts.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="permutation-in-string" role="group" aria-label="Permutation in String: Compare each fixed-width window with the pattern counts."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether any substring has the same letter counts as the pattern.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Build the pattern count"><div class="coding-trace-frame-heading"><span>Build the pattern count</span><strong>The pattern ab needs one a and one b. The first text window ei has neither.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">window</th><th scope="col">a</th><th scope="col">b</th></tr></thead><tbody><tr><td class="is-active">ab</td><td class="is-active">1</td><td class="is-active">1</td></tr><tr><td class="">ei</td><td class="">0</td><td class="">0</td></tr></tbody></table></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Slide to a candidate"><div class="coding-trace-frame-heading"><span>Slide to a candidate</span><strong>The window ba has the same counts as ab, even though the order differs.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">window</th><th scope="col">a</th><th scope="col">b</th></tr></thead><tbody><tr><td class="">ab</td><td class="">1</td><td class="">1</td></tr><tr><td class="is-active">ba</td><td class="is-active">1</td><td class="is-active">1</td></tr></tbody></table></div><div class="coding-trace-meta"><span><b>status</b>match</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Return true"><div class="coding-trace-frame-heading"><span>Return true</span><strong>A matching count window means a permutation appears in the text.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">e</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">i</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">d</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">window</span><span class="coding-trace-array-cell">b</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">window</span><span class="coding-trace-array-cell">a</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">o</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">o</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">o</span></span></div><div class="coding-trace-meta"><span><b>result</b>true</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Build the pattern count</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Slide to a candidate</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return true</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Compare each fixed-width window with the pattern counts.</p></div><figcaption><strong>Read it this way:</strong> The pattern ab needs one a and one b. The first text window ei has neither. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
