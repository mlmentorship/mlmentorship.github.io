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
<figure class="learning-figure coding-visual-figure" aria-labelledby="valid-anagram-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="valid-anagram-state-title">Valid Anagram: Compare letter counts, not letter positions.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="valid-anagram" role="group" tabindex="0" aria-label="Valid Anagram: Compare letter counts, not letter positions."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether two strings contain the same letters with the same counts.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Count the first word"><div class="coding-trace-frame-heading"><span>Count the first word</span><strong>eat contributes one e, one a, and one t.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">letter</th><th scope="col">eat</th><th scope="col">tea</th></tr></thead><tbody><tr><td class="">a</td><td class="is-active">1</td><td class="">-</td></tr><tr><td class="">e</td><td class="is-active">1</td><td class="">-</td></tr><tr><td class="">t</td><td class="is-active">1</td><td class="">-</td></tr></tbody></table></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Consume the second word"><div class="coding-trace-frame-heading"><span>Consume the second word</span><strong>tea removes the same three counts in a different order.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">letter</th><th scope="col">eat</th><th scope="col">tea</th></tr></thead><tbody><tr><td class="">a</td><td class="is-active">1</td><td class="is-active">1</td></tr><tr><td class="">e</td><td class="is-active">1</td><td class="is-active">1</td></tr><tr><td class="">t</td><td class="is-active">1</td><td class="is-active">1</td></tr></tbody></table></div><div class="coding-trace-meta"><span><b>status</b>all counts match</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Accept"><div class="coding-trace-frame-heading"><span>Accept</span><strong>Every count is equal, so the strings are anagrams.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">letter</th><th scope="col">left</th><th scope="col">right</th></tr></thead><tbody><tr><td class="">a</td><td class="">1</td><td class="">1</td></tr><tr><td class="">e</td><td class="">1</td><td class="">1</td></tr><tr><td class="">t</td><td class="">1</td><td class="">1</td></tr></tbody></table></div><div class="coding-trace-meta"><span><b>status</b>true</span><span><b>result</b>true</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Count the first word</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Consume the second word</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Accept</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Compare letter counts, not letter positions.</p></div><figcaption><strong>Read it this way:</strong> eat contributes one e, one a, and one t. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Frequency map.

**Simple idea:** Anagrams have the same letter count. `Counter` builds that count map.

```python
from collections import Counter

def is_anagram(first: str, second: str) -> bool:
   return Counter(first) == Counter(second)
```

**Cost:** $O(n)$ time and $O(k)$ space, where $k$ is the number of different characters.
