---
title: "Palindromic Substrings"
description: "Count every continuous palindrome in a string."
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

> Count every continuous palindrome in a string.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:palindromic-substrings-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="palindromic-substrings-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="palindromic-substrings-state-title">Palindromic Substrings: Each successful center expansion contributes exactly one palindrome.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="palindromic-substrings" role="group" tabindex="0" aria-label="Palindromic Substrings: Each successful center expansion contributes exactly one palindrome."><div class="coding-visual-example"><span>Input and goal</span><strong>Count every continuous palindrome in a string.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Count an odd center"><div class="coding-trace-frame-heading"><span>Count an odd center</span><strong>Center a gives a, then expand to aba.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-palindrome">palindrome</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="marker-center">center</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="marker-palindrome">palindrome</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>count</b>2</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Count every center"><div class="coding-trace-frame-heading"><span>Count every center</span><strong>For aaa, three single letters, two pairs, and aaa all count.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-1">1</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="marker-4">4</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="marker-1">1</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>count</b>6 total</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Return the total"><div class="coding-trace-frame-heading"><span>Return the total</span><strong>The six palindromic substrings of aaa are all center expansions.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-a">a</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="marker-a-aa">a/aa</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="marker-a">a</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>result</b>6</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Count an odd center</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Count every center</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the total</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Each successful center expansion contributes exactly one palindrome.</p></div><figcaption><strong>Read it this way:</strong> Center a gives a, then expand to aba. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Expand from every center.

**Simple idea:** This is the same center rule as Longest Palindromic Substring. Count each
valid expansion instead of saving the longest one.

```python
def count_palindromic_substrings(text: str) -> int:
   def expand(left: int, right: int) -> int:
      count = 0
      while left >= 0 and right < len(text) and text[left] == text[right]:
         count += 1
         left -= 1
         right += 1
      return count

   return sum(
      expand(middle, middle) + expand(middle, middle + 1)
      for middle in range(len(text))
   )
```

**Cost:** $O(n^2)$ time and $O(1)$ space.
