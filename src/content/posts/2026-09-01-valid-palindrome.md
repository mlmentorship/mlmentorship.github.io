---
title: "Valid Palindrome"
description: "Ignore punctuation and letter case, then check whether text reads the same both ways."
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

> Ignore punctuation and letter case, then check whether text reads the same both ways.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:valid-palindrome-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="valid-palindrome-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="valid-palindrome-state-title">Valid Palindrome: Move inward while comparing the next alphanumeric character from each end.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="valid-palindrome" role="group" aria-label="Valid Palindrome: Move inward while comparing the next alphanumeric character from each end."><div class="coding-visual-example"><span>Input and goal</span><strong>Ignore punctuation and letter case, then check whether text reads the same both ways.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Skip punctuation"><div class="coding-trace-frame-heading"><span>Skip punctuation</span><strong>Ignore spaces and commas; the meaningful endpoints are A and a.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">L</span><span class="coding-trace-array-cell">A</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">m</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">n</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">m</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">R</span><span class="coding-trace-array-cell">a</span></span></div><div class="coding-trace-meta"><span><b>normalize</b>lowercase, alphanumeric</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Compare inward"><div class="coding-trace-frame-heading"><span>Compare inward</span><strong>Matching pairs move both pointers toward the center.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">match</span><span class="coding-trace-array-cell">m</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">n</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">match</span><span class="coding-trace-array-cell">m</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span></span></div><div class="coding-trace-meta"><span><b>detail</b>m == m</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Meet in the middle"><div class="coding-trace-frame-heading"><span>Meet in the middle</span><strong>Every pair matches, so the normalized string is a palindrome.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">m</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">center</span><span class="coding-trace-array-cell">n</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">m</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span></span></div><div class="coding-trace-meta"><span><b>result</b>true</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Skip punctuation</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Compare inward</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Meet in the middle</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Move inward while comparing the next alphanumeric character from each end.</p></div><figcaption><strong>Read it this way:</strong> Ignore spaces and commas; the meaningful endpoints are A and a. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Two pointers at opposite ends.

**Simple idea:** Skip non-letter and non-number characters. Compare the next real character
from each side, then move inward.

```python
def valid_palindrome(text: str) -> bool:
   left, right = 0, len(text) - 1

   while left < right:
      if not text[left].isalnum():
         left += 1
      elif not text[right].isalnum():
         right -= 1
      elif text[left].lower() != text[right].lower():
         return False
      else:
         left += 1
         right -= 1
   return True
```

**Cost:** $O(n)$ time and $O(1)$ space.
