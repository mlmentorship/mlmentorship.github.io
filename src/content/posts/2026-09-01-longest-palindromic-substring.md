---
title: "Longest Palindromic Substring"
description: "Return the longest continuous palindrome in a string."
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

> Return the longest continuous palindrome in a string.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:longest-palindromic-substring-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="longest-palindromic-substring-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="longest-palindromic-substring-state-title">Longest Palindromic Substring: Every palindrome grows from one character center or one gap center.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="longest-palindromic-substring" role="group" tabindex="0" aria-label="Longest Palindromic Substring: Every palindrome grows from one character center or one gap center."><div class="coding-visual-example"><span>Input and goal</span><strong>Return the longest continuous palindrome in a string.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Try an odd center"><div class="coding-trace-frame-heading"><span>Try an odd center</span><strong>Expand around b in babad to get bab.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-edge">edge</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="marker-center">center</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="marker-edge">edge</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">d</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>candidate</b>bab</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Try an even center"><div class="coding-trace-frame-heading"><span>Try an even center</span><strong>A gap between two equal characters handles even-length palindromes.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">c</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="marker-gap">gap</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="marker-gap">gap</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">d</span><small class="coding-trace-array-index">3</small></span></div><div class="coding-trace-meta"><span><b>candidate</b>bb</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Keep the longest"><div class="coding-trace-frame-heading"><span>Keep the longest</span><strong>The widest expansion wins.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-best">best</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="marker-best">best</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="marker-best">best</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">d</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>result</b>bab or aba</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Try an odd center</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Try an even center</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Keep the longest</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Every palindrome grows from one character center or one gap center.</p></div><figcaption><strong>Read it this way:</strong> Expand around b in babad to get bab. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Expand from every center.

**Simple idea:** Every palindrome has one center character or a gap between two center
characters. Expand both forms at every position and save the longest range.

```python
def longest_palindrome(text: str) -> str:
   best_left = best_right = 0

   def expand(left: int, right: int) -> None:
      nonlocal best_left, best_right
      while left >= 0 and right < len(text) and text[left] == text[right]:
         if right - left > best_right - best_left:
            best_left, best_right = left, right
         left -= 1
         right += 1

   for middle in range(len(text)):
      expand(middle, middle)
      expand(middle, middle + 1)
   return text[best_left : best_right + 1]
```

**Cost:** $O(n^2)$ time and $O(1)$ space.
