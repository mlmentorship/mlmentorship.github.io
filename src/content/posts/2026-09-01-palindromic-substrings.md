---
title: "Palindromic Substrings"
description: "Count each accepted center radius as a separate index range, then distinguish occurrence lists from unique palindrome text values."
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

Count one range for every accepted radius around a character or gap center. Equal text at different positions remains separate occurrences.

## Problem

> Count every continuous palindrome in a string.

Count nonempty contiguous **occurrences**, identified by their boundaries. Empty input returns zero.
Equality is exact Python code-point equality, without normalization or grapheme grouping.

## Picture

Reuse [center expansion](/questions/longest-palindromic-substring/). Only aggregation changes: add one for each accepted radius instead of retaining a longest range.

For `aaa`, report half-open ranges:

| `middle` | Odd-center ranges | Even-center ranges | New occurrences | Total |
| --- | --- | --- | --- | --- |
| `0` | `[0,1)` is `a` | `[0,2)` is `aa` | `2` | `2` |
| `1` | `[1,2)` is `a`; `[0,3)` is `aaa` | `[1,3)` is `aa` | `3` | `5` |
| `2` | `[2,3)` is `a` | None: right endpoint is outside | `1` | `6` |

Every palindromic range has one center and radius, so contributions are disjoint.
The two `aa` ranges differ by position; a text set would incorrectly merge them for this task.

Inside `expand`, `left/right` are inclusive candidate endpoints. The guard certifies a palindrome before `count += 1`.
Moving outward creates the next unchecked candidate.
Each call returns its local count. The outer `sum` combines both center types lazily, without storing per-center results.

<!-- visual:palindromic-substrings-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="palindromic-substrings-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="palindromic-substrings-state-title">Palindromic Substrings: Each accepted radius contributes one range. Equal text at different positions counts separately.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="palindromic-substrings" role="group" tabindex="0" aria-label="Palindromic Substrings: Each accepted radius contributes one range. Equal text at different positions counts separately."><div class="coding-visual-example"><span>Input and goal</span><strong>Count every continuous palindrome in a string.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize" role="group" aria-label="Initialize center sum"><div class="coding-trace-frame-heading"><span>Initialize center sum</span><strong>For aaa, the lazy sum has not consumed a center yet. Its accumulated total is zero.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:3" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="middle">m</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>input</b>aaa</span><span><b>runningTotal</b>0</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="middle-0" hidden role="group" aria-label="Count middle 0"><div class="coding-trace-frame-heading"><span>Count middle 0</span><strong>Middle zero accepts [0,1) and [0,2). Two new occurrences increase the total from zero to two.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:3" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="left">L</span><span class="coding-trace-array-pointer" data-motion-key="middle">m</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="right">R</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>input</b>aaa</span><span><b>oddExpansion</b>[0,1) = a</span><span><b>evenExpansion</b>[0,2) = aa</span><span><b>contribution</b>1 odd + 1 even</span><span><b>runningTotal</b>2</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="middle-1" hidden role="group" aria-label="Count middle 1"><div class="coding-trace-frame-heading"><span>Count middle 1</span><strong>Middle one accepts [1,2), [0,3) and [1,3). Three more occurrences raise the total to five.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:3" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="left">L</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="middle">m</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="right">R</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>input</b>aaa</span><span><b>oddExpansion</b>[1,2) = a, [0,3) = aaa</span><span><b>evenExpansion</b>[1,3) = aa</span><span><b>contribution</b>2 odd + 1 even</span><span><b>runningTotal</b>5</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="middle-2" hidden role="group" aria-label="Count middle 2 and return"><div class="coding-trace-frame-heading"><span>Count middle 2 and return</span><strong>Middle two accepts [2,3). The even call starts outside the string. Total six includes three separate singleton positions.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:3" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-state trace-tone-output trace-tone-focus" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="left">L</span><span class="coding-trace-array-pointer" data-motion-key="right">R</span><span class="coding-trace-array-pointer" data-motion-key="middle">m</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>input</b>aaa</span><span><b>oddExpansion</b>[2,3) = a</span><span><b>evenExpansion</b>none: right=3 is outside</span><span><b>contribution</b>1 odd + 0 even</span><span><b>runningTotal</b>6</span><span><b>result</b>6</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 4</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize center sum</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Count middle 0</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Count middle 1</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Count middle 2 and return</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">The task counts contiguous palindrome occurrences, including equal text at different positions, so each center-radius pair should contribute separately.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Each successful while iteration adds one new palindrome for that center and then expands one cell outward. Completed centers contribute disjoint occurrences because every palindrome has one unique center.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Map each combinatorial object to a unique center and radius, then count every successful expansion once; the same enumeration underlies longest-palindrome and palindrome-radius algorithms.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> For aaa, the lazy sum has not consumed a center yet. Its accumulated total is zero. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

## Code

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

**Cost:** $O(n^2)$ worst-case comparisons and a constant number of auxiliary scalar slots.
The count is at most $n(n+1)/2$; scalar bit widths grow with $n$.
The source copies no substrings inside its loops.

<!-- learning:core-end -->

## Change the Output Identity

Start from `count_palindromic_substrings`. Keep both center types and their equality guard; change what an accepted radius saves.

1. **Return occurrence ranges.**

   **New task:** Inspect or highlight every palindrome's boundaries.

   **Picture and change:** Append `[left,right+1]` instead of incrementing a scalar. Different centers remain distinct occurrences.

   **Code:** Complete replacement, sorted by boundaries:

   ```python
   def palindrome_occurrences(text):
      result = []
      for middle in range(len(text)):
         for left, right in ((middle, middle), (middle, middle + 1)):
            while left >= 0 and right < len(text) and text[left] == text[right]:
               result.append([left, right + 1])
               left -= 1
               right += 1
      return sorted(result)
   ```

   **Check:** `aaa` returns `[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]`.

   **Use when:** Positions are required.
   For $p$ occurrences, stored ranges take $O(p)$ slots and sorting adds $O(p\log(p+1))$ work beyond expansion.

2. **Count unique text values.**

   **New task:** Equal palindrome text must count once, regardless of position.

   **Picture and change:** Insert the accepted slice into a set. This changes answer identity, not the center enumeration.

   **Code:** Complete replacement:

   ```python
   def distinct_palindrome_count(text):
      distinct = set()
      for middle in range(len(text)):
         for left, right in ((middle, middle), (middle, middle + 1)):
            while left >= 0 and right < len(text) and text[left] == text[right]:
               distinct.add(text[left:right + 1])
               left -= 1
               right += 1
      return len(distinct)
   ```

   **Check:** `aaa` retains only `a`, `aa`, `aaa`, so returns three. Empty input returns zero.

   **Use when:** Unique content matters rather than occurrence positions.
   Copying and hashing every accepted slice can take $O(n^3)$ total work. The set also retains text, so storage is not constant.

## Rebuild the Counting Proof

Can two different centers produce the same occurrence range?

**Answer:** No. Its endpoints determine one midpoint and parity. Different centers can produce equal text at different positions.
