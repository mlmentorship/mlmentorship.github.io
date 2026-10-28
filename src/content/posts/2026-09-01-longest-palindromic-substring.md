---
title: "Longest Palindromic Substring"
description: "Expand matching endpoints around character and gap centers, retain the first longest range, and adapt the result to ties or repeated interval queries."
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

Return a longest contiguous piece of a string that reads identically forward and backward.

## Problem

**Input.** A Python string `text`.

**Output.** Return a longest palindromic substring, not merely its length.

**Rules.** The selected characters must be contiguous; skipping intervening characters is not allowed.
Use exact code-point equality, without case conversion, Unicode normalization, or grapheme grouping.
Any longest palindrome is valid; the reference retains its first equal-length answer.
Leave the string unchanged. Empty input returns `""`.

**Example.** `"babad"` permits `"bab"` or `"aba"`; the reference returns `"bab"`.
For `"cbbd"`, return `"bb"`.

## Picture

An odd palindrome has a character center. An even palindrome has a gap center:

| Text | Center | Accepted inclusive ranges | Stop |
| --- | --- | --- | --- |
| `babad` | Character one | `[1,1]` is `a`, then `[0,2]` is `bab` | Next radius exceeds bounds |
| `babad` | Character two | `[2,2]` is `b`, then `[1,3]` is `aba` | Next endpoints `b` and `d` differ |
| `cbbd` | Gap between one and two | `[1,2]` is `bb` | Outer `c` and `d` differ |

Every nonempty palindromic range has exactly one midpoint and radius. Trying both center types therefore covers every answer.
Matching endpoints extend an already-palindromic inner range. A singleton or empty gap supplies the first inner case.
On mismatch, every larger radius still contains that unequal pair. Bounds failure also forbids any larger range at that center.

`middle` selects a center. `left` and `right` are inclusive candidate endpoints.
The `while` guard certifies the current candidate **before** saving it. Moving outward creates an unchecked candidate, not another proven palindrome.

`best_left` and `best_right` retain the longest accepted range across completed centers.
The code compares `right-left`, which is length minus one. Strict `>` saves `bab` and keeps it when `aba` ties.

In the visual, `m` marks the center and `aL/aR` mark accepted odd endpoints.
The underline preserves the best half-open range. Final `bL/bR` mark the saved return range, not the last center's `d`.

<!-- visual:longest-palindromic-substring-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="longest-palindromic-substring-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="longest-palindromic-substring-state-title">Longest Palindromic Substring: Expand every character center and every gap center. Save only a strictly longer accepted range.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="longest-palindromic-substring" role="group" tabindex="0" aria-label="Longest Palindromic Substring: Expand every character center and every gap center. Save only a strictly longer accepted range."><div class="coding-visual-example"><span>Input and goal</span><strong>Return the longest continuous palindrome in a string.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize" role="group" aria-label="Initialize best range"><div class="coding-trace-frame-heading"><span>Initialize best range</span><strong>For babad, save inclusive [0,0], the first b. The underline uses half-open range [0,1).</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state is-saved-range" role="listitem" data-motion-key="value-0" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="middle">m</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">d</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>input</b>babad</span><span><b>bestRange</b>[0,1) = b</span><span><b>evenContrast</b>cbbd: gap at 1/2 accepts [1,3) = bb</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>middle</dt><dd><strong class="coding-state-after">0</strong></dd></div><div class="coding-state-change"><dt>input</dt><dd><strong class="coding-state-after">babad</strong></dd></div><div class="coding-state-change"><dt>best range</dt><dd><strong class="coding-state-after">[0,1) = b</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="middle-0" hidden role="group" aria-label="Process middle 0"><div class="coding-trace-frame-heading"><span>Process middle 0</span><strong>Odd center zero accepts b. Even center zero rejects b versus a. The saved range does not change.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state trace-tone-focus is-saved-range" role="listitem" data-motion-key="value-0" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="left">aL</span><span class="coding-trace-array-pointer" data-motion-key="middle">m</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">d</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>input</b>babad</span><span><b>calls</b>expand(0,0); expand(0,1)</span><span><b>acceptedOdd</b>[0,1)</span><span><b>expansions</b>odd b; even b!=a</span><span><b>bestRange</b>[0,1) = b</span><span><b>evenContrast</b>cbbd: gap at 1/2 accepts [1,3) = bb</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>left</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0</strong></dd></div><div class="coding-state-change"><dt>calls</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">expand(0,0); expand(0,1)</strong></dd></div><div class="coding-state-change"><dt>accepted odd</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[0,1)</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="middle-1" hidden role="group" aria-label="Process middle 1"><div class="coding-trace-frame-heading"><span>Process middle 1</span><strong>Odd center one accepts a, then bab. Save inclusive [0,2]. Its next radius is outside the string.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state is-saved-range" role="listitem" data-motion-key="value-0" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="left">aL</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus is-saved-range" role="listitem" data-motion-key="value-1" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="middle">m</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus is-saved-range" role="listitem" data-motion-key="value-2" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="right">aR</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">d</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>input</b>babad</span><span><b>calls</b>expand(1,1); expand(1,2)</span><span><b>acceptedOdd</b>[0,3)</span><span><b>expansions</b>odd a, then bab, then bounds stop; even a!=b</span><span><b>bestRange</b>[0,3) = bab</span><span><b>evenContrast</b>cbbd: gap at 1/2 accepts [1,3) = bb</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>right</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2</strong></dd></div><div class="coding-state-change"><dt>middle</dt><dd><span class="coding-state-before">0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>calls</dt><dd><span class="coding-state-before">expand(0,0); expand(0,1)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">expand(1,1); expand(1,2)</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="middle-2" hidden role="group" aria-label="Process middle 2"><div class="coding-trace-frame-heading"><span>Process middle 2</span><strong>Odd center two accepts b, then aba. Equal width does not replace bab. The next endpoints b and d disagree.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-0" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state is-saved-range" role="listitem" data-motion-key="value-1" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="left">aL</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus is-saved-range" role="listitem" data-motion-key="value-2" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="middle">m</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="right">aR</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">d</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>input</b>babad</span><span><b>calls</b>expand(2,2); expand(2,3)</span><span><b>acceptedOdd</b>[1,4)</span><span><b>expansions</b>odd b, then aba, then b!=d; even b!=a</span><span><b>bestRange</b>[0,3) = bab</span><span><b>evenContrast</b>cbbd: gap at 1/2 accepts [1,3) = bb</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>left</dt><dd><span class="coding-state-before">0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>right</dt><dd><span class="coding-state-before">2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>middle</dt><dd><span class="coding-state-before">1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="middle-3" hidden role="group" aria-label="Process middle 3"><div class="coding-trace-frame-heading"><span>Process middle 3</span><strong>Odd center three accepts a. Its next endpoints b and d disagree. Even endpoints a and d also disagree.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-0" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-1" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-2" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-state trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="left">aL</span><span class="coding-trace-array-pointer" data-motion-key="middle">m</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">d</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>input</b>babad</span><span><b>calls</b>expand(3,3); expand(3,4)</span><span><b>acceptedOdd</b>[3,4)</span><span><b>expansions</b>odd a, then b!=d; even a!=d</span><span><b>bestRange</b>[0,3) = bab</span><span><b>evenContrast</b>cbbd: gap at 1/2 accepts [1,3) = bb</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>left</dt><dd><span class="coding-state-before">1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>middle</dt><dd><span class="coding-state-before">2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3</strong></dd></div><div class="coding-state-change"><dt>right</dt><dd><span class="coding-state-before">3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="middle-4" hidden role="group" aria-label="Process middle 4 and return"><div class="coding-trace-frame-heading"><span>Process middle 4 and return</span><strong>Odd center four accepts d. Its next radius and the even call exceed the boundary. Return saved bab, not d.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output is-saved-range" role="listitem" data-motion-key="value-0" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="left">bL</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-1" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-output is-saved-range" role="listitem" data-motion-key="value-2" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="right">bR</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-pointer" data-motion-key="middle">m</span><span class="coding-trace-array-cell">d</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>input</b>babad</span><span><b>calls</b>expand(4,4); expand(4,5)</span><span><b>acceptedOdd</b>[4,5) = d</span><span><b>expansions</b>odd d, then boundary; even right=5 outside</span><span><b>bestRange</b>[0,3) = bab</span><span><b>evenContrast</b>cbbd: gap at 1/2 accepts [1,3) = bb</span><span><b>result</b>bab</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>left</dt><dd><span class="coding-state-before">3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0</strong></dd></div><div class="coding-state-change"><dt>right</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2</strong></dd></div><div class="coding-state-change"><dt>middle</dt><dd><span class="coding-state-before">3</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">4</strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 6</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize best range</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Process middle 0</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Process middle 1</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Process middle 2</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Process middle 3</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Process middle 4 and return</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">The answer is a contiguous palindrome, whose symmetry guarantees one unique odd character center or even gap center.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">A successful guard certifies text[left:right+1] before updates. After the pointers move, the next candidate is untested. After each center, the saved range is a longest accepted range so far.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">When validity is symmetric around a center, enumerate both center types and expand until the invariant breaks; change only the aggregation to find a longest value, count all values, or validate radii.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> For babad, save inclusive [0,0], the first b. The underline uses half-open range [0,1). Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="longest-palindromic-substring-state"><strong>Check:</strong> Why does the final center at d not replace bab? Why does equal-length aba leave the saved answer unchanged?</span></figcaption></figure>

## Code

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

**Cost:** $O(n^2)$ worst-case comparisons and $O(1)$ auxiliary pointer/count slots.
The returned slice allocates $O(L)$ space for answer length $L$.
On empty input, slicing returns `""`; initialized `[0,0]` does not imply an existing character.

<!-- learning:core-end -->

## Change the Saved Result or the Workload

Start from `longest_palindrome`. Change either tie retention or the need to answer many range queries.

1. **Keep every longest occurrence.**

   **New task:** Return all maximum-length half-open ranges, not only the first copied text.

   **Picture and change:** Reset saved ranges on a longer candidate. Append on an equal length; unique centers prevent duplicate ranges.

   **Code:** Complete replacement:

   ```python
   def longest_palindrome_ranges(text):
      ranges, best_length = [], 0
      for middle in range(len(text)):
         for left, right in ((middle, middle), (middle, middle + 1)):
            while left >= 0 and right < len(text) and text[left] == text[right]:
               length = right - left + 1
               if length > best_length:
                  ranges, best_length = [[left, right + 1]], length
               elif length == best_length:
                  ranges.append([left, right + 1])
               left -= 1
               right += 1
      return sorted(ranges)
   ```

   **Check:** `babad` returns `[[0,3],[1,4]]`; `cbbd` returns `[[1,3]]`. Empty input returns `[]`.

   **Use when:** Tied occurrence witnesses matter.
   Saved candidates require up to $O(n)$ ranges; sorting $p$ final ties adds $O(p\log(p+1))$ work.

2. **Answer many interval queries.**

   **New task:** Decide whether each supplied half-open range is palindromic.

   **Picture and change:** Store answers for inner intervals instead of discarding them after expansion.
   Descending `left` solves row `left+1` first. Equal endpoints extend its inner answer; lengths one and two need no lookup.

   **Code:** Complete replacement. Queries must satisfy `0 <= left <= end <= len(text)`:

   ```python
   def palindrome_queries(text, ranges):
      size = len(text)
      table = [[False] * size for _ in range(size)]
      for left in range(size - 1, -1, -1):
         for right in range(left, size):
            table[left][right] = text[left] == text[right] and (
               right - left < 2 or table[left + 1][right - 1]
            )
      return [True if left == end else table[left][end - 1] for left, end in ranges]
   ```

   **Check:** `babad` with `[[0,3],[1,4],[0,5],[2,2]]` gives `[True,True,False,True]`. Empty ranges are palindromic here.

   **Use when:** Repeated decisions justify $O(n^2)$ preprocessing and table storage, then $O(1)$ per query, plus returned answers.

## Rebuild the Aggregation

What changes when the task asks for [all palindrome occurrences](/questions/palindromic-substrings/)?

**Answer:** Keep the same center enumeration. Add one for each accepted radius instead of retaining only a longest range.
