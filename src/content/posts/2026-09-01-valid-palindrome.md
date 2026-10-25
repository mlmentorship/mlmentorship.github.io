---
title: "Valid Palindrome"
description: "Repair one irrelevant endpoint at a time, preserve matched outer pairs, and distinguish per-character lowercase from normalization and one-deletion edits."
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

Remove irrelevant endpoints before testing symmetry. Move inward only after discarding an irrelevant endpoint or verifying a matching pair.

## Problem

> Ignore punctuation and letter case, then check whether text reads the same both ways.

The precise reference policy uses Python `isalnum()` and compares each retained code point's `lower()` result.
It does not implement whole-string case folding or grapheme comparison.

Return a Boolean without changing the string. Empty and entirely filtered strings return `True`.

## Picture

Keep raw `"A, b a!"` at indices zero through six. `left` and `right` bound the unanswered interval.

| Action | Remaining raw indices |
| --- | --- |
| Initialize | `[0,6]` |
| Skip right `!` only | `[0,5]` |
| Match `A.lower()` with `a.lower()` | `[1,4]` |
| Skip left comma | `[2,4]` |
| Skip left space | `[3,4]` |
| Skip right space | `[3,3]` |

Only the center `b` remains. It equals itself, so the answer is `True`.

Outside `[left,right]`, every retained pair matches and every discarded character is irrelevant.
Skipping punctuation preserves that claim. Matching both endpoints certifies a pair before moving both pointers.

A relevant mismatch proves `False`: neither endpoint may be discarded under this policy.

One irrelevant endpoint permits only one pointer move. Advancing both after skipping `!` in `"!ab"` would miss the unequal `a,b` pair.

<!-- visual:valid-palindrome-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="valid-palindrome-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="valid-palindrome-state-title">Valid Palindrome: Keep raw text in place, skip non-alphanumeric endpoints one at a time, and move both pointers only after a case-insensitive match.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="valid-palindrome" role="group" tabindex="0" aria-label="Valid Palindrome: Keep raw text in place, skip non-alphanumeric endpoints one at a time, and move both pointers only after a case-insensitive match."><div class="coding-visual-example"><span>Input and goal</span><strong>Ignore punctuation and letter case, then check whether text reads the same both ways.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize" role="group" aria-label="Initialize at both raw endpoints"><div class="coding-trace-frame-heading"><span>Initialize at both raw endpoints</span><strong>For &quot;A, b a!&quot;, L=0 points to A and R=6 points to !.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:7" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">,</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">space</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">space</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">!</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>coveredRange</b>[0..6]</span><span><b>action</b>initialize opposite-end pointers</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="skip-right-punctuation" hidden role="group" aria-label="Skip punctuation on the right"><div class="coding-trace-frame-heading"><span>Skip punctuation on the right</span><strong>L points to alphanumeric A. R points to !, so decrement only R from 6 to 5.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:7" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">,</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">space</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">space</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">!</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>coveredRange</b>[0..5]</span><span><b>movement</b>R: 6 -&gt; 5 because ! is not alphanumeric</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="match-a" hidden role="group" aria-label="Compare the first pair"><div class="coding-trace-frame-heading"><span>Compare the first pair</span><strong>A and a match after lowercase comparison. Advance L from 0 to 1 and R from 5 to 4.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:7" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-cell">,</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">space</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">space</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">!</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>coveredRange</b>[1..4]</span><span><b>comparison</b>a == a</span><span><b>movement</b>L: 0 -&gt; 1; R: 5 -&gt; 4 after match</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="skip-comma" hidden role="group" aria-label="Skip the comma on the left"><div class="coding-trace-frame-heading"><span>Skip the comma on the left</span><strong>At L=1, comma is not alphanumeric. Increment only L to 2.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:7" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">,</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-cell">space</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">space</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">!</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>coveredRange</b>[2..4]</span><span><b>movement</b>L: 1 -&gt; 2 because comma is not alphanumeric</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="skip-left-space" hidden role="group" aria-label="Skip the space on the left"><div class="coding-trace-frame-heading"><span>Skip the space on the left</span><strong>At L=2, space is not alphanumeric, so L advances again to index 3.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:7" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">,</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">space</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">space</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">!</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>coveredRange</b>[3..4]</span><span><b>movement</b>L: 2 -&gt; 3 because space is not alphanumeric</span></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="skip-right-space" hidden role="group" aria-label="Skip the space on the right"><div class="coding-trace-frame-heading"><span>Skip the space on the right</span><strong>L points to b and R=4 points to a space. Decrement only R to 3.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:7" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">,</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">space</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-state trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">space</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">!</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>coveredRange</b>[3..3]</span><span><b>movement</b>R: 4 -&gt; 3 because space is not alphanumeric</span></div></div><div class="coding-trace-frame" data-coding-frame="6" data-frame-key="finish" hidden role="group" aria-label="Stop when pointers meet"><div class="coding-trace-frame-heading"><span>Stop when pointers meet</span><strong>Now L=R=3 at b, so left &lt; right is false. Every outer pair matched, so return True.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:7" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">,</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">space</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-state trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">space</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">!</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>coveredRange</b>[3..3]</span><span><b>guard</b>3 &lt; 3 is false</span><span><b>result</b>true</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 7</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize at both raw endpoints</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Skip punctuation on the right</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Compare the first pair</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Skip the comma on the left</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Skip the space on the left</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Skip the space on the right</strong></button><button type="button" data-coding-frame-button="6"><span>7</span><strong>Stop when pointers meet</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use opposite-end pointers when equality must hold symmetrically after ignoring characters, case, or other endpoint noise.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Before each loop iteration, every accepted alphanumeric pair outside [left,right] matches case-insensitively; discarded outside characters are non-alphanumeric and cannot affect the answer.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">When irrelevant data appears at either boundary, repair one endpoint at a time until both are comparable; only then apply the symmetric predicate and advance both.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> For &quot;A, b a!&quot;, L=0 points to A and R=6 points to !. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

## Code

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

Each pointer moves monotonically, so $n$ raw code points require $O(n)$ bounded-character work and $O(1)$ auxiliary space.

`isalnum()` includes non-ASCII letters and numbers. It is not an ASCII-only test.

<!-- learning:core-end -->

## Change the Comparison Contract

Start from `valid_palindrome`. Two adjustable parts are normalization and permission to delete a relevant character.

1. **Normalize before comparing.**

   **New task:** Apply Unicode NFKC compatibility normalization, then case folding, then alphanumeric filtering.

   **Picture and change:** A raw character can expand into several normalized characters. Replace raw pointers with a transformed sequence.

   **Code:** Complete replacement with an explicit normalization policy:

   ```python
   import unicodedata


   def unicode_normalized_palindrome(text):
      normalized = unicodedata.normalize("NFKC", text).casefold()
      comparable = [char for char in normalized if char.isalnum()]
      return comparable == comparable[::-1]
   ```

   **Check:** `"\u00dfs"` folds to `sss`, so it returns `True`; the reference returns `False`.
   Fullwidth `"\uff21b\uff41"` normalizes to `aba`.

   NFKC and case folding are declared choices, not universal language rules. Grapheme handling remains unspecified.
   Normalization adds work. For transformed length $m$, the comparison and stored sequences cost $O(m)$ time and space.

   **Use when:** The product requires this exact equality policy, including its compatibility transformations.

2. **Permit one deletion.**

   **New task:** Check already comparable ASCII text after deleting at most one character.

   **Picture and change:** At the first mismatch, one unequal endpoint must go. Test both remaining intervals without further deletions.

   **Code:** Complete replacement. This version does not filter punctuation or change case.

   ```python
   def palindrome_one_deletion(text):
      def matches(left, right):
         while left < right:
            if text[left] != text[right]:
               return False
            left, right = left + 1, right - 1
         return True

      left, right = 0, len(text) - 1
      while left < right:
         if text[left] != text[right]:
            return matches(left + 1, right) or matches(left, right - 1)
         left, right = left + 1, right - 1
      return True
   ```

   **Check:** `abca` returns `True` by deleting `b` or `c`; `abc` returns `False`.

   Greedily deleting left fails on `abbab`: only deleting the right endpoint at its first mismatch succeeds.
   At most two extra interval scans preserve $O(n)$ time and $O(1)$ auxiliary space.

   **Use when:** A relevant mismatch may be repaired, not merely rejected.

## Rebuild the Boundary

Why can an odd center remain unpaired? Does this algorithm find the longest palindromic substring?

**Answer:** The center equals itself; outer relevant pairs are already verified. This checks the complete sequence, not candidate substrings.
