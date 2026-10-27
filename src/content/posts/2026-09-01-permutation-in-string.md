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

Determine whether a string contains a contiguous rearrangement of all characters in a given pattern.

## Problem

**Input.** Two Python strings, `pattern` and `text`.

**Output.** Return `True` if any substring of `text` is a permutation of `pattern`; otherwise return `False`.

**Rules.** A matching substring must be contiguous and contain exactly the pattern's character counts.
Use exact code points, including case and whitespace. Do not modify either string.
A longer pattern returns `False`. An empty pattern returns `True`, including with empty text.

**Example.** Pattern `"ab"` and text `"eidbaooo"` return `True` because `"ba"` starts at index 3.
The same pattern with `"eidboaoo"` returns `False`; separated letters do not form a substring.

## Picture: Keep Exactly the Pattern's Width

<!-- visual:permutation-in-string-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="permutation-in-string-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="permutation-in-string-state-title">Permutation in String: Slide one width-2 window over &quot;eidbaooo&quot;; add R, remove the outgoing L character, then compare counts.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="permutation-in-string" role="group" tabindex="0" aria-label="Permutation in String: Slide one width-2 window over &quot;eidbaooo&quot;; add R, remove the outgoing L character, then compare counts."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether any substring has the same letter counts as the pattern.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initial-window" role="group" aria-label="Count the first width-2 window"><div class="coding-trace-frame-heading"><span>Count the first width-2 window</span><strong>Pattern ab needs one a and one b. Initial window ei has different counts.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="left">L</span><span class="coding-trace-array-cell">e</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="right">R</span><span class="coding-trace-array-cell">i</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">d</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">o</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">o</span><small class="coding-trace-array-index">6</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-7"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">o</span><small class="coding-trace-array-index">7</small></span></div><div class="coding-trace-meta"><span><b>range</b>[0..1] = &quot;ei&quot;</span><span><b>comparison</b>e:1,i:1 != a:1,b:1</span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>need</b><span>a:1, b:1</span></span><span class="coding-trace-map-entry"><b>window</b><span>e:1, i:1</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="window-id" hidden role="group" aria-label="Slide from ei to id"><div class="coding-trace-frame-heading"><span>Slide from ei to id</span><strong>At R = 2 add d, remove outgoing text[0] = e, delete e after its count reaches zero, and move L right to 1.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">e</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="left">L</span><span class="coding-trace-array-cell">i</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="right">R</span><span class="coding-trace-array-cell">d</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">o</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">o</span><small class="coding-trace-array-index">6</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-7"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">o</span><small class="coding-trace-array-index">7</small></span></div><div class="coding-trace-meta"><span><b>range</b>[1..2] = &quot;id&quot;</span><span><b>direction</b>L: 0 -&gt; 1; R: 1 -&gt; 2</span><span><b>comparison</b>d:1,i:1 != a:1,b:1</span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>need</b><span>a:1, b:1</span></span><span class="coding-trace-map-entry"><b>window</b><span>d:1, i:1</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="window-db" hidden role="group" aria-label="Slide from id to db"><div class="coding-trace-frame-heading"><span>Slide from id to db</span><strong>At R = 3 add b, remove outgoing text[1] = i, delete zero-count i, and move L right to 2.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">e</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">i</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="left">L</span><span class="coding-trace-array-cell">d</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="right">R</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">o</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">o</span><small class="coding-trace-array-index">6</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-7"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">o</span><small class="coding-trace-array-index">7</small></span></div><div class="coding-trace-meta"><span><b>range</b>[2..3] = &quot;db&quot;</span><span><b>direction</b>L: 1 -&gt; 2; R: 2 -&gt; 3</span><span><b>comparison</b>b:1,d:1 != a:1,b:1</span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>need</b><span>a:1, b:1</span></span><span class="coding-trace-map-entry"><b>window</b><span>b:1, d:1</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="window-ba-match" hidden role="group" aria-label="Match the permutation ba"><div class="coding-trace-frame-heading"><span>Match the permutation ba</span><strong>At R = 4 add a, remove outgoing text[2] = d, and delete zero-count d. The window counter is now exactly a:1,b:1.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">e</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">i</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">d</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="left">L</span><span class="coding-trace-array-cell">b</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-pointer" data-motion-key="right">R</span><span class="coding-trace-array-cell">a</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">o</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">o</span><small class="coding-trace-array-index">6</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-7"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">o</span><small class="coding-trace-array-index">7</small></span></div><div class="coding-trace-meta"><span><b>range</b>[3..4] = &quot;ba&quot;</span><span><b>direction</b>L: 2 -&gt; 3; R: 3 -&gt; 4</span><span><b>comparison</b>a:1,b:1 = a:1,b:1</span><span><b>result</b>true</span></div><div class="coding-trace-map"><span class="coding-trace-label">saved state</span><span class="coding-trace-map-entry"><b>need</b><span>a:1, b:1</span></span><span class="coding-trace-map-entry"><b>window</b><span>a:1, b:1</span></span></div></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 4</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Count the first width-2 window</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Slide from ei to id</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Slide from id to db</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Match the permutation ba</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when a contiguous match may appear in any order: equal multisets require equal length, so only windows whose width equals the pattern length can qualify.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Before each comparison, window contains exactly text[R - pattern_length + 1..R], because the new right character was added and the one character that fell off the left was removed.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">When every candidate has a known width, maintain its summary by adding the entering item and removing the outgoing item; compare summaries instead of rebuilding or enumerating arrangements.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Pattern ab needs one a and one b. Initial window ei has different counts. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

| Window | Add/remove to reach it | Compare with `{a:1, b:1}` |
| --- | --- | --- |
| `ei` | Initialize width 2 | Different |
| `id` | Add d, remove e | Different |
| `db` | Add b, remove i | Different |
| `ba` | Add a, remove d | Equal: return `True` |

Before every comparison, `window` counts exactly the current width-sized substring.
Add `text[right]`, subtract `text[right-width]`, then delete a zero entry before comparing.
The figure's `L` is `right-width+1`; `R` is `right` after the slide.
Deletion keeps state sparse without relying on zero-entry equality rules.
[Exact multiset equality](/questions/valid-anagram/) explains why equal counts suffice; fixed width selects the candidate intervals.

## Code

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

For pattern length $m$, text length $n$, and $w$ tested windows, expected cost is $O(m+n+wd)$.
Here $d$ bounds counter keys inspected per equality. A fixed alphabet gives $O(m+n)$ time; auxiliary space is $O(d)$.

## Change 1: Return Every Matching Start

**New task.** Report all matches, including overlaps. An empty pattern matches all $n+1$ boundaries.

**Picture and change.** Continue after equality and save each start.

**Code.**

```python
from collections import Counter

def anagram_starts(pattern, text):
   width = len(pattern)
   if width > len(text):
      return []
   need, window = Counter(pattern), Counter(text[:width])
   answer = [0] if window == need else []
   for right in range(width, len(text)):
      window[text[right]] += 1
      outgoing = text[right - width]
      window[outgoing] -= 1
      if window[outgoing] == 0:
         del window[outgoing]
      if window == need:
         answer.append(right - width + 1)
   return answer
```

**Check.** `("ab", "abab")` gives `[0, 1, 2]`; `("", "ab")` also gives `[0, 1, 2]`.

**Use when.** Every position is needed. Output storage can be $O(n)$.

## Change 2: Return the First Start

**New task.** Return a location, with -1 for no match.

**Picture and change.** Reuse the complete function above; this simple wrapper still enumerates all matches.

**Code.** Place after `anagram_starts`:

```python
def first_anagram_start(pattern, text):
   starts = anagram_starts(pattern, text)
   return starts[0] if starts else -1
```

**Check.** `("ab", "eidbaooo")` gives 3; `("ab", "eeee")` gives -1; empty pattern gives 0.

**Use when.** A location is required. This wrapper does not inherit the Boolean reference's early-stop or constant-output-space behavior.

## Failure: Comparing Before Removal

The summary temporarily has width plus one after insertion. Comparing then tests the wrong interval.

## Transfer Check

What changes if the pattern repeats a character?

### Reasoned Answer

Width and add/remove mechanics stay unchanged. Exact multiplicities enforce repetitions; character membership alone cannot.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Why Fixed Width Is Enough

Equal multiplicities imply equal total length, so no other substring widths can match.
This differs from windows that repeatedly shrink to repair a variable validity condition.
Each step here removes exactly the character that falls outside the next fixed-width interval.

With zero width, the all-starts variant adds and removes the same character before each comparison.
Its counter stays empty, producing every boundary from 0 through `len(text)`.
The original Boolean reference returns immediately at its initial empty-window equality.

To avoid enumerating all matches for a first-index request, return the current start directly at the first equality.
That implementation would need its own early-stop tests; the displayed wrapper intentionally prioritizes reuse over that optimization.
Counter equality can inspect multiple keys, so do not call each comparison constant-time for an unbounded alphabet.

</div>
