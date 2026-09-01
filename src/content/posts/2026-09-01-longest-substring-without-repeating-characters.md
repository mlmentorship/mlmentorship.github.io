---
title: "Longest Substring Without Repeating Characters"
description: "Find the longest substring with no repeated character."
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

> Find the longest substring with no repeated character.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:longest-substring-without-repeating-characters-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="longest-substring-without-repeating-characters-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="longest-substring-without-repeating-characters-state-title">Longest Substring Without Repeating Characters: Keep the longest window whose characters are all distinct.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="longest-substring-without-repeating-characters" role="group" aria-label="Longest Substring Without Repeating Characters: Keep the longest window whose characters are all distinct."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the longest substring with no repeated character.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Grow the window"><div class="coding-trace-frame-heading"><span>Grow the window</span><strong>The first window abc contains no duplicate.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">L</span><span class="coding-trace-array-cell">a</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">R</span><span class="coding-trace-array-cell">c</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">c</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span></span></div><div class="coding-trace-meta"><span><b>range</b>abc</span><span><b>state</b>a,b,c</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Repair the duplicate"><div class="coding-trace-frame-heading"><span>Repair the duplicate</span><strong>The next a repeats, so move L past the old a before continuing.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">L</span><span class="coding-trace-array-cell">b</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">c</span></span><span class="coding-trace-array-item trace-tone-focus" role="listitem"><span class="coding-trace-array-mark">R</span><span class="coding-trace-array-cell">a</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">c</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span></span></div><div class="coding-trace-meta"><span><b>range</b>bca</span><span><b>state</b>b,c,a</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Save the best window"><div class="coding-trace-frame-heading"><span>Save the best window</span><strong>The longest distinct window seen has length 3.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">a</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">best</span><span class="coding-trace-array-cell">b</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">c</span></span><span class="coding-trace-array-item trace-tone-output" role="listitem"><span class="coding-trace-array-mark">best</span><span class="coding-trace-array-cell">a</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">c</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">b</span></span></div><div class="coding-trace-meta"><span><b>range</b>bca</span><span><b>result</b>3</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Grow the window</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Repair the duplicate</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Save the best window</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Keep the longest window whose characters are all distinct.</p></div><figcaption><strong>Read it this way:</strong> The first window abc contains no duplicate. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Sliding window with last positions.

**Simple idea:** When a repeated character is inside the current window, move `left` to one
position after its last copy. Never move `left` backward.

```python
def length_of_longest_substring(text: str) -> int:
   last_seen: dict[str, int] = {}
   left = 0
   best = 0

   for right, char in enumerate(text):
      if char in last_seen:
         left = max(left, last_seen[char] + 1)
      last_seen[char] = right
      best = max(best, right - left + 1)

   return best
```

**Cost:** $O(n)$ time and $O(k)$ space.
