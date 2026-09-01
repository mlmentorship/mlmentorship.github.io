---
title: "Longest Repeating Character Replacement"
description: "Replace at most `k` letters so the longest possible substring has one repeated letter."
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

> Replace at most `k` letters so the longest possible substring has one repeated letter.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:longest-repeating-character-replacement-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="longest-repeating-character-replacement-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="longest-repeating-character-replacement-state-title">Longest Repeating Character Replacement: A window is valid when every non-majority character fits inside the replacement budget.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="longest-repeating-character-replacement" role="group" tabindex="0" aria-label="Longest Repeating Character Replacement: A window is valid when every non-majority character fits inside the replacement budget."><div class="coding-visual-example"><span>Input and goal</span><strong>Replace at most `k` letters so the longest possible substring has one repeated letter.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Count the window"><div class="coding-trace-frame-heading"><span>Count the window</span><strong>AAB has majority count 2. One B needs one replacement.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-l">L</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="marker-r">R</span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>range</b>AAB</span><span><b>formula</b>3 - max_count 2 = 1</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Keep a valid length-4 window"><div class="coding-trace-frame-heading"><span>Keep a valid length-4 window</span><strong>AABA uses one replacement: length 4 minus majority count 3 equals 1.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-l">L</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="marker-r">R</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>range</b>AABA</span><span><b>formula</b>4 - max_count 3 = 1</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Return the longest length"><div class="coding-trace-frame-heading"><span>Return the longest length</span><strong>The scan may later see ABAB, but the valid window AABA already proves length 4.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-best">best</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="marker-best">best</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">5</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">6</small></span></div><div class="coding-trace-meta"><span><b>range</b>AABA</span><span><b>result</b>4</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Count the window</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Keep a valid length-4 window</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return the longest length</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>A window is valid when every non-majority character fits inside the replacement budget.</p></div><figcaption><strong>Read it this way:</strong> AAB has majority count 2. One B needs one replacement. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Sliding window with counts.

**Simple idea:** Keep the most common letter. Every other letter in the window needs one
replacement. The window is valid when:

`window length - largest letter count <= k`

```python
from collections import defaultdict

def character_replacement(text: str, replacements: int) -> int:
   counts: dict[str, int] = defaultdict(int)
   left = 0
   largest_count = 0
   best = 0

   for right, char in enumerate(text):
      counts[char] += 1
      largest_count = max(largest_count, counts[char])

      while right - left + 1 - largest_count > replacements:
         counts[text[left]] -= 1
         left += 1

      best = max(best, right - left + 1)

   return best
```

**Cost:** $O(n)$ time and $O(k)$ space.
