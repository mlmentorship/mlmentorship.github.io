---
title: "Decode Ways"
description: "Count ways to decode digits where `1` through `26` map to letters."
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

> Count ways to decode digits where `1` through `26` map to letters.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:decode-ways-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="decode-ways-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="decode-ways-state-title">Decode Ways: A digit can extend one prior decoding; a valid two-digit number can extend two prior decodings.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="decode-ways" role="group" tabindex="0" aria-label="Decode Ways: A digit can extend one prior decoding; a valid two-digit number can extend two prior decodings."><div class="coding-visual-example"><span>Input and goal</span><strong>Count ways to decode digits where `1` through `26` map to letters.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Read 2"><div class="coding-trace-frame-heading"><span>Read 2</span><strong>The first digit gives one decoding.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-1-way">1 way</span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">2</small></span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="At 22"><div class="coding-trace-frame-heading"><span>At 22</span><strong>22 is valid, so one-digit and two-digit choices contribute.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="marker-2-ways">2 ways</span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>choices</b>2|2 and 22</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="At 226"><div class="coding-trace-frame-heading"><span>At 226</span><strong>6 can follow 22 or stand after 2, giving three total decodings.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="marker-3-ways">3 ways</span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>result</b>3</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Read 2</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>At 22</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>At 226</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>A digit can extend one prior decoding; a valid two-digit number can extend two prior decodings.</p></div><figcaption><strong>Read it this way:</strong> The first digit gives one decoding. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** DP with one-digit and two-digit choices.

**Simple idea:** A nonzero current digit can extend every decoding from one position back.
A valid two-digit number from 10 through 26 can extend every decoding from two positions
back.

```python
def num_decodings(text: str) -> int:
   if not text or text[0] == "0":
      return 0

   two_back = one_back = 1
   for index in range(1, len(text)):
      current = one_back if text[index] != "0" else 0
      if 10 <= int(text[index - 1 : index + 1]) <= 26:
         current += two_back
      two_back, one_back = one_back, current
   return one_back
```

**Cost:** $O(n)$ time and $O(1)$ space.
