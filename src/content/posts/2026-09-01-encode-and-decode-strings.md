---
title: "Encode and Decode Strings"
description: "Convert a list of any strings into one string and recover the exact list."
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

> Convert a list of any strings into one string and recover the exact list.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:encode-and-decode-strings-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="encode-and-decode-strings-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="encode-and-decode-strings-state-title">Encode and Decode Strings: A length prefix tells the decoder exactly how many characters belong to each string.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="encode-and-decode-strings" role="group" tabindex="0" aria-label="Encode and Decode Strings: A length prefix tells the decoder exactly how many characters belong to each string."><div class="coding-visual-example"><span>Input and goal</span><strong>Convert a list of any strings into one string and recover the exact list.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Encode with lengths"><div class="coding-trace-frame-heading"><span>Encode with lengths</span><strong>lint becomes 4#lint and # becomes 1##.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-length-4">length 4</span><span class="coding-trace-array-cell">4#lint</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1##</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0#</span><small class="coding-trace-array-index">2</small></span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Read one length"><div class="coding-trace-frame-heading"><span>Read one length</span><strong>The decoder reads 4, skips #, and consumes exactly four characters.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-read-length">read length</span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="marker-start">start</span><span class="coding-trace-array-cell">l</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">i</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">n</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-pointer" data-motion-key="marker-end">end</span><span class="coding-trace-array-cell">t</span><small class="coding-trace-array-index">5</small></span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Recover the list"><div class="coding-trace-frame-heading"><span>Recover the list</span><strong>Lengths make delimiters inside the original strings harmless.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="marker-decoded">decoded</span><span class="coding-trace-array-cell">lint</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="marker-decoded">decoded</span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">&quot;&quot;</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>result</b>[&quot;lint&quot;,&quot;#&quot;,&quot;&quot;]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Encode with lengths</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Read one length</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Recover the list</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>A length prefix tells the decoder exactly how many characters belong to each string.</p></div><figcaption><strong>Read it this way:</strong> lint becomes 4#lint and # becomes 1##. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Length prefix.

**Simple idea:** Save each string as `length#text`. The decoder reads the length first, so
the text may contain any character, including `#`.

```python
def encode_strings(strings: list[str]) -> str:
   return "".join(f"{len(text)}#{text}" for text in strings)


def decode_strings(data: str) -> list[str]:
   strings = []
   index = 0

   while index < len(data):
      separator = data.index("#", index)
      length = int(data[index:separator])
      index = separator + 1
      strings.append(data[index : index + length])
      index += length
   return strings
```

**Cost:** $O(n)$ time and $O(n)$ output space.
