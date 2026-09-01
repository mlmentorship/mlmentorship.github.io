---
title: "Decode String"
description: "Decode text such as `3[a2[c]]` into `accaccacc`."
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

> Decode text such as `3[a2[c]]` into `accaccacc`.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:decode-string-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="decode-string-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="decode-string-state-title">Decode String: Save the outer text and repeat count whenever a nested bracket opens.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="decode-string" role="group" tabindex="0" aria-label="Decode String: Save the outer text and repeat count whenever a nested bracket opens."><div class="coding-visual-example"><span>Input and goal</span><strong>Decode text such as `3[a2[c]]` into `accaccacc`.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Enter the outer repeat"><div class="coding-trace-frame-heading"><span>Enter the outer repeat</span><strong>3[ starts a new inner string while saving repeat 3.</strong></div><div class="coding-trace-stack-layout"><div class="coding-trace-stack-input"><span class="coding-trace-label">input</span><strong>3[a2[c]]</strong></div><div class="coding-trace-stack-column"><span class="coding-trace-label">top</span><span class="coding-trace-stack-item is-top">outer=&quot;&quot;, count=3</span></div></div><div class="coding-trace-meta"><span><b>current</b>[</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Nest again"><div class="coding-trace-frame-heading"><span>Nest again</span><strong>At 2[, save the current a and repeat count 2.</strong></div><div class="coding-trace-stack-layout"><div class="coding-trace-stack-input"><span class="coding-trace-label">input</span><strong>3[a2[c]]</strong></div><div class="coding-trace-stack-column"><span class="coding-trace-label">top</span><span class="coding-trace-stack-item">outer=&quot;&quot;, count=3</span><span class="coding-trace-stack-item is-top">outer=&quot;a&quot;, count=2</span></div></div><div class="coding-trace-meta"><span><b>current</b>[</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Close from the inside"><div class="coding-trace-frame-heading"><span>Close from the inside</span><strong>c becomes cc, then acc, then accaccacc.</strong></div><div class="coding-trace-stack-layout"><div class="coding-trace-stack-input"><span class="coding-trace-label">input</span><strong>3[a2[c]]</strong></div><div class="coding-trace-stack-column"><span class="coding-trace-label">top</span><span class="coding-trace-stack-item is-top">outer=&quot;&quot;, count=3</span></div></div><div class="coding-trace-meta"><span><b>current</b>]</span><span><b>action</b>restore outer</span><span><b>result</b>accaccacc</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Enter the outer repeat</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Nest again</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Close from the inside</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Save the outer text and repeat count whenever a nested bracket opens.</p></div><figcaption><strong>Read it this way:</strong> 3[ starts a new inner string while saving repeat 3. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Stack for nested state.

**Simple idea:** At `[`, save the text and repeat count built so far. Start a new inner
string. At `]`, finish the inner string and attach it to the saved outer string.

```python
def decode_string(text: str) -> str:
   stack: list[tuple[str, int]] = []
   current = ""
   repeat = 0

   for char in text:
      if char.isdigit():
         repeat = repeat * 10 + int(char)
      elif char == "[":
         stack.append((current, repeat))
         current, repeat = "", 0
      elif char == "]":
         previous, count = stack.pop()
         current = previous + current * count
      else:
         current += char
   return current
```

**Cost:** $O(n + m)$ time and $O(n + m)$ space, where $m$ is the output length.
