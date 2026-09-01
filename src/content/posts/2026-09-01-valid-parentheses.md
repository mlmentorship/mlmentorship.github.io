---
title: "Valid Parentheses"
description: "Check whether all brackets close in the correct order."
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

> Check whether all brackets close in the correct order.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:valid-parentheses-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="valid-parentheses-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="valid-parentheses-state-title">Valid Parentheses: The newest unmatched opening bracket must match the next closing bracket.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="valid-parentheses" role="group" aria-label="Valid Parentheses: The newest unmatched opening bracket must match the next closing bracket."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether all brackets close in the correct order.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Push openings"><div class="coding-trace-frame-heading"><span>Push openings</span><strong>Read ( and [; both remain unfinished in the stack.</strong></div><div class="coding-trace-stack-layout"><div class="coding-trace-stack-input"><span class="coding-trace-label">input</span><strong>([</strong></div><div class="coding-trace-stack-column"><span class="coding-trace-label">top</span><span class="coding-trace-stack-item">(</span><span class="coding-trace-stack-item is-top">[</span></div></div><div class="coding-trace-meta"><span><b>current</b>[</span></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Match the top"><div class="coding-trace-frame-heading"><span>Match the top</span><strong>The next ] matches the stack top [, then } must match {.</strong></div><div class="coding-trace-stack-layout"><div class="coding-trace-stack-input"><span class="coding-trace-label">input</span><strong>([{}])</strong></div><div class="coding-trace-stack-column"><span class="coding-trace-label">top</span><span class="coding-trace-stack-item">(</span><span class="coding-trace-stack-item">[</span><span class="coding-trace-stack-item is-top">{</span></div></div><div class="coding-trace-meta"><span><b>current</b>}</span><span><b>action</b>pop {</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Empty means valid"><div class="coding-trace-frame-heading"><span>Empty means valid</span><strong>All openings were closed in reverse order.</strong></div><div class="coding-trace-stack-layout"><div class="coding-trace-stack-input"><span class="coding-trace-label">input</span><strong>([{}])</strong></div><div class="coding-trace-stack-column"><span class="coding-trace-label">top</span><span class="coding-trace-empty">empty</span></div></div><div class="coding-trace-meta"><span><b>result</b>true</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Push openings</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Match the top</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Empty means valid</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>The newest unmatched opening bracket must match the next closing bracket.</p></div><figcaption><strong>Read it this way:</strong> Read ( and [; both remain unfinished in the stack. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Stack.

**Simple idea:** Save opening brackets. A closing bracket must match the newest opening
bracket.

```python
def valid_parentheses(text: str) -> bool:
   opening: list[str] = []
   matching = {")": "(", "]": "[", "}": "{"}

   for char in text:
      if char not in matching:
         opening.append(char)
      elif not opening or opening.pop() != matching[char]:
         return False

   return not opening
```

**Cost:** $O(n)$ time and $O(n)$ space.
