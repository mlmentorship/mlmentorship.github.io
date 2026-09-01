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
<figure class="learning-figure coding-visual-figure" aria-labelledby="valid-parentheses-state-title"><p class="visual-kicker">Last unfinished, first resolved</p><p class="visual-title" id="valid-parentheses-state-title">Valid Parentheses: Keep unresolved work in the order it must finish</p><div class="coding-visual coding-visual--stack" data-coding-visual data-coding-mode="stack" data-coding-slug="valid-parentheses" role="group" aria-label="Valid Parentheses: ([{}]) -&gt; push (, [, {; each close must match the top. The top of the stack is the newest unresolved item and the only one that can be resolved next."><div class="coding-visual-example"><span>Concrete trace</span><strong>([{}]) -&gt; push (, [, {; each close must match the top</strong></div><div class="coding-visual-sketch coding-visual-sketch--stack"><div class="coding-sketch-stack"><span class="coding-sketch-label">older work</span><span class="coding-sketch-stack-item">waiting</span><span class="coding-sketch-stack-item">waiting</span><span class="coding-sketch-stack-item coding-sketch-stack-item--active">top resolves next</span></div><p class="coding-sketch-note">the newest unfinished item blocks everything below it</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Arrive</span><strong>new token</strong><small>Read the next symbol or value.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Hold</span><strong>stack top</strong><small>Keep work that cannot finish yet.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Resolve</span><strong>match or warmer</strong><small>A new item may finish the newest waiting item.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Restore</span><strong>remaining stack</strong><small>Anything left is still unfinished or invalid.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The top of the stack is the newest unresolved item and the only one that can be resolved next.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Look at the top, not the whole history. Nesting and next-greater relationships work because the newest unresolved item blocks everything below it. For this problem, hold onto the concrete trace: ([{}]) -&gt; push (, [, {; each close must match the top.</figcaption></figure>

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
