---
title: "Valid Palindrome"
description: "Ignore punctuation and letter case, then check whether text reads the same both ways."
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

> Ignore punctuation and letter case, then check whether text reads the same both ways.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:valid-palindrome-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="valid-palindrome-state-title"><p class="visual-kicker">Characters and centers</p><p class="visual-title" id="valid-palindrome-state-title">Valid Palindrome: Compare the only characters that can still decide the answer</p><div class="coding-visual coding-visual--string" data-coding-visual data-coding-mode="string" data-coding-slug="valid-palindrome" role="group" aria-label="Valid Palindrome: A man, a plan -&gt; skip spaces and commas, compare normalized ends. Everything outside the active pointers or center expansion has already been resolved."><div class="coding-visual-example"><span>Concrete trace</span><strong>A man, a plan -&gt; skip spaces and commas, compare normalized ends</strong></div><div class="coding-visual-sketch coding-visual-sketch--string"><div class="coding-sketch-array"><span class="coding-sketch-pointer">L</span><span class="coding-sketch-cell">left</span><span class="coding-sketch-cell coding-sketch-cell--active">focus</span><span class="coding-sketch-cell coding-sketch-cell--active">focus</span><span class="coding-sketch-cell">right</span><span class="coding-sketch-pointer">R</span></div><p class="coding-sketch-note">compare or expand around the active characters in the concrete trace</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Point</span><strong>left / right</strong><small>Choose the two positions or a possible center.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Normalize</span><strong>skip or align</strong><small>Ignore separators or align matching lengths.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Expand</span><strong>equal pair</strong><small>Move outward while the local rule survives.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Record</span><strong>best text</strong><small>Keep the longest, valid, or decodable result.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Everything outside the active pointers or center expansion has already been resolved.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The string is a line, not a bag of characters. The pointers identify the only unresolved comparison, and each successful comparison makes the next one smaller. For this problem, hold onto the concrete trace: A man, a plan -&gt; skip spaces and commas, compare normalized ends.</figcaption></figure>

**Pattern:** Two pointers at opposite ends.

**Simple idea:** Skip non-letter and non-number characters. Compare the next real character
from each side, then move inward.

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

**Cost:** $O(n)$ time and $O(1)$ space.
