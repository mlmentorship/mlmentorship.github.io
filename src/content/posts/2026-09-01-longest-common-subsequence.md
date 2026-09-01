---
title: "Longest Common Subsequence"
description: "Find the longest sequence of characters that appears in two strings in the same order. Characters do not need to be next to each other."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Intermediate"
priority: "Core"
aliases: []
prerequisites: []
---

> Find the longest sequence of characters that appears in two strings in the same order. Characters do not need to be next to each other.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:longest-common-subsequence-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="longest-common-subsequence-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="longest-common-subsequence-state-title">Longest Common Subsequence: A matching pair advances both prefixes; a mismatch keeps the better skipped prefix.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="longest-common-subsequence" role="group" tabindex="0" aria-label="Longest Common Subsequence: A matching pair advances both prefixes; a mismatch keeps the better skipped prefix."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the longest sequence of characters that appears in two strings in the same order. Characters do not need to be next to each other.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="frame-1" role="group" aria-label="Compare prefixes"><div class="coding-trace-frame-heading"><span>Compare prefixes</span><strong>The grid state answers LCS for prefixes of abcde and ace.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col"></th><th scope="col">0</th><th scope="col">a</th><th scope="col">c</th><th scope="col">e</th></tr></thead><tbody><tr><td class="">0</td><td class="is-active">0</td><td class="">0</td><td class="">0</td><td class="">0</td></tr><tr><td class="">a</td><td class="">0</td><td class="">1</td><td class="">1</td><td class="">1</td></tr><tr><td class="">b</td><td class="">0</td><td class="">1</td><td class="">1</td><td class="">1</td></tr><tr><td class="">c</td><td class="">0</td><td class="">1</td><td class="">2</td><td class="">2</td></tr><tr><td class="">d</td><td class="">0</td><td class="">1</td><td class="">2</td><td class="">2</td></tr><tr><td class="">e</td><td class="">0</td><td class="">1</td><td class="">2</td><td class="">3</td></tr></tbody></table></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="frame-2" hidden role="group" aria-label="Match c"><div class="coding-trace-frame-heading"><span>Match c</span><strong>The c/c cell takes the diagonal answer and adds one.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col"></th><th scope="col">0</th><th scope="col">a</th><th scope="col">c</th><th scope="col">e</th></tr></thead><tbody><tr><td class="">0</td><td class="">0</td><td class="">0</td><td class="">0</td><td class="">0</td></tr><tr><td class="">a</td><td class="">0</td><td class="">1</td><td class="">1</td><td class="">1</td></tr><tr><td class="">b</td><td class="">0</td><td class="">1</td><td class="">1</td><td class="">1</td></tr><tr><td class="">c</td><td class="">0</td><td class="">1</td><td class="is-active">2</td><td class="">2</td></tr><tr><td class="">d</td><td class="">0</td><td class="">1</td><td class="">2</td><td class="">2</td></tr><tr><td class="">e</td><td class="">0</td><td class="">1</td><td class="">2</td><td class="">3</td></tr></tbody></table></div><div class="coding-trace-meta"><span><b>action</b>diagonal + 1</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="frame-3" hidden role="group" aria-label="Read the bottom-right"><div class="coding-trace-frame-heading"><span>Read the bottom-right</span><strong>The complete prefixes share subsequence ace of length 3.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col"></th><th scope="col">0</th><th scope="col">a</th><th scope="col">c</th><th scope="col">e</th></tr></thead><tbody><tr><td class="">0</td><td class="">0</td><td class="">0</td><td class="">0</td><td class="">0</td></tr><tr><td class="">a</td><td class="">0</td><td class="">1</td><td class="">1</td><td class="">1</td></tr><tr><td class="">b</td><td class="">0</td><td class="">1</td><td class="">1</td><td class="">1</td></tr><tr><td class="">c</td><td class="">0</td><td class="">1</td><td class="">2</td><td class="">2</td></tr><tr><td class="">d</td><td class="">0</td><td class="">1</td><td class="">2</td><td class="">2</td></tr><tr><td class="">e</td><td class="">0</td><td class="">1</td><td class="">2</td><td class="is-active">3</td></tr></tbody></table></div><div class="coding-trace-meta"><span><b>result</b>3</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Compare prefixes</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Match c</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Read the bottom-right</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>A matching pair advances both prefixes; a mismatch keeps the better skipped prefix.</p></div><figcaption><strong>Read it this way:</strong> The grid state answers LCS for prefixes of abcde and ace. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Two-dimensional DP stored as two rows.

**State:** The best answer for two string prefixes.

**Simple idea:** Matching characters add one to the answer before both characters. Different
characters skip one character from either string and keep the better result.

```python
def longest_common_subsequence(first: str, second: str) -> int:
   previous = [0] * (len(second) + 1)

   for first_char in first:
      current = [0]
      for index, second_char in enumerate(second, 1):
         if first_char == second_char:
            current.append(1 + previous[index - 1])
         else:
            current.append(max(current[-1], previous[index]))
      previous = current

   return previous[-1]
```

**Cost:** $O(mn)$ time and $O(n)$ space.
