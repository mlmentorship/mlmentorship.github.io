---
title: "Coin Change"
description: "Find the fewest coins needed to make an amount."
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

> Find the fewest coins needed to make an amount.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:coin-change-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="coin-change-state-title"><p class="visual-kicker">A small state graph</p><p class="visual-title" id="coin-change-state-title">Coin Change: Keep the complete answer for each smaller state</p><div class="coding-visual coding-visual--dp" data-coding-visual data-coding-mode="dp" data-coding-slug="coin-change" role="group" aria-label="Coin Change: amount 6 with coins 1,3,4 -&gt; fewest[6] builds from fewest[3] + 3. Each saved state is the complete answer for its prefix, amount, cell, or pair of prefixes."><div class="coding-visual-example"><span>Concrete trace</span><strong>amount 6 with coins 1,3,4 -&gt; fewest[6] builds from fewest[3] + 3</strong></div><div class="coding-visual-sketch coding-visual-sketch--dp"><div class="coding-sketch-dp-grid"><span class="coding-sketch-cell coding-sketch-cell--state">base</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell">smaller</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell coding-sketch-cell--active">current</span></div><p class="coding-sketch-note">each cell is a complete answer to one smaller question</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Base</span><strong>known state</strong><small>Initialize the smallest solvable problem.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Read</span><strong>earlier answers</strong><small>Look only at states the transition depends on.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Build</span><strong>current state</strong><small>Choose, count, or combine those answers.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Compress</span><strong>rolling memory</strong><small>Discard old states that no future step needs.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Each saved state is the complete answer for its prefix, amount, cell, or pair of prefixes.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> Treat the table as a map of smaller questions. The recurrence is the arrow between states; space optimization is safe only after the dependencies are visible. For this problem, hold onto the concrete trace: amount 6 with coins 1,3,4 -&gt; fewest[6] builds from fewest[3] + 3.</figcaption></figure>

**Pattern:** One-dimensional DP.

**State:** `fewest[total]` is the fewest coins needed for `total`.

**Simple idea:** To finish `total` with one coin, look at the answer for `total - coin` and
add one.

```python
def coin_change(coins: list[int], amount: int) -> int:
   unreachable = amount + 1
   fewest = [0] + [unreachable] * amount

   for total in range(1, amount + 1):
      for coin in coins:
         if 0 < coin <= total:
            fewest[total] = min(fewest[total], 1 + fewest[total - coin])

   return fewest[amount] if fewest[amount] != unreachable else -1
```

**Cost:** $O(amount \times coins)$ time and $O(amount)$ space.
