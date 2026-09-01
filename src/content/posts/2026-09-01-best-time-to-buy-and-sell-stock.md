---
title: "Best Time to Buy and Sell Stock"
description: "Buy once, then sell later. Return the largest profit."
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

> Buy once, then sell later. Return the largest profit.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:best-time-to-buy-and-sell-stock-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="best-time-to-buy-and-sell-stock-state-title"><p class="visual-kicker">One pass, running best</p><p class="visual-title" id="best-time-to-buy-and-sell-stock-state-title">Best Time to Buy and Sell Stock: Carry the smallest, largest, or best state seen so far</p><div class="coding-visual coding-visual--running" data-coding-visual data-coding-mode="running" data-coding-slug="best-time-to-buy-and-sell-stock" role="group" aria-label="Best Time to Buy and Sell Stock: prices 7,1,5 -&gt; at 5, the saved buy price 1 yields profit 4. The carried state is the complete summary needed to make the next position optimal."><div class="coding-visual-example"><span>Concrete trace</span><strong>prices 7,1,5 -&gt; at 5, the saved buy price 1 yields profit 4</strong></div><div class="coding-visual-sketch coding-visual-sketch--running"><div class="coding-sketch-row"><span class="coding-sketch-pill coding-sketch-pill--state">best so far</span><span class="coding-sketch-arrow">&larr;</span><span class="coding-sketch-pill coding-sketch-pill--input">current</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--focus">new best?</span></div><p class="coding-sketch-note">the carried summary is enough to judge the next value</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Observe</span><strong>current value</strong><small>Read the next price, sum, or candidate.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Carry</span><strong>state so far</strong><small>Keep the summary future positions can use.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Update</span><strong>best decision</strong><small>Compare starting fresh, extending, buying, or selling.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Record</span><strong>best answer</strong><small>Preserve the strongest result seen anywhere.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>The carried state is the complete summary needed to make the next position optimal.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The scan does not remember every earlier value. It remembers the one summary that gives every future position its best possible continuation. For this problem, hold onto the concrete trace: prices 7,1,5 -&gt; at 5, the saved buy price 1 yields profit 4.</figcaption></figure>

**Pattern:** Running minimum.

**Simple idea:** For each selling price, the best earlier buy is the lowest price seen so
far. Update that lowest price and the best profit as you scan.

```python
def max_profit(prices: list[int]) -> int:
   lowest = float("inf")
   best = 0

   for price in prices:
      lowest = min(lowest, price)
      best = max(best, price - lowest)
   return best
```

**Cost:** $O(n)$ time and $O(1)$ space.
