---
title: "Best Time to Buy and Sell Stock"
description: "Summarize the cheapest eligible baseline for each sale, preserve optional profit, and change update timing for witnesses and mandatory trades."
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

Choose at most one purchase and a later sale to maximize the profit from a daily price list.

## Problem

**Input.** A list of finite integer `prices` in chronological day order.

**Output.** Return the largest achievable sale price minus purchase price, or zero if no profitable trade exists.

**Rules.** Buy at most once and sell on a strictly later day.
There are no fees, short sales, or multiple transactions.
Doing nothing is allowed. Empty and singleton lists return zero; leave the list unchanged.

**Example.** For prices `[7, 1, 5, 3, 6, 4]`, return `5`.
Buy at index 1 for 1 and sell at index 4 for 6.
For falling prices `[7, 6, 4]`, return `0`, not a negative forced-trade profit.

## Picture

Scan `[7,1,5,3,6,4]` from left to right:

| Day | Price | `lowest` after insertion | Candidate | `best` |
| --- | --- | --- | --- | --- |
| 0 | 7 | 7 | 0 | 0 |
| 1 | 1 | 1 | 0 | 0 |
| 2 | 5 | 1 | 4 | 4 |
| 3 | 3 | 1 | 2 | 4 |
| 4 | 6 | 1 | 5 | 5 |
| 5 | 4 | 1 | 3 | 5 |

For one fixed sale, a cheaper buy cannot reduce profit.

After each day, `lowest` stores the prefix minimum. The saved `best` stores the largest completed optional profit.

The source inserts today's price before computing `price - lowest`. A new minimum produces a same-day zero, not a valid transaction.

That zero cannot improve `best`, which starts at zero. Every positive improvement uses a strictly earlier, lower price.

The result is five: buy on day one and sell on day four. The later price four does not erase that saved answer.

Visual `low` marks the current prefix minimum's day; `day` marks the current candidate sale.
The underline marks the saved trade, not today's candidate.

<!-- visual:best-time-to-buy-and-sell-stock-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="best-time-to-buy-and-sell-stock-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="best-time-to-buy-and-sell-stock-state-title">Best Time to Buy and Sell Stock: For each possible selling day, subtract the lowest price in its scanned prefix and preserve the largest profit.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="best-time-to-buy-and-sell-stock" role="group" tabindex="0" aria-label="Best Time to Buy and Sell Stock: For each possible selling day, subtract the lowest price in its scanned prefix and preserve the largest profit."><div class="coding-visual-example"><span>Input and goal</span><strong>Buy once, then sell later. Return the largest profit.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="day-zero" role="group" aria-label="Initialize from infinity"><div class="coding-trace-frame-heading"><span>Initialize from infinity</span><strong>Day 0 sets lowest=min(infinity,7)=7. Its same-day zero cannot improve the no-trade best=0.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:6" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="lowest-buy">low</span><span class="coding-trace-array-pointer" data-motion-key="sell-cursor">day</span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">5</small></span></div><div class="coding-trace-meta"><span><b>scannedPrefix</b>[0..0]</span><span><b>arithmetic</b>lowest=7; profit=7-7=0; best=0</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="day-one" hidden role="group" aria-label="Move the low to day 1"><div class="coding-trace-frame-heading"><span>Move the low to day 1</span><strong>Price 1 becomes the new minimum before scoring. Its same-day zero does not improve the no-trade option.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:6" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="lowest-buy">low</span><span class="coding-trace-array-pointer" data-motion-key="sell-cursor">day</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">5</small></span></div><div class="coding-trace-meta"><span><b>scannedPrefix</b>[0..1]</span><span><b>arithmetic</b>lowest=min(7,1)=1; profit=1-1=0; best=0</span><span><b>movement</b>lowest-buy moves right because 1 is cheaper</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="day-two" hidden role="group" aria-label="Evaluate selling at 5"><div class="coding-trace-frame-heading"><span>Evaluate selling at 5</span><strong>On day 2, lowest stays 1 and selling at 5 gives 5-1=4, so best becomes 4.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:6" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state is-saved-range" role="listitem" data-motion-key="value-1" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="lowest-buy">low</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus is-saved-range" role="listitem" data-motion-key="value-2" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="sell-cursor">day</span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">5</small></span></div><div class="coding-trace-meta"><span><b>scannedPrefix</b>[0..2]</span><span><b>arithmetic</b>lowest=1; profit=5-1=4; best=4</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="day-three" hidden role="group" aria-label="Evaluate selling at 3"><div class="coding-trace-frame-heading"><span>Evaluate selling at 3</span><strong>Day 3 is not a new low. Profit 3-1=2 does not beat best 4.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:6" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state is-saved-range" role="listitem" data-motion-key="value-1" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="lowest-buy">low</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-2" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="sell-cursor">day</span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">5</small></span></div><div class="coding-trace-meta"><span><b>scannedPrefix</b>[0..3]</span><span><b>arithmetic</b>lowest=1; profit=3-1=2; best=4</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="day-four" hidden role="group" aria-label="Save the best sale at 6"><div class="coding-trace-frame-heading"><span>Save the best sale at 6</span><strong>On day 4, lowest remains 1 and profit 6-1=5 raises best from 4 to 5.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:6" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state is-saved-range" role="listitem" data-motion-key="value-1" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="lowest-buy">low</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-2" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-3" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-focus is-saved-range" role="listitem" data-motion-key="value-4" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="sell-cursor">day</span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">5</small></span></div><div class="coding-trace-meta"><span><b>scannedPrefix</b>[0..4]</span><span><b>arithmetic</b>lowest=1; profit=6-1=5; best=5</span></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="day-five" hidden role="group" aria-label="Finish at price 4"><div class="coding-trace-frame-heading"><span>Finish at price 4</span><strong>Day 5 gives profit 4-1=3, below best 5. Keep buy day 1 and sell day 4.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:6" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">7</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state is-saved-range" role="listitem" data-motion-key="value-1" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="lowest-buy">low</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-2" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">5</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-3" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-4" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">6</span><small class="coding-trace-array-index">4</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-pointer" data-motion-key="sell-cursor">day</span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">5</small></span></div><div class="coding-trace-meta"><span><b>scannedPrefix</b>[0..5]</span><span><b>arithmetic</b>lowest=1; profit=4-1=3; best=5</span><span><b>result</b>5 (buy day 1 at 1; sell day 4 at 6)</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 6</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize from infinity</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Move the low to day 1</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Evaluate selling at 5</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Evaluate selling at 3</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Save the best sale at 6</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Finish at price 4</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use a running minimum when an ordered one-pass problem asks for the best later-minus-earlier difference and the buy or baseline must occur before the current item.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">After each day, lowest is the minimum price in the scanned prefix and best is the largest valid sell-minus-earlier-buy profit whose selling day lies in that prefix.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">For maximum ordered differences, summarize the best earlier baseline while treating each new value as the right endpoint; analogous scans find maximum rise, minimum spread, and best prefix-relative gain.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Day 0 sets lowest=min(infinity,7)=7. Its same-day zero cannot improve the no-trade best=0. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

## Code

```python
def max_profit(prices: list[int]) -> int:
   lowest = float("inf")
   best = 0

   for price in prices:
      lowest = min(lowest, price)
      best = max(best, price - lowest)
   return best
```

For $n$ days, the scan uses $O(n)$ integer operations and $O(1)$ stored slots.
Global maximum minus global minimum fails on `[7,1]`: the earlier seven cannot be a later sale.

<!-- learning:core-end -->

## Change the Saved Result or Obligation

Start from `max_profit`. Two adjustable parts are the returned evidence and permission to skip trading.

1. **Return a feasible witness.**

   **New task:** Return `[profit, buy, sell]` for a profitable trade, or `None`.

   **Picture and change:** Retain the minimum's index. Score each sale against the strictly earlier minimum before inserting today's price.

   **Code:** Complete replacement:

   ```python
   def best_trade_days(prices):
      if len(prices) < 2:
         return None
      lowest, buy = prices[0], 0
      best, result = 0, None
      for sell in range(1, len(prices)):
         profit = prices[sell] - lowest
         if profit > best:
            best, result = profit, [profit, buy, sell]
         if prices[sell] < lowest:
            lowest, buy = prices[sell], sell
      return result
   ```

   **Check:** `[7,1,5,3,6,4]` returns `[5,1,4]`; `[3,2,1]` returns `None`.

   Strict comparisons retain the earliest sale attaining the best profit and its earliest minimum-price buy.
   Time remains $O(n)$, with $O(1)$ slots.

   **Use when:** The caller needs actual, strictly ordered days rather than a scalar optimum.

2. **Require exactly one trade.**

   **New task:** Return the best profit even when every possible trade loses money.

   **Picture and change:** Remove no-trade zero. Initialize `best` from the first valid pair and score before updating `lowest`.

   **Code:** Complete replacement; fewer than two days returns `None`.

   ```python
   def mandatory_trade_profit(prices):
      if len(prices) < 2:
         return None
      lowest, best = prices[0], prices[1] - prices[0]
      for sell in range(1, len(prices)):
         best = max(best, prices[sell] - lowest)
         lowest = min(lowest, prices[sell])
      return best
   ```

   **Check:** `[7,5,1]` returns `-2`, while the optional reference returns zero. A singleton returns `None`.

   Inserting the current day first would introduce an illegal same-day zero. Initializing `best = 0` would also hide every loss.

   **Use when:** Skipping the action is forbidden. Time remains $O(n)$, with $O(1)$ slots.

## Rebuild Eligibility

How would a minimum waiting period change the stored minimum?

**Answer:** Admit a buy day only when it is old enough for the current sale. The minimum's role stays unchanged.

Multiple trades, fees, cooldowns, and short selling need separately derived state. This deterministic array task makes no market forecast.
