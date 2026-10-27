---
title: "Find Median From Data Stream"
description: "Maintain ordered lower and upper heaps at completed insertions, distinguish repair states, and adapt float medians to exact fractions or nearest-rank quantiles."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Advanced"
priority: "Specialist"
aliases: []
prerequisites: []
---

Accept integers one at a time and report the median of all values inserted so far.

## Problem

**Input.** Calls to `add_num(num)` and `find_median()` on an initially empty `MedianFinder`.

**Output.** Each median query returns the middle sorted value, or the average of both middle values for an even count.

**Rules.** Inserted values are integers; repeats count separately.
The reference returns floating-point answers and retains the full history, not a sliding window.
There is no deletion operation. Query only after insertion; an empty query raises `IndexError`.
Extremely large integers may overflow the reference's float conversion.

**Example.** Insert 1, then 2: `find_median()` returns `1.5`.
After inserting 3, it returns `2.0`, regardless of the insertion order of those three values.

## Picture

At every completed insertion, maintain **both** conditions:

- Every logical lower-half value is at most every upper-half value.
- Lower has the same size as upper, or exactly one extra value.

Thus lower's maximum is the odd middle rank. For even size, lower's maximum and upper's minimum bound the midpoint.
Neither condition alone is sufficient.

| Added value | Logical lower heap | Upper heap | Median |
| --- | --- | --- | --- |
| `5` | `[5]` | `[]` | `5.0` |
| `2` | `[2]` | `[5]` | `3.5` |
| `10` | `[5,2]` | `[10]` | `5.0` |
| `4` | `[4,2]` | `[5,10]` | `4.5` |

Each heap orders parent against children, not the whole array.
This Python 3.12 implementation stores negatives in `lower`; its minimum stored root represents the logical maximum.

First push into lower. Transfer its maximum to upper, repairing cross-partition order.
If upper is larger, transfer its minimum back, repairing the required sizes without breaking order.

Repair frames may violate either final condition.
After pushing ten, lower's logical root ten exceeds upper's root five, despite acceptable sizes.
After pushing two, lower has two values and upper none, despite ordered partitions.
Read a median only after the complete insertion.

Source output is `float`. Huge integer medians may round or overflow during conversion; heap storage alone does not guarantee exact output.

<!-- visual:find-median-from-data-stream-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="find-median-from-data-stream-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="find-median-from-data-stream-state-title">Find Median From Data Stream: After completed insertions, ordered halves and balanced sizes expose the middle ranks at the heap roots.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="find-median-from-data-stream" role="group" tabindex="0" aria-label="Find Median From Data Stream: After completed insertions, ordered halves and balanced sizes expose the middle ranks at the heap roots."><div class="coding-visual-example"><span>Input and goal</span><strong>Add numbers one at a time and return the current median.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize-heaps" role="group" aria-label="Initialize two empty heaps"><div class="coding-trace-frame-heading"><span>Initialize two empty heaps</span><strong>Both heaps start empty, so a median query is undefined. Negated lower values make its stored minimum the logical maximum.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">heap</th><th scope="col">stored values</th><th scope="col">logical role</th></tr></thead><tbody><tr><td class="" data-motion-key="table-0-0">lower</td><td class="" data-motion-key="table-0-1">[]</td><td class="" data-motion-key="table-0-2">max-heap via negatives</td></tr><tr><td class="" data-motion-key="table-1-0">upper</td><td class="" data-motion-key="table-1-1">[]</td><td class="" data-motion-key="table-1-2">min-heap</td></tr></tbody></table></div><div class="coding-trace-meta"><span><b>stream</b>[5,2,10,4]</span><span><b>status</b>initialized: no median yet</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="push-5-lower" hidden role="group" aria-label="Push 5 into lower"><div class="coding-trace-frame-heading"><span>Push 5 into lower</span><strong>heappush(lower, -5) produces lower stored [-5], whose logical max-root is 5.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>lower max-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root is-current" data-motion-key="heap-value-5-0">5</span></li></ul></section><section class="coding-trace-heap-group"><h4>upper min-heap</h4><span class="coding-trace-empty">empty</span></section></div><div class="coding-trace-meta"><span><b>stream</b>[5,2,10,4]</span><span><b>seen</b>5</span><span><b>status</b>insertion repair: not a query point</span><span><b>partitionOrder</b>all lower &lt;= all upper</span><span><b>sizes</b>lower 1, upper 0: final size rule holds</span><span><b>rootSummary</b>lower 5, upper none</span><span><b>current</b>5</span><span><b>lowerStored</b>[-5]</span><span><b>lowerLogical</b>[5]</span><span><b>upper</b>[]</span><span><b>phase</b>push lower</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="transfer-5-upper" hidden role="group" aria-label="Transfer 5 to upper"><div class="coding-trace-frame-heading"><span>Transfer 5 to upper</span><strong>Pop stored negative five from lower. Push logical five into upper. Sizes zero and one still need repair.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>lower max-heap</h4><span class="coding-trace-empty">empty</span></section><section class="coding-trace-heap-group"><h4>upper min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root is-current" data-motion-key="heap-value-5-0">5</span></li></ul></section></div><div class="coding-trace-meta"><span><b>stream</b>[5,2,10,4]</span><span><b>seen</b>5</span><span><b>status</b>insertion repair: not a query point</span><span><b>partitionOrder</b>all lower &lt;= all upper</span><span><b>sizes</b>lower 0, upper 1: final size rule violated</span><span><b>rootSummary</b>lower none, upper 5</span><span><b>current</b>5</span><span><b>lowerStored</b>[]</span><span><b>lowerLogical</b>[]</span><span><b>upper</b>[5]</span><span><b>phase</b>lower-&gt;upper</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="rebalance-5-lower" hidden role="group" aria-label="Rebalance 5 back to lower"><div class="coding-trace-frame-heading"><span>Rebalance 5 back to lower</span><strong>Upper is larger, so transfer its root five back to lower. Sizes one and zero complete the insertion.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>lower max-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-5-0">5</span></li></ul></section><section class="coding-trace-heap-group"><h4>upper min-heap</h4><span class="coding-trace-empty">empty</span></section></div><div class="coding-trace-meta"><span><b>stream</b>[5,2,10,4]</span><span><b>seen</b>5</span><span><b>status</b>completed insertion: query allowed</span><span><b>partitionOrder</b>all lower &lt;= all upper</span><span><b>sizes</b>lower 1, upper 0: final size rule holds</span><span><b>rootSummary</b>lower 5, upper none</span><span><b>lowerStored</b>[-5]</span><span><b>lowerLogical</b>[5]</span><span><b>upper</b>[]</span><span><b>balance</b>1 &gt;= 0 and difference=1</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="median-after-5" hidden role="group" aria-label="Read odd median 5"><div class="coding-trace-frame-heading"><span>Read odd median 5</span><strong>len(lower)=1 is greater than len(upper)=0, so find_median returns float(-lower[0]) = -(-5) = 5.0.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>lower max-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-5-0">5</span></li></ul></section><section class="coding-trace-heap-group"><h4>upper min-heap</h4><span class="coding-trace-empty">empty</span></section></div><div class="coding-trace-meta"><span><b>stream</b>[5,2,10,4]</span><span><b>seen</b>5</span><span><b>status</b>completed insertion: query allowed</span><span><b>partitionOrder</b>all lower &lt;= all upper</span><span><b>sizes</b>lower 1, upper 0: final size rule holds</span><span><b>rootSummary</b>lower 5, upper none</span><span><b>lowerStored</b>[-5]</span><span><b>lowerLogical</b>[5]</span><span><b>upper</b>[]</span><span><b>branch</b>lower larger</span><span><b>result</b>5.0</span></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="push-2-lower" hidden role="group" aria-label="Push 2 into lower"><div class="coding-trace-frame-heading"><span>Push 2 into lower</span><strong>heappush(lower, -2) changes stored lower to [-5,-2], representing max-heap root 5 with child 2.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>lower max-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-5-0">5</span><ul><li><span class="coding-trace-heap-value is-current" data-motion-key="heap-value-2-0">2</span></li></ul></li></ul></section><section class="coding-trace-heap-group"><h4>upper min-heap</h4><span class="coding-trace-empty">empty</span></section></div><div class="coding-trace-meta"><span><b>stream</b>[5,2,10,4]</span><span><b>seen</b>5, 2</span><span><b>status</b>insertion repair: not a query point</span><span><b>partitionOrder</b>all lower &lt;= all upper</span><span><b>sizes</b>lower 2, upper 0: final size rule violated</span><span><b>rootSummary</b>lower 5, upper none</span><span><b>current</b>2</span><span><b>lowerStored</b>[-5,-2]</span><span><b>lowerLogical</b>[5,2]</span><span><b>upper</b>[]</span><span><b>phase</b>push lower</span></div></div><div class="coding-trace-frame" data-coding-frame="6" data-frame-key="transfer-5-after-2" hidden role="group" aria-label="Transfer lower root 5 to upper"><div class="coding-trace-frame-heading"><span>Transfer lower root 5 to upper</span><strong>Transfer lower maximum five to upper. Both heaps now have size one. No further repair is needed.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>lower max-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-2-0">2</span></li></ul></section><section class="coding-trace-heap-group"><h4>upper min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-5-0">5</span></li></ul></section></div><div class="coding-trace-meta"><span><b>stream</b>[5,2,10,4]</span><span><b>seen</b>5, 2</span><span><b>status</b>completed insertion: query allowed</span><span><b>partitionOrder</b>all lower &lt;= all upper</span><span><b>sizes</b>lower 1, upper 1: final size rule holds</span><span><b>rootSummary</b>lower 2, upper 5</span><span><b>lowerStored</b>[-2]</span><span><b>lowerLogical</b>[2]</span><span><b>upper</b>[5]</span><span><b>rebalance</b>1 &gt; 1 is false</span></div></div><div class="coding-trace-frame" data-coding-frame="7" data-frame-key="median-after-2" hidden role="group" aria-label="Read even median 3.5"><div class="coding-trace-frame-heading"><span>Read even median 3.5</span><strong>The heap sizes are equal, so return (-lower[0] + upper[0]) / 2 = (2 + 5) / 2 = 3.5.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>lower max-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-2-0">2</span></li></ul></section><section class="coding-trace-heap-group"><h4>upper min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-5-0">5</span></li></ul></section></div><div class="coding-trace-meta"><span><b>stream</b>[5,2,10,4]</span><span><b>seen</b>5, 2</span><span><b>status</b>completed insertion: query allowed</span><span><b>partitionOrder</b>all lower &lt;= all upper</span><span><b>sizes</b>lower 1, upper 1: final size rule holds</span><span><b>rootSummary</b>lower 2, upper 5</span><span><b>lowerStored</b>[-2]</span><span><b>lowerLogical</b>[2]</span><span><b>upper</b>[5]</span><span><b>arithmetic</b>(2+5)/2</span><span><b>result</b>3.5</span></div></div><div class="coding-trace-frame" data-coding-frame="8" data-frame-key="push-10-lower" hidden role="group" aria-label="Push 10 into lower"><div class="coding-trace-frame-heading"><span>Push 10 into lower</span><strong>Pushing ten gives logical lower [10,2], above upper root five. Sizes fit, but partition order is temporarily wrong.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>lower max-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root is-current" data-motion-key="heap-value-10-0">10</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-2-0">2</span></li></ul></li></ul></section><section class="coding-trace-heap-group"><h4>upper min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-5-0">5</span></li></ul></section></div><div class="coding-trace-meta"><span><b>stream</b>[5,2,10,4]</span><span><b>seen</b>5, 2, 10</span><span><b>status</b>insertion repair: not a query point</span><span><b>partitionOrder</b>violated: lower root 10 &gt; upper root 5</span><span><b>sizes</b>lower 2, upper 1: final size rule holds</span><span><b>rootSummary</b>lower 10, upper 5</span><span><b>current</b>10</span><span><b>lowerStored</b>[-10,-2]</span><span><b>lowerLogical</b>[10,2]</span><span><b>upper</b>[5]</span><span><b>phase</b>push lower</span></div></div><div class="coding-trace-frame" data-coding-frame="9" data-frame-key="transfer-10-upper" hidden role="group" aria-label="Transfer lower root 10 to upper"><div class="coding-trace-frame-heading"><span>Transfer lower root 10 to upper</span><strong>Pop stored -10, leaving lower [-2]. Push 10 into upper [5,10], whose min-root remains 5.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>lower max-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-2-0">2</span></li></ul></section><section class="coding-trace-heap-group"><h4>upper min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-5-0">5</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-10-0">10</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>stream</b>[5,2,10,4]</span><span><b>seen</b>5, 2, 10</span><span><b>status</b>insertion repair: not a query point</span><span><b>partitionOrder</b>all lower &lt;= all upper</span><span><b>sizes</b>lower 1, upper 2: final size rule violated</span><span><b>rootSummary</b>lower 2, upper 5</span><span><b>lowerStored</b>[-2]</span><span><b>lowerLogical</b>[2]</span><span><b>upper</b>[5,10]</span><span><b>balance</b>upper size 2 &gt; lower size 1</span></div></div><div class="coding-trace-frame" data-coding-frame="10" data-frame-key="rebalance-5-after-10" hidden role="group" aria-label="Rebalance upper root 5"><div class="coding-trace-frame-heading"><span>Rebalance upper root 5</span><strong>Upper is larger, so move its minimum five back to lower. Lower [5,2] and upper [10] satisfy both conditions.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>lower max-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-5-0">5</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-2-0">2</span></li></ul></li></ul></section><section class="coding-trace-heap-group"><h4>upper min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-10-0">10</span></li></ul></section></div><div class="coding-trace-meta"><span><b>stream</b>[5,2,10,4]</span><span><b>seen</b>5, 2, 10</span><span><b>status</b>completed insertion: query allowed</span><span><b>partitionOrder</b>all lower &lt;= all upper</span><span><b>sizes</b>lower 2, upper 1: final size rule holds</span><span><b>rootSummary</b>lower 5, upper 10</span><span><b>lowerStored</b>[-5,-2]</span><span><b>lowerLogical</b>[5,2]</span><span><b>upper</b>[10]</span><span><b>balance</b>sizes 2 and 1</span></div></div><div class="coding-trace-frame" data-coding-frame="11" data-frame-key="median-after-10" hidden role="group" aria-label="Read odd median 5"><div class="coding-trace-frame-heading"><span>Read odd median 5</span><strong>lower has one extra value, so return -lower[0] = -(-5) = 5.0.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>lower max-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-5-0">5</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-2-0">2</span></li></ul></li></ul></section><section class="coding-trace-heap-group"><h4>upper min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-10-0">10</span></li></ul></section></div><div class="coding-trace-meta"><span><b>stream</b>[5,2,10,4]</span><span><b>seen</b>5, 2, 10</span><span><b>status</b>completed insertion: query allowed</span><span><b>partitionOrder</b>all lower &lt;= all upper</span><span><b>sizes</b>lower 2, upper 1: final size rule holds</span><span><b>rootSummary</b>lower 5, upper 10</span><span><b>lowerStored</b>[-5,-2]</span><span><b>lowerLogical</b>[5,2]</span><span><b>upper</b>[10]</span><span><b>branch</b>lower larger</span><span><b>result</b>5.0</span></div></div><div class="coding-trace-frame" data-coding-frame="12" data-frame-key="push-4-lower" hidden role="group" aria-label="Push 4 into lower"><div class="coding-trace-frame-heading"><span>Push 4 into lower</span><strong>heappush(lower, -4) produces stored [-5,-2,-4], representing logical max-heap [5,2,4].</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>lower max-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-5-0">5</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-2-0">2</span></li><li><span class="coding-trace-heap-value is-current" data-motion-key="heap-value-4-0">4</span></li></ul></li></ul></section><section class="coding-trace-heap-group"><h4>upper min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-10-0">10</span></li></ul></section></div><div class="coding-trace-meta"><span><b>stream</b>[5,2,10,4]</span><span><b>seen</b>5, 2, 10, 4</span><span><b>status</b>insertion repair: not a query point</span><span><b>partitionOrder</b>all lower &lt;= all upper</span><span><b>sizes</b>lower 3, upper 1: final size rule violated</span><span><b>rootSummary</b>lower 5, upper 10</span><span><b>current</b>4</span><span><b>lowerStored</b>[-5,-2,-4]</span><span><b>lowerLogical</b>[5,2,4]</span><span><b>upper</b>[10]</span><span><b>phase</b>push lower</span></div></div><div class="coding-trace-frame" data-coding-frame="13" data-frame-key="transfer-5-after-4" hidden role="group" aria-label="Transfer lower root 5 to upper"><div class="coding-trace-frame-heading"><span>Transfer lower root 5 to upper</span><strong>Transfer lower maximum five to upper. Lower [4,2] and upper [5,10] have equal sizes, completing the insertion.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>lower max-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-4-0">4</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-2-0">2</span></li></ul></li></ul></section><section class="coding-trace-heap-group"><h4>upper min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-5-0">5</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-10-0">10</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>stream</b>[5,2,10,4]</span><span><b>seen</b>5, 2, 10, 4</span><span><b>status</b>completed insertion: query allowed</span><span><b>partitionOrder</b>all lower &lt;= all upper</span><span><b>sizes</b>lower 2, upper 2: final size rule holds</span><span><b>rootSummary</b>lower 4, upper 5</span><span><b>lowerStored</b>[-4,-2]</span><span><b>lowerLogical</b>[4,2]</span><span><b>upper</b>[5,10]</span><span><b>rebalance</b>2 &gt; 2 is false</span></div></div><div class="coding-trace-frame" data-coding-frame="14" data-frame-key="median-after-4" hidden role="group" aria-label="Read final even median 4.5"><div class="coding-trace-frame-heading"><span>Read final even median 4.5</span><strong>Equal heap sizes select both roots: (-lower[0] + upper[0]) / 2 = (4 + 5) / 2 = 4.5.</strong></div><div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology"><section class="coding-trace-heap-group"><h4>lower max-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-4-0">4</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-2-0">2</span></li></ul></li></ul></section><section class="coding-trace-heap-group"><h4>upper min-heap</h4><ul class="coding-trace-heap-tree"><li><span class="coding-trace-heap-value is-root" data-motion-key="heap-value-5-0">5</span><ul><li><span class="coding-trace-heap-value" data-motion-key="heap-value-10-0">10</span></li></ul></li></ul></section></div><div class="coding-trace-meta"><span><b>stream</b>[5,2,10,4]</span><span><b>seen</b>5, 2, 10, 4</span><span><b>status</b>completed insertion: query allowed</span><span><b>partitionOrder</b>all lower &lt;= all upper</span><span><b>sizes</b>lower 2, upper 2: final size rule holds</span><span><b>rootSummary</b>lower 4, upper 5</span><span><b>lowerStored</b>[-4,-2]</span><span><b>lowerLogical</b>[4,2]</span><span><b>upper</b>[5,10]</span><span><b>arithmetic</b>(4+5)/2</span><span><b>result</b>4.5</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 15</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize two empty heaps</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Push 5 into lower</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Transfer 5 to upper</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Rebalance 5 back to lower</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Read odd median 5</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Push 2 into lower</strong></button><button type="button" data-coding-frame-button="6"><span>7</span><strong>Transfer lower root 5 to upper</strong></button><button type="button" data-coding-frame-button="7"><span>8</span><strong>Read even median 3.5</strong></button><button type="button" data-coding-frame-button="8"><span>9</span><strong>Push 10 into lower</strong></button><button type="button" data-coding-frame-button="9"><span>10</span><strong>Transfer lower root 10 to upper</strong></button><button type="button" data-coding-frame-button="10"><span>11</span><strong>Rebalance upper root 5</strong></button><button type="button" data-coding-frame-button="11"><span>12</span><strong>Read odd median 5</strong></button><button type="button" data-coding-frame-button="12"><span>13</span><strong>Push 4 into lower</strong></button><button type="button" data-coding-frame-button="13"><span>14</span><strong>Transfer lower root 5 to upper</strong></button><button type="button" data-coding-frame-button="14"><span>15</span><strong>Read final even median 4.5</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use two heaps when numbers arrive online and every median query must avoid sorting the entire history.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">After each completed insertion, every lower value is at most every upper value, and lower has either equal size or one extra item. These two conditions expose the middle ranks at roots. Repair frames need not satisfy them.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Maintain an ordered partition whose size difference is bounded, then answer rank queries from boundary roots. The same balancing idea supports streaming quantiles and running order statistics.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Both heaps start empty, so a median query is undefined. Negated lower values make its stored minimum the logical maximum. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

## Code

```python
import heapq

class MedianFinder:
   def __init__(self) -> None:
      self.lower: list[int] = []
      self.upper: list[int] = []

   def add_num(self, num: int) -> None:
      heapq.heappush(self.lower, -num)
      heapq.heappush(self.upper, -heapq.heappop(self.lower))
      if len(self.upper) > len(self.lower):
         heapq.heappush(self.lower, -heapq.heappop(self.upper))

   def find_median(self) -> float:
      if len(self.lower) > len(self.upper):
         return float(-self.lower[0])
      return (-self.lower[0] + self.upper[0]) / 2
```

**Cost:** After $n$ insertions, updates take $O(\log(n+1))$ heap work, queries $O(1)$ root operations, and retained storage $O(n)$ values.
These bounds exclude integer-width and conversion costs.

<!-- learning:core-end -->

## Change Numeric Output or the Rank

Start from `MedianFinder`. Change either numeric representation or which rank the partition exposes.

1. **Keep exact integer medians.**

   **New task:** Avoid rounding or overflow from float conversion.

   **Picture and change:** Reuse source insertion unchanged. Build a `Fraction` from one root or the integer sum of two roots.
   The helper returns strings only to display a prefix trace.

   **Code:** Complete helper using the source class above:

   ```python
   from fractions import Fraction


   def exact_median_trace(values):
      finder, answer = MedianFinder(), []
      for value in values:
         finder.add_num(value)
         if len(finder.lower) > len(finder.upper):
            median = Fraction(-finder.lower[0], 1)
         else:
            median = Fraction(-finder.lower[0] + finder.upper[0], 2)
         answer.append(str(median))
      return answer
   ```

   **Check:** `[5,2,10,4]` gives `["5","7/2","5","9/2"]`.
   The singleton integer `10000000000000001` remains exact, unlike its rounded float conversion.

   **Use when:** Integer-rank arithmetic must be exact.
   Fraction construction, formatting and returned strings add bit-width work and output storage.

2. **Choose one nearest-rank quantile.**

   **New task:** Return rank $\lceil qn\rceil$, without midpoint averaging.

   **Picture and change:** Define $q=numerator/denominator$ using positive integers with `numerator <= denominator`.
   Keep that target number of items in lower. Place each new value by partition order, then transfer boundary roots to repair size.

   **Code:** Complete replacement:

   ```python
   import heapq


   def quantile_trace(values, numerator, denominator):
      if not 0 < numerator <= denominator:
         raise ValueError("invalid quantile")
      lower, upper, answer = [], [], []
      for count, value in enumerate(values, 1):
         if not lower or value <= -lower[0]:
            heapq.heappush(lower, -value)
         else:
            heapq.heappush(upper, value)
         needed = (numerator * count + denominator - 1) // denominator
         while len(lower) > needed:
            heapq.heappush(upper, -heapq.heappop(lower))
         while len(lower) < needed:
            heapq.heappush(lower, -heapq.heappop(upper))
         answer.append(-lower[0])
      return answer
   ```

   **Check:** `[5,2,10,4]` gives `[5,2,2,2]` for $q=1/4$, or `[5,5,10,5]` for $q=3/4$.
   At $q=1/2$, even samples return the lower middle value, not the averaged median.

   **Use when:** One fixed rank definition replaces the median.
   Updates remain logarithmic and retain all values; returned answers add storage. This is not a bounded-memory quantile sketch.

## Rebuild the Retention Decision

Why not discard values outside a bounded top-$k$ heap?

**Answer:** Future insertions can move the middle rank. Exact median maintenance retains both ordered halves, unlike [bounded top-$k$ retention](/questions/kth-largest-element/).
