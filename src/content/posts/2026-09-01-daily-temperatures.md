---
title: "Daily Temperatures"
description: "For each day, find how many days pass before a warmer temperature."
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

Keep indices of days still waiting for a strictly warmer temperature.
A new warmer day resolves every colder index it pops, at that index's first qualifying later day.

## Problem

For `[73, 74, 75, 71, 69, 72, 76, 73]`, return `[1, 1, 4, 2, 1, 1, 0, 0]`.
Zero means no qualifying later day in this input, not a zero-day wait. Empty input returns `[]`.

## Picture: Resolve Only the Colder Waiting Days

<!-- visual:daily-temperatures-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="daily-temperatures-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="daily-temperatures-state-title">Daily Temperatures: Unresolved temperatures are nonincreasing; a strictly warmer day pops waiting indices and writes their first qualifying distances.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="daily-temperatures" role="group" tabindex="0" aria-label="Daily Temperatures: Unresolved temperatures are nonincreasing; a strictly warmer day pops waiting indices and writes their first qualifying distances."><div class="coding-visual-example"><span>Input and goal</span><strong>For each day, find how many days pass before a warmer temperature.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize" role="group" aria-label="Initialize unanswered days"><div class="coding-trace-frame-heading"><span>Initialize unanswered days</span><strong>Initialize eight zero answers and an empty stack of waiting indices.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>empty</span><small>top waiting day</small></span></div><div class="coding-trace-meta"><span><b>temperatures</b>[73, 74, 75, 71, 69, 72, 76, 73]</span><span><b>layout</b>oldest to newest</span><span><b>current</b>before day 0</span><span><b>answer</b>[0,0,0,0,0,0,0,0]</span><span><b>action</b>initialize</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="push-day-0" hidden role="group" aria-label="Push day 0"><div class="coding-trace-frame-heading"><span>Push day 0</span><strong>Day 0 is 73. No earlier day waits, so append index 0.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>day 0: 73</span><small>top waiting day</small></span></div><div class="coding-trace-meta"><span><b>temperatures</b>[73, 74, 75, 71, 69, 72, 76, 73]</span><span><b>layout</b>oldest to newest</span><span><b>current</b>day 0: 73</span><span><b>answer</b>[0,0,0,0,0,0,0,0]</span><span><b>action</b>push 0</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="resolve-day-0" hidden role="group" aria-label="Resolve day 0 with day 1"><div class="coding-trace-frame-heading"><span>Resolve day 0 with day 1</span><strong>74 &gt; 73, so pop day 0 and write answer[0] = 1 - 0 = 1; then push day 1.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>day 1: 74</span><small>top waiting day</small></span></div><div class="coding-trace-meta"><span><b>temperatures</b>[73, 74, 75, 71, 69, 72, 76, 73]</span><span><b>layout</b>oldest to newest</span><span><b>current</b>day 1: 74</span><span><b>answer</b>[1,0,0,0,0,0,0,0]</span><span><b>action</b>pop 0; 1 - 0 = 1; push 1</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="resolve-day-1" hidden role="group" aria-label="Resolve day 1 with day 2"><div class="coding-trace-frame-heading"><span>Resolve day 1 with day 2</span><strong>75 &gt; 74, so pop day 1 and write answer[1] = 2 - 1 = 1; then push day 2.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>day 2: 75</span><small>top waiting day</small></span></div><div class="coding-trace-meta"><span><b>temperatures</b>[73, 74, 75, 71, 69, 72, 76, 73]</span><span><b>layout</b>oldest to newest</span><span><b>current</b>day 2: 75</span><span><b>answer</b>[1,1,0,0,0,0,0,0]</span><span><b>action</b>pop 1; 2 - 1 = 1; push 2</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="push-day-3" hidden role="group" aria-label="Keep cooler day 3 waiting"><div class="coding-trace-frame-heading"><span>Keep cooler day 3 waiting</span><strong>71 is not warmer than stack-top 75, so no answer is safe yet; push day 3.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>day 2: 75</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>day 3: 71</span><small>top waiting day</small></span></div><div class="coding-trace-meta"><span><b>temperatures</b>[73, 74, 75, 71, 69, 72, 76, 73]</span><span><b>layout</b>oldest to newest</span><span><b>current</b>day 3: 71</span><span><b>answer</b>[1,1,0,0,0,0,0,0]</span><span><b>action</b>71 &lt;= 75; push 3</span></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="push-day-4" hidden role="group" aria-label="Keep cooler day 4 waiting"><div class="coding-trace-frame-heading"><span>Keep cooler day 4 waiting</span><strong>69 is not warmer than stack-top 71, so append day 4 and preserve decreasing temperatures.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>day 2: 75</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>day 3: 71</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>day 4: 69</span><small>top waiting day</small></span></div><div class="coding-trace-meta"><span><b>temperatures</b>[73, 74, 75, 71, 69, 72, 76, 73]</span><span><b>layout</b>oldest to newest</span><span><b>current</b>day 4: 69</span><span><b>answer</b>[1,1,0,0,0,0,0,0]</span><span><b>action</b>69 &lt;= 71; push 4</span></div></div><div class="coding-trace-frame" data-coding-frame="6" data-frame-key="resolve-day-4" hidden role="group" aria-label="Day 5 resolves day 4"><div class="coding-trace-frame-heading"><span>Day 5 resolves day 4</span><strong>72 &gt; 69, so the while loop pops day 4 and writes answer[4] = 5 - 4 = 1.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>day 2: 75</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>day 3: 71</span><small>top waiting day</small></span></div><div class="coding-trace-meta"><span><b>temperatures</b>[73, 74, 75, 71, 69, 72, 76, 73]</span><span><b>layout</b>oldest to newest</span><span><b>current</b>day 5: 72</span><span><b>answer</b>[1,1,0,0,1,0,0,0]</span><span><b>action</b>pop 4; 5 - 4 = 1</span></div></div><div class="coding-trace-frame" data-coding-frame="7" data-frame-key="resolve-day-3" hidden role="group" aria-label="Day 5 also resolves day 3"><div class="coding-trace-frame-heading"><span>Day 5 also resolves day 3</span><strong>The same 72 &gt; new top 71, so pop day 3, write answer[3] = 5 - 3 = 2, then push day 5 below 75.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>day 2: 75</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>day 5: 72</span><small>top waiting day</small></span></div><div class="coding-trace-meta"><span><b>temperatures</b>[73, 74, 75, 71, 69, 72, 76, 73]</span><span><b>layout</b>oldest to newest</span><span><b>current</b>day 5: 72</span><span><b>answer</b>[1,1,0,2,1,0,0,0]</span><span><b>action</b>pop 3; 5 - 3 = 2; push 5</span></div></div><div class="coding-trace-frame" data-coding-frame="8" data-frame-key="resolve-day-5" hidden role="group" aria-label="Day 6 resolves day 5"><div class="coding-trace-frame-heading"><span>Day 6 resolves day 5</span><strong>76 &gt; 72, so pop day 5 and write answer[5] = 6 - 5 = 1.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>day 2: 75</span><small>top waiting day</small></span></div><div class="coding-trace-meta"><span><b>temperatures</b>[73, 74, 75, 71, 69, 72, 76, 73]</span><span><b>layout</b>oldest to newest</span><span><b>current</b>day 6: 76</span><span><b>answer</b>[1,1,0,2,1,1,0,0]</span><span><b>action</b>pop 5; 6 - 5 = 1</span></div></div><div class="coding-trace-frame" data-coding-frame="9" data-frame-key="resolve-day-2" hidden role="group" aria-label="Day 6 also resolves day 2"><div class="coding-trace-frame-heading"><span>Day 6 also resolves day 2</span><strong>76 &gt; new top 75, so pop day 2, write answer[2] = 6 - 2 = 4, then push day 6.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>day 6: 76</span><small>top waiting day</small></span></div><div class="coding-trace-meta"><span><b>temperatures</b>[73, 74, 75, 71, 69, 72, 76, 73]</span><span><b>layout</b>oldest to newest</span><span><b>current</b>day 6: 76</span><span><b>answer</b>[1,1,4,2,1,1,0,0]</span><span><b>action</b>pop 2; 6 - 2 = 4; push 6</span></div></div><div class="coding-trace-frame" data-coding-frame="10" data-frame-key="finish" hidden role="group" aria-label="Leave final cooler day waiting"><div class="coding-trace-frame-heading"><span>Leave final cooler day waiting</span><strong>Day 7 is 73, not warmer than 76, so push it; unresolved days 6 and 7 correctly retain zero.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>day 6: 76</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>day 7: 73</span><small>top waiting day</small></span></div><div class="coding-trace-meta"><span><b>temperatures</b>[73, 74, 75, 71, 69, 72, 76, 73]</span><span><b>layout</b>oldest to newest</span><span><b>current</b>day 7: 73</span><span><b>answer</b>[1,1,4,2,1,1,0,0]</span><span><b>action</b>73 &lt;= 76; push 7</span><span><b>result</b>[1,1,4,2,1,1,0,0]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 11</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize unanswered days</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Push day 0</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Resolve day 0 with day 1</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Resolve day 1 with day 2</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Keep cooler day 3 waiting</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Keep cooler day 4 waiting</strong></button><button type="button" data-coding-frame-button="6"><span>7</span><strong>Day 5 resolves day 4</strong></button><button type="button" data-coding-frame-button="7"><span>8</span><strong>Day 5 also resolves day 3</strong></button><button type="button" data-coding-frame-button="8"><span>9</span><strong>Day 6 resolves day 5</strong></button><button type="button" data-coding-frame-button="9"><span>10</span><strong>Day 6 also resolves day 2</strong></button><button type="button" data-coding-frame-button="10"><span>11</span><strong>Leave final cooler day waiting</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Look for the first later greater value or the waiting distance to it; those words suggest keeping unresolved indices until a new value is large enough to answer them.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">After each day is appended, waiting indices increase from bottom to top while their temperatures are non-increasing, and every popped index receives its first strictly warmer later day.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">For next-greater or next-smaller problems, store unresolved indices in a monotonic stack; the first value that violates the monotonic order resolves every eligible item popped from the top.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Initialize eight zero answers and an empty stack of waiting indices. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

Indices increase from bottom to top; their temperatures are **nonincreasing**, so equal values can remain together.
This order makes every resolvable colder day accessible through consecutive pops at the top.

| New day | Resolution | Still waiting afterward |
| --- | --- | --- |
| 3: 71 | None | Days 2, 3 |
| 4: 69 | None | Days 2, 3, 4 |
| 5: 72 | Day 4 waits 1; day 3 waits 2 | Days 2, 5 |
| 6: 76 | Day 5 waits 1; day 2 waits 4 | Day 6 |
| 7: 73 | None | Days 6, 7; final sentinels stay zero |

If an earlier warmer day existed, it would already have popped the waiting index.
Thus a pop certifies the **first** qualifying day, not merely some warmer day.
Store indices, not just temperatures: `day-earlier_day` supplies the distance, and equal values can have different waits.

## Code

```python
def daily_temperatures(temperatures: list[int]) -> list[int]:
   answer = [0] * len(temperatures)
   waiting: list[int] = []

   for day, temperature in enumerate(temperatures):
      while waiting and temperatures[waiting[-1]] < temperature:
         earlier_day = waiting.pop()
         answer[earlier_day] = day - earlier_day
      waiting.append(day)

   return answer
```

Time and space are $O(n)$ under ordinary comparison costs.
Each index is pushed once and popped at most once; some never pop. Total inner-loop work is linear.

## Change 1: Equal Temperature Also Qualifies

**New task.** Find the first later day at least as warm.

**Picture and change.** Pop with `<=`; equal days now resolve, leaving strictly decreasing waiting temperatures.

**Code.**

```python
def days_until_at_least(temperatures):
   answer = [0] * len(temperatures)
   waiting = []
   for day, temperature in enumerate(temperatures):
      while waiting and temperatures[waiting[-1]] <= temperature:
         earlier = waiting.pop()
         answer[earlier] = day - earlier
      waiting.append(day)
   return answer
```

**Check.** `[70, 70, 71]` gives `[1, 1, 0]`, instead of the original `[2, 1, 0]`.

**Use when.** Equality qualifies; this is a changed contract, not an optimization.

## Change 2: Find the Next Strictly Smaller Value

**New task.** Return distances to the first later smaller value.

**Picture and change.** Reverse the inequality; waiting values become nondecreasing. Keep indices and the no-answer zero sentinel.

**Code.**

```python
def days_until_smaller(values):
   answer = [0] * len(values)
   waiting = []
   for index, value in enumerate(values):
      while waiting and values[waiting[-1]] > value:
         earlier = waiting.pop()
         answer[earlier] = index - earlier
      waiting.append(index)
   return answer
```

**Check.** `[3, 2, 2, 1]` gives `[1, 2, 1, 0]`; equal 2 does not resolve the earlier 2.

**Use when.** The qualifying direction reverses while the first-answer proof still applies.

## Failure: Losing Strictness or Position

Replacing `<` with `<=` incorrectly accepts equal temperatures under the original task.
Storing only values loses the coordinates needed for wait distances.

## Transfer Check

Why is index order alone insufficient?

### Reasoned Answer

It cannot identify which waiting days the new value resolves. Temperature monotonicity makes eligible indices contiguous at the top.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Unresolved Does Not Mean Impossible

During an ongoing stream, a waiting day may still receive a warmer future day.
The final zero sentinel means no such day occurred within the completed input, not that none could ever occur.

Equal values must remain separately represented for strictly warmer queries.
In `[70, 70, 71]`, the two 70-degree days resolve together but have waits 2 and 1.
Deduplicating them would lose an output position.

Both adaptations keep the one-push/at-most-one-pop accounting and linear auxiliary/output storage.
Changing a threshold or adding expiry rules needs another proof of which waiting entries a new item can safely resolve.
The [bracket stack](/questions/valid-parentheses/) also stores unfinished work, but its pop rule follows nesting rather than temperature order.

</div>
