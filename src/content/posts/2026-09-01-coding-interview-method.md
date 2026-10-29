---
title: "How to learn coding interview problems without memorizing them"
description: "Use cues, state, invariants, and spaced rebuilding to turn coding problems into recognizable mental models."
date: "2026-09-01"
draft: false
tags: ["coding interview", "learning method"]
category: "guides"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Foundation"
priority: "Core"
aliases: ["coding interview practice", "coding questions phone guide"]
prerequisites: []
---

Two values must add to a target, and their positions must differ.
Each new value determines a missing partner. Query earlier values before saving the current value.

## Problem

Given `nums = [3, 2, 4, 7]` and `target = 6`, return one pair of distinct original indices.
The result is `[1, 2]`: `2 + 4 == 6`. Return `[]` when no pair exists.
Do not sort or reuse one position. Assume integer inputs.

## Picture: Remember Earlier Partners

<!-- visual:coding-interview-method-state -->
<figure class="learning-figure editorial-illustration" aria-labelledby="coding-interview-method-state-edition-title"><span class="visual-fragment-anchor" id="coding-interview-method-state-title" aria-hidden="true"></span><p class="visual-title" id="coding-interview-method-state-edition-title">Look up the missing value, then return its saved earlier index</p><svg viewBox="0 0 360 618" role="img" aria-labelledby="coding-interview-method-state-edition-svg-title coding-interview-method-state-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="coding-interview-method-state-edition-svg-title">The complement is a value; the saved map entry recovers its earlier index</title><desc id="coding-interview-method-state-edition-svg-desc">For toy input four, two, seven and target nine, current index two contains seven. Nine minus seven is two. The earlier-value map contains four to index zero and two to index one. Looking up two returns one, so indices one and two identify two plus seven. The current index is absent during the query; insertion happens only after a miss.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Toy input [4, 2, 7]; target 9</tspan></text><rect x="72" y="53" width="72" height="46" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="108" y="83" style="font-size:17px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="108" dy="0">4</tspan></text><text x="108" y="121" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="108" dy="0">index 0</tspan></text><rect x="144" y="53" width="72" height="46" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="83" style="font-size:17px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">2</tspan></text><text x="180" y="121" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">index 1</tspan></text><rect x="216" y="53" width="72" height="46" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="252" y="83" style="font-size:17px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="252" dy="0">7</tspan></text><text x="252" y="121" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="252" dy="0">index 2</tspan></text><g data-editorial-node><rect x="20" y="154" width="320" height="65" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="180" y="182" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Current index 2: value 7</tspan><tspan x="180" dy="19">complement = target 9 - 7 = 2</tspan></text></g><path d="M180 219L180 254" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,247.8569220667674L180,254L183.35597877022943,247.8569220667674" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><text x="180" y="276" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">seen contains ONLY earlier positions</tspan></text><text x="110" y="309" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="110" dy="0">value</tspan></text><text x="252" y="309" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="252" dy="0">earlier index</tspan></text><path d="M70 319L310 319" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="110" y="345" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="110" dy="0">4</tspan></text><text x="252" y="345" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="252" dy="0">0</tspan></text><rect x="70" y="357" width="240" height="36" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="110" y="381" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="110" dy="0">2</tspan></text><text x="252" y="381" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="252" dy="0">1</tspan></text><text x="180" y="425" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Lookup 2 returns earlier index 1.</tspan></text><g data-editorial-node><rect x="20" y="450" width="320" height="60" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="180" y="475.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Return [1, 2]: 2 + 7 = 9</tspan><tspan x="180" dy="19">Two different input positions</tspan></text></g><text x="180" y="549" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Index 2 was absent during the query.</tspan></text><text x="180" y="579" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">On a miss, store the current value only after</tspan><tspan x="180" dy="19">querying.</tspan></text></svg><figcaption><strong>Read it this way:</strong> Follow value 2 from the complement calculation to the saved index 1. Returning [1,2] uses two different positions. Query-before-store is the invariant, not a variable spelling. <span class="visual-reader-check" data-reader-check="coding-interview-method-state"><strong>Check:</strong> How does the matching value recover an earlier index without reusing the current position?</span></figcaption></figure>

| index | num | needed | seen before query | Action |
| ---: | ---: | ---: | --- | --- |
| 0 | 3 | 3 | `{}` | Save `3: 0` |
| 1 | 2 | 4 | `{3: 0}` | Save `2: 1` |
| 2 | 4 | 2 | `{3: 0, 2: 1}` | Return `[1, 2]` |

## The four things to remember

### Cue

A missing partner suggests replacing repeated scans; the clue alone is not a proof.

### State

`seen` retains one earlier index per value. Any such index suffices for one witness, but not for counting every pair.

### Invariant

Before each query, `seen` represents all earlier values, with an earlier index for each.

### Move

Query, then store. A hit is valid; otherwise saving the current value preserves the invariant for the next index.

## Code

This is the literal [Two Sum owner](/questions/two-sum/) implementation:

```python
def two_sum(nums: list[int], target: int) -> list[int]:
   seen: dict[int, int] = {}
   for index, num in enumerate(nums):
      needed = target - num
      if needed in seen:
         return [seen[needed], index]
      seen[num] = index
   return []
```

Under usual dictionary and unit-cost arithmetic assumptions, time is expected $O(n)$ and auxiliary space is $O(n)$.
Check `[3]`, target 6: no earlier partner, so `[]`. Check `[3, 3]`: the second query returns `[0, 1]`.

## Change 1: Count Every Index Pair

**New task.** Return the number of distinct index pairs, not one witness.

**Picture and change.** Keep prior frequencies rather than one index. Add matches before incrementing the current frequency.

**Code.**

```python
def count_pairs(nums, target):
	counts = {}
	answer = 0
	for value in nums:
		answer += counts.get(target - value, 0)
		counts[value] = counts.get(value, 0) + 1
	return answer
```

**Check.** `[3, 3, 3]`, target 6, adds 0, 1, then 2: total 3. `[3]` gives 0.

**Use when.** The output asks how many witnesses exist.

## Change 2: Count Signed Zero-Sum Ranges

**New task.** Without a pattern heading, count contiguous zero-sum ranges in `[1, -1, 1]`.

**Picture and change.** Equal prefix totals mark range boundaries. Store boundary multiplicity, including the initial zero boundary.

**Code.**

```python
def count_zero_ranges(nums):
	counts = {0: 1}
	prefix = answer = 0
	for value in nums:
		prefix += value
		answer += counts.get(prefix, 0)
		counts[prefix] = counts.get(prefix, 0) + 1
	return answer
```

**Check.** Prefixes `0, 1, 0, 1` contain two equal-boundary pairs, so the answer is 2. Empty input gives 0.

**Use when.** Signed sums lack the monotonicity needed for a simple shrinking window.

## Failure: Remembering Only "Hash Map"

A family name cannot explain query order, state multiplicity or the assumptions behind a safe move.

## Transfer Check

Why does a contiguous range not always imply a sliding window?

### Reasoned Answer

Adding a negative value can decrease a sum. Without the needed repair monotonicity, derive the prefix relation instead.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## A three-pass practice loop

### First pass: learn one anchor per pattern

These 20 anchors form one suggested sequence, not a universal optimum:

1. Two Sum
2. Subarray Sum Equals K
3. 3Sum
4. Longest Substring Without Repeating Characters
5. Binary Search
6. Valid Parentheses
7. Daily Temperatures
8. Rotting Oranges
9. Kth Largest Element
10. Network Delay Time
11. Number of Islands
12. Maximum Depth of Binary Tree
13. Subsets
14. House Robber
15. Merge Intervals
16. Jump Game
17. Course Schedule
18. Redundant Connection
19. Reverse Linked List
20. Implement Trie

Read the problem first; hide code, inspect the picture, explain the invariant, then implement and test.

### Second pass: study nearby variations

Change the output, admissible inputs or state requirements; re-prove the move. A surface resemblance does not establish correctness.

### Mixed pass: choose the pattern yourself

Hide the pattern heading and ask:

1. What is the brute-force method?
2. What work does it repeat?
3. Which state removes that repeated work?
4. What rule stays true?
5. Why is the next move safe?
6. What are the time and space costs?

An optional 15-minute attempt can precede a limited hint. Hide the hint and rebuild; adjust timing to observed needs.

## Review schedule

Optional retry delays are 1, 3, 7 and 14 days; adapt them to performance, not a learning guarantee.
Record six miss types:

- Recognition.
- Wrong state.
- Unsafe move.
- Implementation.
- Edge case.
- Complexity.

Use a card for retrieval and the full lesson to repair the explanation.

## Quick pattern map

| Prompt shape | State to draw first | Likely family |
| --- | --- | --- |
| Pair/duplicate | Prior values/counts | Map/set |
| Contiguous range | Boundaries/prefixes | Window/prefix |
| Sorted values | Ends/answer bounds | Pointers/bisection |
| Nested input | Unfinished items | Stack |
| Next greater | Waiting indices | Monotonic stack |
| Fewest unweighted steps | Layers | BFS |
| Nonnegative weighted path | Cheapest frontier | Dijkstra |
| Reachability | Visited frontier | Graph search |
| Every valid choice | Path/options | Backtracking |
| Repeated subproblem | Complete cached state | Dynamic programming |
| Time ranges | Active end | Intervals/greedy |
| Prerequisites | Incoming counts | Topological sort |
| Joining groups | Roots | Union-find |
| Node reordering | Saved next | Linked list |
| Shared prefixes | Prefix path | Trie |

Continue at [Learn by rebuilding](/library/coding-interview/method/).

</div>
