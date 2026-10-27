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
<figure class="learning-figure coding-visual-figure" aria-labelledby="coding-interview-method-state-title">
	<p class="visual-kicker">A four-question rehearsal</p>
	<p class="visual-title" id="coding-interview-method-state-title">Turn the missing partner into sufficient state</p>
	<div class="coding-visual coding-visual--state" role="img" aria-label="For target six, index two has value four and needs two. Earlier value two maps to index one. Query before storing the current index, so the returned indices one and two differ.">
		<div class="coding-visual-example"><span>At index 2</span><strong>Need 2 -> return [1, 2]</strong></div>
		<div class="coding-visual-flow">
			<div class="coding-visual-step"><span class="coding-visual-step-label">1. Cue</span><strong>Repeated partner search</strong><small>Each num needs target - num.</small></div>
			<div class="coding-visual-step"><span class="coding-visual-step-label">2. State</span><strong>seen: value -> index</strong><small>One earlier index per value suffices for one witness.</small></div>
			<div class="coding-visual-step"><span class="coding-visual-step-label">3. Invariant</span><strong>Every saved index is earlier</strong><small>The current position is absent during its query.</small></div>
			<div class="coding-visual-step"><span class="coding-visual-step-label">4. Move</span><strong>Query, then store</strong><small>This prevents matching one position with itself.</small></div>
		</div>
		<p class="coding-visual-invariant"><span>Rebuild</span>Hide the code, explain the invariant, implement it and test a changed input.</p>
	</div>
	<figcaption><strong>Read it this way:</strong> replace the repeated partner scan with earlier-value state. Query before storing to guarantee distinct indices. Remember the proof, not a variable spelling.</figcaption>
</figure>

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
