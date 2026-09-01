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

The goal is not to remember 106 solutions. The goal is to recognize the state a problem needs, preserve its invariant, and rebuild the code when the surface details change.

<!-- visual:coding-interview-method-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="coding-interview-method-state-title"><p class="visual-kicker">A four-question rehearsal</p><p class="visual-title" id="coding-interview-method-state-title">Turn a new prompt into a small state machine</p><div class="coding-visual coding-visual--state" role="img" aria-label="A four-step coding interview learning loop: identify the cue, name the state, explain the invariant, and make the safe move. The loop ends by rebuilding the solution and reviewing it later."><div class="coding-visual-example"><span>Memory target</span><strong>Cue -> state -> invariant -> move</strong></div><div class="coding-visual-flow"><div class="coding-visual-step"><span class="coding-visual-step-label">1. Cue</span><strong>What repeats?</strong><small>Pair, range, dependency, path, or state.</small></div><div class="coding-visual-step"><span class="coding-visual-step-label">2. State</span><strong>What survives?</strong><small>Keep only facts future steps may need.</small></div><div class="coding-visual-step"><span class="coding-visual-step-label">3. Invariant</span><strong>What stays true?</strong><small>State the rule that makes the next move safe.</small></div><div class="coding-visual-step"><span class="coding-visual-step-label">4. Move</span><strong>Why now?</strong><small>Advance, choose, merge, pop, or update.</small></div></div><p class="coding-visual-invariant"><span>Rebuild</span>Close the page, write the algorithm, test one edge case, and revisit after 1, 3, 7, and 14 days.</p></div><figcaption><strong>Read it this way:</strong> begin with the repeated work, then name what must remain in memory. The invariant is the bridge between the state and the move. That bridge is what you want to remember, not the exact variable names.</figcaption></figure>

## The four things to remember

### Cue

The cue is the shape of the prompt, not a keyword to memorize. A contiguous range suggests a window or prefix sum. A sorted list suggests two pointers or binary search. A dependency list suggests topological order.

### State

Ask what a future step needs from the past. A hash map may keep an index, a stack may keep unfinished openings, and dynamic programming may keep one answer per prefix. If the state is larger than the future needs, the solution is probably carrying noise.

### Invariant

An invariant is the rule that stays true while the algorithm runs. Examples:

- the window contains exactly the tracked characters;
- the queue is ordered by nondecreasing distance;
- every finalized shortest-path distance is final;
- each DP cell is the complete answer for its smaller problem;
- the path contains exactly the choices made on this branch.

Say the invariant aloud before trusting the code.

### Move

Every pointer movement, pop, merge, and recursive call needs a reason. Move a left boundary because the current window has extra information. Pop a heap item because no cheaper frontier item exists. Remove a stack entry because the new value resolves it. Return from a subtree because its promised fact is complete.

## A three-pass practice loop

### First pass: learn one anchor per pattern

Start with these 20 anchor problems, in order:

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

For each anchor, look at the visual first. Cover the code. Explain the trace, state, invariant, and move. Then write the implementation.

### Second pass: study nearby variations

Solve related problems together. Notice what changes and what stays fixed:

- Two Sum becomes 3Sum after sorting and fixing one value.
- BFS becomes Dijkstra when edges have different nonnegative costs.
- DFS becomes dynamic programming when repeated states are cached.
- Implement Trie becomes wildcard search when a dot can follow any child.
- Reverse Linked List becomes one step inside Reorder List.

Variation is the antidote to memorizing one example.

### Mixed pass: choose the pattern yourself

Hide the pattern heading and ask:

1. What is the brute-force method?
2. What work does it repeat?
3. Which state removes that repeated work?
4. What rule stays true?
5. Why is the next move safe?
6. What are the time and space costs?

After 15 focused minutes, read only the visual and simple idea. Try again. Read the code only when needed, close it, and rebuild it from the invariant.

## Review schedule

Rebuild missed problems after 1, 3, 7, and 14 days. Record the reason for each miss:

- I did not recognize the pattern.
- I chose the pattern but kept the wrong state.
- I could not explain why a move was safe.
- I knew the idea but could not write the code.
- I missed an edge case.
- I gave the wrong complexity.

The review card is meant to be a retrieval cue. The full page is for repairing the explanation. Keep those jobs separate.

## Quick pattern map

| Prompt shape | State to draw first | Likely family |
| --- | --- | --- |
| Pair, duplicate, fast lookup | seen values or counts | Hash map or set |
| Contiguous range | left/right window or prefix totals | Sliding window or prefix sum |
| Sorted values | two ends or low/high answer bounds | Two pointers or binary search |
| Nested input | newest unfinished item | Stack |
| Next greater item | waiting indices in order | Monotonic stack |
| Fewest unweighted steps | distance layers | BFS |
| Lowest weighted path | cheapest frontier | Dijkstra |
| All reachable groups | visited frontier | DFS or graph search |
| Every valid choice | partial path and choices left | Backtracking |
| Repeated smaller question | complete saved state | Dynamic programming |
| Time ranges | active end boundary | Intervals or greedy |
| Prerequisites | incoming-edge counts | Topological sort |
| Edges joining groups | representative roots | Union-find |
| Changed node order | saved next pointer | Linked list |
| Shared word beginnings | prefix path | Trie |

The companion book starts with [Learn by rebuilding](/library/coding-interview/method/) and then keeps every problem in the order used by the supplied phone guide. Each problem has a visual state map before its implementation, so the first question is always: what should I be able to see?
