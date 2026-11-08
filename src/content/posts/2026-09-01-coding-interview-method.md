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
<figure class="learning-figure editorial-illustration" aria-labelledby="coding-interview-method-state-edition-title"><span class="visual-fragment-anchor" id="coding-interview-method-state-title" aria-hidden="true"></span><p class="visual-title" id="coding-interview-method-state-edition-title">The complement is a value; the saved map entry recovers its earlier index</p><svg viewBox="0 0 360 618" role="img" aria-labelledby="coding-interview-method-state-edition-svg-title coding-interview-method-state-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="coding-interview-method-state-edition-svg-title">The complement is a value; the saved map entry recovers its earlier index</title><desc id="coding-interview-method-state-edition-svg-desc">For toy input four, two, seven and target nine, current index two contains seven. Nine minus seven is two. The earlier-value map contains four to index zero and two to index one. Looking up two returns one, so indices one and two identify two plus seven. The current index is absent during the query; insertion happens only after a miss.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Toy input [4, 2, 7]; target 9</tspan></text><rect x="72" y="53" width="72" height="46" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="108" y="83" style="font-size:17px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="108" dy="0">4</tspan></text><text x="108" y="121" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="108" dy="0">index 0</tspan></text><rect x="144" y="53" width="72" height="46" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="83" style="font-size:17px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">2</tspan></text><text x="180" y="121" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">index 1</tspan></text><rect x="216" y="53" width="72" height="46" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="252" y="83" style="font-size:17px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="252" dy="0">7</tspan></text><text x="252" y="121" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="252" dy="0">index 2</tspan></text><g data-editorial-node><rect x="20" y="154" width="320" height="65" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="180" y="182" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Current index 2: value 7</tspan><tspan x="180" dy="19">complement = target 9 - 7 = 2</tspan></text></g><path d="M180 219L180 254" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,247.8569220667674L180,254L183.35597877022943,247.8569220667674" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><text x="180" y="276" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">seen contains ONLY earlier positions</tspan></text><text x="110" y="309" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="110" dy="0">value</tspan></text><text x="252" y="309" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="252" dy="0">earlier index</tspan></text><path d="M70 319L310 319" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="110" y="345" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="110" dy="0">4</tspan></text><text x="252" y="345" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="252" dy="0">0</tspan></text><rect x="70" y="357" width="240" height="36" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="110" y="381" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="110" dy="0">2</tspan></text><text x="252" y="381" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="252" dy="0">1</tspan></text><text x="180" y="425" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Lookup 2 returns earlier index 1.</tspan></text><g data-editorial-node><rect x="20" y="450" width="320" height="60" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="180" y="475.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Return [1, 2]: 2 + 7 = 9</tspan><tspan x="180" dy="19">Two different input positions</tspan></text></g><text x="180" y="549" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Index 2 was absent during the query.</tspan></text><text x="180" y="579" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">On a miss, store the current value only after</tspan><tspan x="180" dy="19">querying.</tspan></text></svg><figcaption><strong>Read it this way:</strong> Follow value 2 from the complement calculation to the saved index 1. Returning [1,2] uses two different positions. Query-before-store is the invariant, not a variable spelling. <span class="visual-reader-check" data-reader-check="coding-interview-method-state"><strong>Check:</strong> How does the matching value recover an earlier index without reusing the current position?</span></figcaption></figure>

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

These 35 anchors form one suggested collection, not a universal optimum. They are grouped by the state they keep and the moves that state makes safe. The problem numbers identify anchors; they do not set the practice order.

Learn one anchor in each group first. Before reading its code, name the cue, state, invariant and next safe move. Then compare nearby problems: what changed, and does the old rule still hold? The snippets show compact Python implementations; linked-list and tree node types come from the problem.

#### Group 1: Hash Maps, Prefix Sums & Weighted Sampling

**Mental model:** Use expected $O(1)$ map lookups for missing complements or earlier running sums. For weighted sampling, search cumulative bucket boundaries.

<details>
<summary>1. Two Sum</summary>

- Each value tells you the partner it needs: `target - n`.
- Save earlier values and their indices. Look up the partner before saving the current index, so one position cannot pair with itself.
- `[3, 3]`, target 6, returns `[0, 1]`. Expected time is $O(n)$; space is $O(n)$.

```python
def two_sum(nums, target):
	seen = {}
	for i, n in enumerate(nums):
		if target - n in seen:
			return [seen[target - n], i]
		seen[n] = i
	return []
```

</details>

<details>
<summary>2. Subarray Sum Equals K</summary>

- A range sum is the current prefix total minus an earlier prefix total. Find earlier totals equal to `curr_sum - target`.
- Store frequencies, not just presence. Start with one zero prefix, and count matches before recording the current prefix.
- Negative values can break a shrinking-window rule; prefix differences still work. Expected time is $O(n)$; space is $O(n)$.

```python
def subarray_sum(nums, target):
	seen_sums = {0: 1}
	curr_sum = count = 0
	for n in nums:
		curr_sum += n
		count += seen_sums.get(curr_sum - target, 0)
		seen_sums[curr_sum] = seen_sums.get(curr_sum, 0) + 1
	return count
```

</details>

<details>
<summary>34. Random Pick with Weight</summary>

- Integer weights give each index that many tickets. With `[1, 3]`, index 0 owns ticket 1 and index 1 owns tickets 2 through 4.
- Build cumulative totals once. Draw an integer from 1 through the total, then find the first boundary at least as large as that ticket.
- Weights are nonnegative integers with a positive total. Setup takes $O(n)$ time and space; each draw takes $O(\log n)$ time.

```python
from bisect import bisect_left
from itertools import accumulate
from random import randint

prefix_sums = list(accumulate(weights))

def pick_index():
	ticket = randint(1, prefix_sums[-1])
	return bisect_left(prefix_sums, ticket)
```

</details>

#### Group 2: Two Pointers & Sliding Windows

**Mental model:** Move converging boundaries, or expand a window and move its left boundary only when the window's rule breaks.

<details>
<summary>4. Longest Substring Without Repeating Characters</summary>

- Keep a window with no repeated character. A repeat tells you exactly which earlier position must leave.
- Jump `left` past the previous occurrence, but never move it backward. An occurrence outside the current window needs no repair.
- `"abba"` has length 2, not 3: the last `a` must not move `left` backward. Time is $O(n)$; space depends on the distinct characters.

```python
def longest_substring(s):
	left = max_len = 0
	char_map = {}
	for right, c in enumerate(s):
		if c in char_map and char_map[c] >= left:
			left = char_map[c] + 1
		char_map[c] = right
		max_len = max(max_len, right - left + 1)
	return max_len
```

</details>

<details>
<summary>3. 3Sum</summary>

- Sort the values and fix one number. The remaining task is to find a pair whose sum cancels it.
- A total below zero needs a larger left value; a total above zero needs a smaller right value. Skip duplicate choices to avoid repeated triples.
- This sorts the input in place. Time is $O(n^2)$; Python's sort uses $O(n)$ auxiliary space, excluding the returned triples.

```python
def three_sum(nums):
	nums.sort()
	res = []
	for i, a in enumerate(nums):
		if i > 0 and a == nums[i - 1]:
			continue
		l, r = i + 1, len(nums) - 1
		while l < r:
			total = a + nums[l] + nums[r]
			if total == 0:
				res.append([a, nums[l], nums[r]])
				l, r = l + 1, r - 1
				while l < r and nums[l] == nums[l - 1]:
					l += 1
			elif total < 0:
				l += 1
			else:
				r -= 1
	return res
```

</details>

<details>
<summary>30. Valid Palindrome II</summary>

- Compare the two ends until they disagree. Before that mismatch, no deletion is needed.
- At the first mismatch, the only useful choices are to skip the left character or the right character. The remaining range must be a palindrome with no more deletions.
- `"abca"` succeeds by deleting `b` or `c`. Time is $O(n)$; the slices shown here use $O(n)$ space.

```python
def valid_palindrome_ii(s):
	l, r = 0, len(s) - 1
	while l < r:
		if s[l] != s[r]:
			skip_left = s[l + 1:r + 1]
			skip_right = s[l:r]
			return (skip_left == skip_left[::-1]
					or skip_right == skip_right[::-1])
		l, r = l + 1, r - 1
	return True
```

</details>

#### Group 3: Binary Search & Search Space

**Mental model:** Halve an ordered search space. That space can contain array positions or candidate answers with a monotone feasibility rule.

<details>
<summary>5. Binary Search</summary>

- In a sorted array, the middle value tells you which half cannot contain the target.
- Keep the target's possible positions inside `[l, r]`. Remove `mid` with `mid + 1` or `mid - 1` after a miss.
- An empty range means no match. Time is $O(\log n)$; space is $O(1)$.

```python
def binary_search(arr, target):
	l, r = 0, len(arr) - 1
	while l <= r:
		mid = (l + r) // 2
		if arr[mid] == target:
			return mid
		elif arr[mid] < target:
			l = mid + 1
		else:
			r = mid - 1
	return -1
```

</details>

<details>
<summary>31. Koko Eating Bananas</summary>

- Search eating speeds, not pile positions. If a speed finishes within `h` hours, every faster speed also works.
- A feasible middle speed may not be the slowest one, so keep it as an answer and search lower. Use integer ceiling division for each pile.
- Piles are positive and `h` is at least the number of piles. Time is $O(n \log M)$ for largest pile $M$; space is $O(1)$.

```python
def min_eating_speed(piles, h):
	l, r = 1, max(piles)
	ans = r
	while l <= r:
		mid = (l + r) // 2
		hours = sum((p + mid - 1) // mid for p in piles)
		if hours <= h:
			ans, r = mid, mid - 1
		else:
			l = mid + 1
	return ans
```

</details>

#### Group 4: Monotonic Stacks & String Parsing

**Mental model:** A stack holds unfinished work: unmatched symbols, indices awaiting a larger value, or terms awaiting an operator.

<details>
<summary>6. Valid Parentheses</summary>

- An opening bracket waits for a matching closing bracket. Nested pairs must close in reverse order.
- Each closing bracket must match the stack's top. An empty stack or wrong opener fails immediately.
- An unmatched opener also fails at the end: return `not stack`. Time and worst-case space are $O(n)$.

```python
def valid_parentheses(s):
	stack, mapping = [], {")": "(", "}": "{", "]": "["}
	for c in s:
		if c in mapping:
			if not stack or stack.pop() != mapping[c]:
				return False
		else:
			stack.append(c)
	return not stack
```

</details>

<details>
<summary>7. Daily Temperatures</summary>

- Each day waits for its first warmer day. Store waiting indices whose temperatures are nonincreasing.
- A warmer temperature resolves every colder index at the top. Pop the waiting index first, then subtract it from today's index.
- Each index enters and leaves the stack at most once. Time and space are $O(n)$; unresolved days keep answer 0.

```python
def daily_temperatures(temperatures):
	ans, stack = [0] * len(temperatures), []
	for i, t in enumerate(temperatures):
		while stack and t > temperatures[stack[-1]]:
			waiting = stack.pop()
			ans[waiting] = i - waiting
		stack.append(i)
	return ans
```

</details>

<details>
<summary>33. Basic Calculator II</summary>

- Read a valid expression of nonnegative integers and `+`, `-`, `*`, `/`, with no parentheses. Build a number until the next operator arrives.
- Save addition and subtraction as signed terms. Apply multiplication and division to the last term immediately, before the final sum.
- Division truncates toward zero: `-3 / 2` contributes `-1`, not `-2`. Integer arithmetic avoids float rounding; time and space are $O(n)$.

```python
def calculate(s):
	stack, num, sign = [], 0, "+"
	for c in s + "+":
		if c.isdigit():
			num = num * 10 + int(c)
		elif c in "+-*/":
			if sign == "+":
				stack.append(num)
			elif sign == "-":
				stack.append(-num)
			elif sign == "*":
				stack.append(stack.pop() * num)
			elif sign == "/":
				previous = stack.pop()
				quotient = abs(previous) // num
				stack.append(quotient if previous >= 0 else -quotient)
			sign, num = c, 0
	return sum(stack)
```

</details>

#### Group 5: Intervals, Heaps & Greedy

**Mental model:** Sort boundaries or keep useful extremes in a heap. A reachable bound can summarize earlier greedy choices.

<details>
<summary>15. Merge Intervals</summary>

- Sort by start time. Then an interval can overlap only the last merged range, not an earlier finished range.
- If its start is at most the last end, extend that end. Otherwise, start a new range.
- Touching endpoints merge. This sorts and updates the input intervals; time is $O(n \log n)$ and space is $O(n)$.

```python
def merge_intervals(intervals):
	if not intervals:
		return []
	intervals.sort(key=lambda interval: interval[0])
	merged = [intervals[0]]
	for curr in intervals[1:]:
		if curr[0] <= merged[-1][1]:
			merged[-1][1] = max(merged[-1][1], curr[1])
		else:
			merged.append(curr)
	return merged
```

</details>

<details>
<summary>27. Meeting Rooms II</summary>

- Process meetings by start time. A min-heap keeps one end time per allocated room.
- Reuse the earliest-ending room when its end is at most the new start. Otherwise, allocate another room; each new meeting can reuse only one.
- The heap's final size is the required room count, including rooms reused earlier. Time is $O(n \log n)$; space is $O(n)$.

```python
import heapq

def meeting_rooms(intervals):
	intervals.sort(key=lambda interval: interval[0])
	min_heap = []
	for start, end in intervals:
		if min_heap and min_heap[0] <= start:
			heapq.heappop(min_heap)
		heapq.heappush(min_heap, end)
	return len(min_heap)
```

</details>

<details>
<summary>9. Kth Largest Element</summary>

- Keep only the largest `k` values seen so far. Their smallest value is the current cutoff.
- Push each new value into a min-heap, then remove the smallest whenever the heap grows beyond `k`.
- Duplicates count as separate elements, and `1 <= k <= len(nums)`. Time is $O(n \log k)$; space is $O(k)$.

```python
import heapq

def kth_largest(nums, k):
	min_heap = []
	for n in nums:
		heapq.heappush(min_heap, n)
		if len(min_heap) > k:
			heapq.heappop(min_heap)
	return min_heap[0]
```

</details>

<details>
<summary>16. Jump Game</summary>

- Each nonnegative value sets the farthest jump from its position. Track the farthest index reachable from any earlier reachable position.
- If the current index lies beyond that bound, no earlier jump can reach it. Otherwise, extend the bound with `i + n`.
- You need a reachable range, not one chosen jump path. Time is $O(n)$; space is $O(1)$.

```python
def can_jump(nums):
	furthest = 0
	for i, n in enumerate(nums):
		if i > furthest:
			return False
		furthest = max(furthest, i + n)
	return True
```

</details>

#### Group 6: Linked Lists & Caching

**Mental model:** Use dummy anchors to simplify endpoints. Preserve the next pointer before rewiring. A map plus a doubly linked list makes lookup and recency updates fast.

<details>
<summary>19. Reverse Linked List</summary>

- Split the list into a reversed prefix and an untouched suffix. `prev` heads the prefix; `curr` heads the suffix.
- Save `curr.next` before reversing the link. Without that saved pointer, the remaining nodes become unreachable.
- When the suffix is empty, `prev` is the new head. Time is $O(n)$; extra space is $O(1)$.

```python
def reverse_list(head):
	prev, curr = None, head
	while curr:
		nxt = curr.next
		curr.next = prev
		prev, curr = curr, nxt
	return prev
```

</details>

<details>
<summary>21. Merge Two Sorted Lists</summary>

- The smaller current head is the next output node, because neither sorted list hides a smaller value farther ahead.
- A dummy head gives the first node the same attachment rule as every later node. Advance only the list that supplied the node.
- Attach the remaining suffix when one list ends. This reuses nodes; time is $O(n + m)$ and extra space is $O(1)$.

```python
def merge_two_lists(l1, l2):
	dummy = curr = ListNode(0)
	while l1 and l2:
		if l1.val < l2.val:
			curr.next, l1 = l1, l1.next
		else:
			curr.next, l2 = l2, l2.next
		curr = curr.next
	curr.next = l1 if l1 else l2
	return dummy.next
```

</details>

<details>
<summary>28. LRU Cache</summary>

- Use a positive capacity. Keep cached values, and evict the least recently used key when adding a key exceeds capacity.
- A map finds each node; a doubly linked list orders use from oldest to newest. Every read or write moves its node just before the tail.
- Here, `Node(key=None, value=None)` supplies `key`, `value`, `prev` and `next`. Dummy head and tail nodes make each `get` and `put` $O(1)$; space is $O(\text{capacity})$.

```python
class LRUCache:
	def __init__(self, capacity):
		self.capacity = capacity
		self.cache = {}
		self.head, self.tail = Node(), Node()
		self.head.next, self.tail.prev = self.tail, self.head

	def _remove(self, node):
		node.prev.next, node.next.prev = node.next, node.prev

	def _add_to_end(self, node):
		previous = self.tail.prev
		previous.next, node.prev = node, previous
		node.next, self.tail.prev = self.tail, node

	def get(self, key):
		if key not in self.cache:
			return -1
		node = self.cache[key]
		self._remove(node)
		self._add_to_end(node)
		return node.value

	def put(self, key, value):
		if key in self.cache:
			node = self.cache[key]
			node.value = value
			self._remove(node)
		else:
			node = self.cache[key] = Node(key, value)
		self._add_to_end(node)
		if len(self.cache) > self.capacity:
			oldest = self.head.next
			self._remove(oldest)
			del self.cache[oldest.key]
```

</details>

#### Group 7: Trees, BSTs & Tries

**Mental model:** Use recursion for subtrees and queue snapshots for levels. Track inherited bounds in BSTs and shared prefix paths in tries.

<details>
<summary>12. Maximum Depth of Binary Tree</summary>

- An empty subtree has depth 0. A nonempty subtree adds its root to the deeper child path.
- Ask each child for the same quantity, then combine the answers with `1 + max(...)`.
- Visit each node once. Time is $O(n)$; the recursion stack uses $O(h)$ space for tree height $h$.

```python
def max_depth(root):
	return (1 + max(max_depth(root.left), max_depth(root.right))
			if root else 0)
```

</details>

<details>
<summary>22. Binary Tree Level Order Traversal</summary>

- A queue visits parents before their children. At the start of each outer loop, it contains exactly one level.
- Save that level's values and process only its initial queue length. Newly added children belong to the next level.
- Time is $O(n)$; the queue uses $O(w)$ space for maximum width $w$, excluding the returned levels.

```python
from collections import deque

def level_order(root):
	queue, res = deque([root] if root else []), []
	while queue:
		res.append([node.val for node in queue])
		for _ in range(len(queue)):
			node = queue.popleft()
			if node.left:
				queue.append(node.left)
			if node.right:
				queue.append(node.right)
	return res
```

</details>

<details>
<summary>23. Validate Binary Search Tree</summary>

- A BST node must satisfy every ancestor's bounds, not just compare correctly with its parent.
- Going left tightens the upper bound; going right tightens the lower bound. Use strict bounds to reject duplicate values.
- A right-subtree value below the root is invalid even if its parent comparison passes. Time is $O(n)$; stack space is $O(h)$.

```python
def validate(node, low=-float("inf"), high=float("inf")):
	return (not node
			or low < node.val < high
			and validate(node.left, low, node.val)
			and validate(node.right, node.val, high))
```

</details>

<details>
<summary>20. Implement Trie</summary>

- Lowercase words with the same prefix share the same path of character nodes.
- Mark the end of a complete word separately. After inserting `"apple"`, `"app"` is a prefix but not yet a stored word.
- Each operation follows at most the word's or prefix's length $L$: expected time is $O(L)$; storage follows the inserted characters.

```python
class Trie:
	def __init__(self):
		self.root = {}

	def insert(self, word):
		node = self.root
		for char in word:
			node = node.setdefault(char, {})
		node["$"] = True

	def _find(self, text):
		node = self.root
		for char in text:
			if char not in node:
				return None
			node = node[char]
		return node

	def search(self, word):
		node = self._find(word)
		return node is not None and "$" in node

	def starts_with(self, prefix):
		return self._find(prefix) is not None
```

</details>

#### Group 8: Graphs, Matrices, Shortest Paths & Coloring

**Mental model:** Follow grid neighbors, remove zero-in-degree nodes, or join components. Expand cheapest weighted frontiers, transform coordinates, or color edge endpoints oppositely.

<details>
<summary>11. Number of Islands</summary>

- One island is one connected component of land cells using the four grid neighbors, not diagonals.
- When you find unvisited land, count one island and flood-fill its entire component. Mark a cell before visiting its neighbors to stop revisits.
- This changes land from `"1"` to `"0"`. Time is $O(RC)$; the DFS stack can use $O(RC)$ space.

```python
def num_islands(grid):
	rows, cols = len(grid), len(grid[0]) if grid else 0

	def dfs(r, c):
		if not (0 <= r < rows and 0 <= c < cols
				and grid[r][c] == "1"):
			return
		grid[r][c] = "0"
		for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
			dfs(r + dr, c + dc)

	count = 0
	for r in range(rows):
		for c in range(cols):
			if grid[r][c] == "1":
				count += 1
				dfs(r, c)
	return count
```

</details>

<details>
<summary>8. Rotting Oranges</summary>

- Every initially rotten orange starts spreading at minute 0. Put all of them in the same BFS queue.
- One queue layer is one minute. Mark fresh neighbors rotten when you enqueue them, so two sources cannot count the same orange twice.
- Fresh oranges left after the queue empties are unreachable. This mutates the grid; time and space are $O(RC)$.

```python
from collections import deque

def oranges_rotting(grid):
	rows, cols = len(grid), len(grid[0]) if grid else 0
	queue = deque((row, col) for row in range(rows)
				  for col in range(cols) if grid[row][col] == 2)
	fresh = sum(cell == 1 for row in grid for cell in row)
	minutes = 0
	while queue and fresh:
		for _ in range(len(queue)):
			row, col = queue.popleft()
			for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
				nr, nc = row + dr, col + dc
				if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
					grid[nr][nc] = 2
					fresh -= 1
					queue.append((nr, nc))
		minutes += 1
	return minutes if fresh == 0 else -1
```

</details>

<details>
<summary>17. Course Schedule</summary>

- Direct each edge from a prerequisite to its course. A course with zero incoming edges has no unfinished prerequisite.
- Remove such a course, decrease its neighbors' in-degrees, and enqueue each neighbor when its count reaches zero.
- If you cannot remove all courses, a cycle remains. Kahn's topological sort takes $O(V + E)$ time and space.

```python
from collections import defaultdict, deque

def can_finish(num_courses, prerequisites):
	in_degree, adj = [0] * num_courses, defaultdict(list)
	for course, prerequisite in prerequisites:
		adj[prerequisite].append(course)
		in_degree[course] += 1
	queue = deque(node for node in range(num_courses) if in_degree[node] == 0)
	completed = 0
	while queue:
		node = queue.popleft()
		completed += 1
		for neighbor in adj[node]:
			in_degree[neighbor] -= 1
			if in_degree[neighbor] == 0:
				queue.append(neighbor)
	return completed == num_courses
```

</details>

<details>
<summary>18. Redundant Connection</summary>

- In a tree with one added undirected edge, that extra edge joins nodes already connected by another path.
- Union-find stores a representative for each component. Equal representatives reveal the cycle; otherwise, join the smaller component to the larger one.
- With labels 1 through `len(edges)`, path compression and union by size give near-linear time, $O(n\alpha(n))$, and $O(n)$ space.

```python
def redundant_connection(edges):
	parent = list(range(len(edges) + 1))
	size = [1] * len(parent)

	def find(node):
		while node != parent[node]:
			parent[node] = parent[parent[node]]
			node = parent[node]
		return node

	for start, end in edges:
		root_start, root_end = find(start), find(end)
		if root_start == root_end:
			return [start, end]
		if size[root_start] < size[root_end]:
			root_start, root_end = root_end, root_start
		parent[root_end] = root_start
		size[root_start] += size[root_end]
	return []
```

</details>

<details>
<summary>10. Network Delay Time</summary>

- With nonnegative edge weights, Dijkstra's algorithm expands the smallest known arrival time first. A FIFO queue would minimize edge count, not travel time.
- Relax each outgoing edge when it improves a distance. Skip heap entries whose arrival times have already been improved.
- Return the largest shortest-path distance, or `-1` if any node is unreachable. Time is $O((V + E)\log V)$; space is $O(V + E)$.

```python
import heapq
from collections import defaultdict

def network_delay_time(times, node_count, source):
	adj = defaultdict(list)
	for start, end, weight in times:
		adj[start].append((end, weight))
	distance = [float("inf")] * (node_count + 1)
	distance[source] = 0
	heap = [(0, source)]
	while heap:
		elapsed, node = heapq.heappop(heap)
		if elapsed != distance[node]:
			continue
		for neighbor, weight in adj[node]:
			arrival = elapsed + weight
			if arrival < distance[neighbor]:
				distance[neighbor] = arrival
				heapq.heappush(heap, (arrival, neighbor))
	answer = max(distance[1:])
	return -1 if answer == float("inf") else answer
```

</details>

<details>
<summary>29. Rotate Image</summary>

- Rotate a square matrix 90 degrees clockwise: position `(row, col)` moves to `(col, n - 1 - row)`.
- Transpose across the main diagonal, then reverse each row. Those two moves produce the required coordinate change.
- Swap only one side of the diagonal, or each pair would be swapped twice. This is in place: time is $O(n^2)$; extra space is $O(1)$.

```python
def rotate(matrix):
	size = len(matrix)
	for row in range(size):
		for col in range(row + 1, size):
			matrix[row][col], matrix[col][row] = matrix[col][row], matrix[row][col]
	for row in matrix:
		row.reverse()
```

</details>

<details>
<summary>32. Is Graph Bipartite?</summary>

- Every edge must connect nodes with opposite colors. Once a node has a color, a later visit must agree with it.
- Color an unvisited node, then require the opposite color at every neighbor. Propagate a failed recursive check instead of accepting a node just because it was colored.
- Start a check in every disconnected component. An odd cycle fails; time and space are $O(V + E)$ and $O(V)$ respectively.

```python
def is_bipartite(graph):
	color = {}

	def dfs(node, current_color):
		if node in color:
			return color[node] == current_color
		color[node] = current_color
		return all(dfs(neighbor, 1 - current_color) for neighbor in graph[node])

	return all(dfs(node, 0) for node in range(len(graph)) if node not in color)
```

</details>

#### Group 9: Dynamic Programming & String Alignment

**Mental model:** Store answers for smaller states. A transition must use enough state to make the remaining decision independent of how you reached it.

<details>
<summary>14. House Robber</summary>

- At each house, either skip it or take its money. Taking it also needs the best total two houses back.
- Only those two earlier totals matter, so keep two rolling values instead of the full DP array.
- Update both from their old values at once. Empty and one-house inputs work without special indexing; time is $O(n)$ and space is $O(1)$.

```python
def rob(nums):
	prev = curr = 0
	for amount in nums:
		prev, curr = curr, max(curr, prev + amount)
	return curr
```

</details>

<details>
<summary>24. Coin Change</summary>

- The remaining amount fully identifies a subproblem. Try each positive coin that fits, then ask for the fewest coins for the remainder.
- Cache each remainder's answer. Keep an unreachable remainder at infinity, including when no coin fits; convert the final infinity to `-1`.
- Time is $O(AD)$ for amount $A$ and $D$ coin types; space is $O(A)$. Deep recursive paths need an iterative DP in Python.

```python
from functools import lru_cache

def coin_change(coins, amount):
	@lru_cache(None)
	def dfs(rem):
		if rem == 0:
			return 0
		return min((dfs(rem - coin) + 1 for coin in coins if coin <= rem),
				   default=float("inf"))

	answer = dfs(amount)
	return -1 if answer == float("inf") else answer
```

</details>

<details>
<summary>25. Word Break</summary>

- Ask which string prefixes can be formed from dictionary words. Start with the empty prefix reachable.
- From a reachable boundary, match a whole dictionary word and mark its ending boundary. A locally longest word is not always the right choice.
- `"cars"` with `["car", "ca", "rs"]` succeeds through `"ca" + "rs"`. Time is $O(nDL)$ for $D$ words of maximum length $L$; DP space is $O(n)$.

```python
def word_break(s, words):
	reachable = [True] + [False] * len(s)
	for start in range(len(s)):
		if not reachable[start]:
			continue
		for word in words:
			if s.startswith(word, start):
				reachable[start + len(word)] = True
	return reachable[-1]
```

</details>

<details>
<summary>35. Edit Distance</summary>

- `dp[i][j]` aligns the first `i` characters of one word with the first `j` of the other. Store the fewest edits for that alignment.
- Matching characters need no new edit. Otherwise, add one to the best deletion, insertion or replacement state.
- Converting to or from an empty prefix costs its length, which sets the table's first column and row. Time and space are $O(mn)$.

```python
def edit_distance(w1, w2):
	dp = [[0] * (len(w2) + 1) for _ in range(len(w1) + 1)]
	for i in range(len(w1) + 1):
		dp[i][0] = i
	for j in range(len(w2) + 1):
		dp[0][j] = j
	for i in range(1, len(w1) + 1):
		for j in range(1, len(w2) + 1):
			if w1[i - 1] == w2[j - 1]:
				dp[i][j] = dp[i - 1][j - 1]
			else:
				dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
	return dp[-1][-1]
```

</details>

#### Group 10: Backtracking & Bit Manipulation

**Mental model:** Enumerate choices with choose-explore-unchoose, or use XOR when duplicate values must cancel rather than be stored.

<details>
<summary>13. Subsets</summary>

- Every current path is a valid subset, including the empty path. Copy it into the result before adding more choices.
- Choose a value, explore only later indices, then undo the choice. Increasing indices avoid generating different orders of the same subset.
- Copying results takes $O(n2^n)$ time and output space; the working path and recursion stack use $O(n)$ space.

```python
def subsets(nums):
	res = []

	def backtrack(start, path):
		res.append(list(path))
		for i in range(start, len(nums)):
			path.append(nums[i])
			backtrack(i + 1, path)
			path.pop()

	backtrack(0, [])
	return res
```

</details>

<details>
<summary>26. Single Number</summary>

- This rule needs every value to appear twice except one. XOR makes two equal values cancel: `x ^ x == 0`.
- XOR all values together. Order does not matter, and `0 ^ x == x`, so the unpaired value remains.
- This is not a counting solution for arbitrary frequencies. Time is $O(n)$; extra space is $O(1)$ under fixed-width integer assumptions.

```python
def single_number(nums):
	ans = 0
	for n in nums:
		ans ^= n
	return ans
```

</details>

Read the problem first; hide the code, draw the state, explain the invariant, then implement and test. Recognizing a group is the start of the explanation, not proof that its usual move is safe.

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
