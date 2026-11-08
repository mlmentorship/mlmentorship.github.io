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

Start with one problem. Work out what each step needs, then choose what to store.

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

A missing partner suggests a lookup instead of scanning the array again.

### State

Save one earlier index per value in `seen`. To count pairs, store frequencies instead.

### Invariant

An invariant is a rule that stays true. Here, `seen` contains only values from earlier positions.

### Move

Look up the partner first, then save the current value. This prevents using one position twice.

## Code

The [Two Sum solution](/questions/two-sum/):

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

Expected time is $O(n)$, with $O(n)$ extra space.
Check `[3]`, target 6: no earlier partner, so `[]`. Check `[3, 3]`: the second query returns `[0, 1]`.

## Change 1: Count Every Index Pair

Count pairs of distinct indices. Store prior counts, not one index. Count matches before adding the current value.

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

## Change 2: Count Signed Zero-Sum Ranges

Count contiguous zero-sum ranges in `[1, -1, 1]`.
A prefix sum is the total up to a boundary. Equal prefix sums mean the values between them sum to zero.
Count matching prefixes, including the initial zero.

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

**Check.** `[1, -1, 1]` has two zero-sum ranges: `[1, -1]` and `[-1, 1]`. Empty input gives 0.

## Failure: Remembering Only "Hash Map"

Knowing "hash map" is not enough. Explain what the map stores and why lookup comes before insertion.

## Transfer Check

Why does a contiguous range not always imply a sliding window?

### Reasoned Answer

Adding a negative value can lower the sum. A window that shrinks whenever the sum is too large may discard a valid answer.
Prefix sums avoid that assumption.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## A three-pass practice loop

### First pass: learn one anchor per pattern

Start with the numbered problems. Use their related problems to practice a similar idea or compare a different method.

The snippets show the main steps, not complete solutions. Work out the missing setup before using them.

#### Group 1: Hash Maps, Prefix Sums & Weighted Sampling

**Idea:** Save earlier values or running totals. Weighted sampling uses running totals as bucket boundaries.

<details>
<summary>1. Subarray Sum Equals K</summary>

- A range sum is the difference between two prefix totals.
- Count earlier totals equal to `curr_sum - target`; frequencies preserve multiple matching boundaries.
- Seed one zero prefix. Count before storing, so only earlier boundaries contribute.

```python
seen_sums = {0: 1}
curr_sum = count = 0
for n in nums:
    curr_sum += n
    count += seen_sums.get(curr_sum - target, 0)
    seen_sums[curr_sum] = seen_sums.get(curr_sum, 0) + 1
```

<details style="margin-inline-start:1.25rem">
<summary>Two Sum</summary>

- **Change:** Return one pair, not a range count. Keep one earlier index per value.

  ```python
  seen = {}  # value -> earlier index
  ```

- **Move:** Query before storing. `[3, 3]`, target 6, returns `[0, 1]` without reusing an index.

  ```python
  for i, n in enumerate(nums):
      if target - n in seen:
          return [seen[target - n], i]
      seen[n] = i
  ```

</details>

<details style="margin-inline-start:1.25rem">
<summary>Random Pick with Weight</summary>

- **Contrast:** Running totals mark weighted buckets instead of counting earlier matches.

  ```python
  prefix_sums = list(accumulate(weights))
  ```

- **Move:** Draw an integer ticket and find its first enclosing boundary. `[1, 3]` gives index 1 three tickets.

  ```python
  ticket = random.randint(1, prefix_sums[-1])
  return bisect_left(prefix_sums, ticket)
  ```

</details>

<details style="margin-inline-start:1.25rem">
<summary>Product of Array Except Self</summary>

- **Contrast:** Combine contributions on either side of each index, without division. First store the product strictly to its left.

  ```python
  res, prefix = [1] * len(nums), 1
  for i, n in enumerate(nums):
      res[i] = prefix
      prefix *= n
  ```

- **Move:** Multiply by the product strictly to the right. `[1, 2, 3, 4]` produces `[24, 12, 8, 6]`.

  ```python
  suffix = 1
  for i in range(len(nums) - 1, -1, -1):
      res[i] *= suffix
      suffix *= nums[i]
  ```

</details>
</details>

#### Group 2: Two Pointers & Sliding Windows

**Idea:** Move two pointers toward each other, or expand and shrink a window while keeping its rule true.

<details>
<summary>2. Minimum Window Substring</summary>

- Cover every required character, including repeats; assume `t` is nonempty.
- Expand until the window covers `t`, then shrink while coverage holds.
- Removing a required occurrence creates a deficit. Keep the shortest valid window seen.

```python
need, missing = Counter(t), len(t)
left = 0
for right, c in enumerate(s):
    missing -= need[c] > 0
    need[c] -= 1
    while missing == 0:
        # Save the shortest window [left, right].
        need[s[left]] += 1
        missing += need[s[left]] > 0
        left += 1
```

<details style="margin-inline-start:1.25rem">
<summary>Longest Substring Without Repeating Characters</summary>

- **Change:** Keep characters unique instead of covering required characters. Store each character's most recent position.

  ```python
  left = max_len = 0
  char_map = {}
  ```

- **Move:** Jump past repeats inside the current window, never backward. `"abba"` has maximum length 2.

  ```python
  for right, c in enumerate(s):
      if c in char_map and char_map[c] >= left:
          left = char_map[c] + 1
      char_map[c] = right
      max_len = max(max_len, right - left + 1)
  ```

</details>
</details>

<details>
<summary>3. 3Sum</summary>

- Sort, then fix one value and seek a pair that cancels it.
- A low total needs a larger left value; a high total needs a smaller right value.
- Skip repeated anchor and left values to avoid duplicate triples.

```python
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
            l += 1
            while l < r and nums[l] == nums[l - 1]:
                l += 1
        elif total < 0:
            l += 1
        else:
            r -= 1
```

<details style="margin-inline-start:1.25rem">
<summary>Valid Palindrome II</summary>

- **Contrast:** Compare string ends instead of numeric totals. Allow at most one deletion.

  ```python
  l, r = 0, len(s) - 1
  ```

- **Move:** At the first mismatch, skip either end and check the remainder. `"abca"` succeeds; `"abc"` fails.

  ```python
  while l < r:
      if s[l] != s[r]:
          skip_l, skip_r = s[l + 1:r + 1], s[l:r]
          return skip_l == skip_l[::-1] or skip_r == skip_r[::-1]
      l, r = l + 1, r - 1
  ```

</details>
</details>

#### Group 3: Binary Search & Search Space

**Idea:** Search sorted positions or possible answers. Each comparison must show which half cannot contain the answer.

<details>
<summary>4. Koko Eating Bananas</summary>

- Search speeds, not pile positions. If a speed finishes in time, every faster speed does too.
- Use ceiling division to count each pile's hours.
- Keep a working speed, then try slower ones. Assume positive piles and enough hours for one visit per pile.

```python
l, r = 1, max(piles)
ans = r
while l <= r:
    mid = (l + r) // 2
    hours = sum((p + mid - 1) // mid for p in piles)
    if hours <= h:
        ans, r = mid, mid - 1
    else:
        l = mid + 1
```

<details style="margin-inline-start:1.25rem">
<summary>Binary Search</summary>

- **Change:** Find an exact value in a sorted array, rather than the smallest working speed.

  ```python
  l, r = 0, len(arr) - 1
  ```

- **Move:** Discard `mid` after a miss. In `[1, 3, 5]`, target 3 is at index 1.

  ```python
  while l <= r:
      mid = (l + r) // 2
      if arr[mid] == target:
          return mid
      elif arr[mid] < target:
          l = mid + 1
      else:
          r = mid - 1
  ```

</details>
</details>

#### Group 4: Monotonic Stacks & String Parsing

**Idea:** A stack holds unfinished work. Waiting indices, unmatched brackets and arithmetic terms need different rules.

<details>
<summary>5. Daily Temperatures</summary>

- Store indices waiting for their first warmer day.
- Waiting temperatures stay the same or decrease from bottom to top. A warmer day resolves colder entries at the top.
- Pop the waiting index before subtracting it; days without a warmer day stay zero.

```python
ans, stack = [0] * len(temperatures), []
for i, t in enumerate(temperatures):
    while stack and t > temperatures[stack[-1]]:
        waiting = stack.pop()
        ans[waiting] = i - waiting
    stack.append(i)
```

<details style="margin-inline-start:1.25rem">
<summary>Valid Parentheses</summary>

- **Contrast:** Nested brackets close in reverse order; no temperature ordering is involved.

  ```python
  stack, mapping = [], {")": "(", "}": "{", "]": "["}
  ```

- **Move:** Match each closer with the latest opener, then require an empty stack. `"([)]"` fails.

  ```python
  for c in s:
      if c in mapping:
          if not stack or stack.pop() != mapping[c]:
              return False
      else:
          stack.append(c)
  return not stack
  ```

</details>

<details style="margin-inline-start:1.25rem">
<summary>Basic Calculator II</summary>

- **Contrast:** Parse a valid expression without parentheses. Store signed terms, but resolve multiplication and division immediately.

  ```python
  stack, num, sign = [], 0, "+"
  ```

- **Move:** A final operator flushes the last number. `"3+2*2"` gives 7; division truncates toward zero.

  ```python
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
              stack.append(int(stack.pop() / num))
          sign, num = c, 0
  ```

</details>
</details>

#### Group 5: Intervals, Heaps & Greedy

**Idea:** Sort intervals, keep useful values in a heap, or track how far earlier choices can reach.

<details>
<summary>6. Merge Intervals</summary>

- Sort by start time; only the last merged range can overlap the next interval.
- Extend that range's end, or start a new range.
- Touching endpoints merge. This fragment assumes a nonempty interval list.

```python
intervals.sort(key=lambda x: x[0])
merged = [intervals[0]]
for curr in intervals[1:]:
    if curr[0] <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], curr[1])
    else:
        merged.append(curr)
```

<details style="margin-inline-start:1.25rem">
<summary>Meeting Rooms II</summary>

- **Contrast:** Count allocated rooms instead of merging ranges. Keep one ending time per room in a min-heap.

  ```python
  intervals.sort(key=lambda x: x[0])
  min_heap = []
  ```

- **Move:** Reuse the earliest-ending room when possible. The final heap size gives the required room count.

  ```python
  for start, end in intervals:
      if min_heap and min_heap[0] <= start:
          heapq.heappop(min_heap)
      heapq.heappush(min_heap, end)
  ```

</details>
</details>

<details>
<summary>7. Kth Largest Element</summary>

- Keep the largest `k` values seen; the smallest retained value is the cutoff.
- Push each value, then pop when the heap exceeds `k` entries.
- With `1 <= k <= len(nums)`, the final heap root is the answer. Duplicates count separately.

```python
min_heap = []
for n in nums:
    heapq.heappush(min_heap, n)
    if len(min_heap) > k:
        heapq.heappop(min_heap)
```

<details style="margin-inline-start:1.25rem">
<summary>Merge k Sorted Lists</summary>

- **Contrast:** Keep one candidate head per sorted list, rather than the largest values. The source index breaks equal-value ties.

  ```python
  heap = [(node.val, i, node) for i, node in enumerate(lists) if node]
  heapq.heapify(heap)
  ```

- **Move:** Emit the smallest candidate, then advance only its source list. `[1, 4]` and `[2, 3]` merge in order.

  ```python
  while heap:
      _, i, node = heapq.heappop(heap)
      # Append node to the output before advancing its source.
      if node.next:
          heapq.heappush(heap, (node.next.val, i, node.next))
  ```

</details>
</details>

<details>
<summary>8. Jump Game</summary>

- Track the farthest position reachable from earlier reachable indices.
- An index beyond that bound cannot be reached; otherwise extend the bound with `i + n`.
- Reachability needs one bound, not a chosen jump path. Values are nonnegative.

```python
furthest = 0
for i, n in enumerate(nums):
    if i > furthest:
        return False
    furthest = max(furthest, i + n)
```

</details>

#### Group 6: Linked Lists & Caching

**Idea:** Save pointers before changing links. A map and doubly linked list make recent items easy to find and move.

<details>
<summary>9. Reverse Linked List</summary>

- `prev` heads the reversed prefix; `curr` heads the untouched suffix.
- Save the next pointer before reversing the current link.
- Once `curr` is empty, `prev` is the new head.

```python
prev, curr = None, head
while curr:
    nxt = curr.next
    curr.next = prev
    prev, curr = curr, nxt
```

<details style="margin-inline-start:1.25rem">
<summary>Merge Two Sorted Lists</summary>

- **Change:** Reuse nodes in sorted order instead of reversing them. A dummy head anchors the output.

  ```python
  dummy = curr = ListNode(0)
  ```

- **Move:** Attach the smaller head and advance its source. Attach the remaining suffix when one source ends.

  ```python
  while l1 and l2:
      if l1.val < l2.val:
          curr.next, l1 = l1, l1.next
      else:
          curr.next, l2 = l2, l2.next
      curr = curr.next
  curr.next = l1 if l1 else l2
  ```

</details>

<details style="margin-inline-start:1.25rem">
<summary>Linked List Cycle</summary>

- **Contrast:** Detect repeated node identity without changing links. Start a slow and a fast pointer together.

  ```python
  slow = fast = head
  ```

- **Move:** Advance one pointer twice as fast. Meeting means a cycle; reaching the end means no cycle.

  ```python
  while fast and fast.next:
      slow, fast = slow.next, fast.next.next
      if slow is fast:
          return True
  ```

</details>
</details>

<details>
<summary>10. LRU Cache</summary>

- A map finds each node; a doubly linked list orders keys from oldest to newest use.
- Each read or update moves its node to the newest end.
- On overflow, evict the oldest node from both structures. Dummy endpoints keep rewiring uniform.

```python
# self.cache = {}  (key -> node)
# self.head.next = self.tail; self.tail.prev = self.head
# self._remove(node); self._add_to_end(node)
# overflow: evict self.head.next from the list and map
```

</details>

#### Group 7: Trees, BSTs & Tries

**Idea:** Combine child results, process one tree level at a time, or pass value bounds down to children.

<details>
<summary>11. Diameter of Binary Tree</summary>

- Each child returns its depth. A path through this node uses both depths.
- Return one branch upward, but keep the best cross-node path separately.
- Measure diameter in edges: a one-node tree has diameter 0.

```python
self.diameter = 0
def depth(node):
    if not node:
        return 0
    l, r = depth(node.left), depth(node.right)
    self.diameter = max(self.diameter, l + r)
    return 1 + max(l, r)
```

<details style="margin-inline-start:1.25rem">
<summary>Maximum Depth of Binary Tree</summary>

- **Change:** Return only the longest downward path. Empty subtrees contribute zero.

  ```python
  if not root:
      return 0
  ```

- **Move:** Add one node to the deeper child path. No separate diameter total is needed.

  ```python
  def maxDepth(root):
      return (1 + max(maxDepth(root.left), maxDepth(root.right))
              if root else 0)
  ```

</details>

<details style="margin-inline-start:1.25rem">
<summary>Binary Tree Level Order Traversal</summary>

- **Contrast:** Group nodes by distance from the root. A queue represents one level at each outer iteration.

  ```python
  queue, res = deque([root] if root else []), []
  ```

- **Move:** Process only the initial queue length. Newly queued children belong to the next level.

  ```python
  while queue:
      res.append([node.val for node in queue])
      for _ in range(len(queue)):
          node = queue.popleft()
          if node.left:
              queue.append(node.left)
          if node.right:
              queue.append(node.right)
  ```

</details>

<details style="margin-inline-start:1.25rem">
<summary>Validate Binary Search Tree</summary>

- **Contrast:** Pass allowed value bounds down to children instead of combining child depths.

  ```python
  low, high = -float("inf"), float("inf")
  ```

- **Move:** Tighten the upper bound on the left and lower bound on the right. Strict bounds reject duplicates.

  ```python
  def validate(node, low=-float("inf"), high=float("inf")):
      return (not node or low < node.val < high
              and validate(node.left, low, node.val)
              and validate(node.right, node.val, high))
  ```

</details>
</details>

<details>
<summary>12. Implement Trie</summary>

- Lowercase words with a shared prefix reuse the same character path.
- A terminal marker distinguishes a complete word from a prefix.
- After inserting `"apple"`, `"app"` is a prefix, not yet a stored word.

```python
node = self.root
for c in word:
    node = node.setdefault(c, {})
node["$"] = True
# search: follow existing edges and require "$" at the end
# startsWith: follow existing edges without the terminal test
```

</details>

#### Group 8: Graphs, Matrices, Shortest Paths & Coloring

**Idea:** Track connected cells, remaining prerequisites, group roots or travel times. Choose the rule that matches the task.

<details>
<summary>13. Rotting Oranges</summary>

- Start with all rotten cells; `fresh` counts the remaining fresh oranges.
- One queue layer is one minute. Mark neighbors when enqueuing them to prevent duplicate infection.
- Fresh cells left after the queue empties are unreachable.

```python
queue, minutes = deque(rotten_cells), 0
while queue and fresh:
    for _ in range(len(queue)):
        r, c = queue.popleft()
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                queue.append((nr, nc))
    minutes += 1
```

<details style="margin-inline-start:1.25rem">
<summary>Number of Islands</summary>

- **Change:** Count connected groups of land, not minutes. Each unvisited land cell starts one flood fill.

  ```python
  if grid[r][c] == "1":
      count += 1
      dfs(r, c)
  ```

- **Move:** Mark before visiting the four neighbors. Diagonal land cells remain separate islands.

  ```python
  def dfs(r, c):
      if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == "1":
          grid[r][c] = "0"
          for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
              dfs(r + dr, c + dc)
  ```

</details>

<details style="margin-inline-start:1.25rem">
<summary>Rotate Image</summary>

- **Contrast:** Change cell positions, not connections. First transpose: swap entries across the main diagonal.

  ```python
  n = len(matrix)
  for r in range(n):
      for c in range(r + 1, n):
          matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]
  ```

- **Move:** Reverse each transposed row. `[[1, 2], [3, 4]]` becomes `[[3, 1], [4, 2]]` clockwise.

  ```python
  for row in matrix:
      row.reverse()
  ```

</details>
</details>

<details>
<summary>14. Course Schedule</summary>

- Direct each edge from prerequisite to course.
- In-degree counts remaining prerequisites. A zero count means ready; finishing that course reduces its neighbors' counts.
- If fewer than `numCourses` courses are processed, a cycle remains.

```python
in_degree, adj = [0] * numCourses, defaultdict(list)
for c, p in prerequisites:
    adj[p].append(c)
    in_degree[c] += 1
queue = deque(i for i in range(numCourses) if in_degree[i] == 0)
# Pop ready courses and decrement each outgoing neighbor's in-degree.
# Enqueue a neighbor exactly when its in-degree reaches zero.
```

</details>

<details>
<summary>15. Redundant Connection</summary>

- Union-find tracks connected groups by their roots.
- An edge between equal roots closes a cycle; different roots can be joined.
- Use path compression and union by size when filling in the omitted setup.

```python
def find(node):
    if node != parent[node]:
        parent[node] = find(parent[node])
    return parent[node]

for u, v in edges:
    ru, rv = find(u), find(v)
    if ru == rv:
        return [u, v]
    if size[ru] < size[rv]:
        ru, rv = rv, ru
    parent[rv] = ru
    size[ru] += size[rv]
```

</details>

<details>
<summary>16. Network Delay Time</summary>

- With nonnegative weights, expand the smallest known arrival time first.
- Update distances when a faster route is found. Skip older heap entries with worse times.
- The latest shortest arrival is the answer. Any unreachable node makes delivery impossible.

```python
dist[k], heap = 0, [(0, k)]
while heap:
    t, u = heapq.heappop(heap)
    if t != dist[u]:
        continue
    for v, w in adj[u]:
        if t + w < dist[v]:
            dist[v] = t + w
            heapq.heappush(heap, (dist[v], v))
```

</details>

<details>
<summary>17. Is Graph Bipartite?</summary>

- Connected nodes need different colors.
- A later visit must match a node's existing color.
- Check every separate part of the graph. An odd cycle causes a color conflict.

```python
color = {}
def is_bipartite(node, c):
    if node in color:
        return color[node] == c
    color[node] = c
    return all(is_bipartite(nei, 1 - c) for nei in adj[node])

return all(is_bipartite(node, 0) for node in range(len(adj)) if node not in color)
```

</details>

#### Group 9: Dynamic Programming & String Alignment

**Idea:** Save answers for smaller problems. Define what each entry means before writing the update rule.

<details>
<summary>18. Coin Change</summary>

- Use the remaining amount as the state. Try each positive coin that fits.
- Cache the fewest coins needed for each amount.
- Zero needs no coins. Infinity means no combination fits.

```python
@lru_cache(None)
def dfs(rem):
    if rem == 0:
        return 0
    return min((dfs(rem - c) + 1 for c in coins if c <= rem),
               default=float("inf"))
```

<details style="margin-inline-start:1.25rem">
<summary>House Robber</summary>

- **Contrast:** Maximize money without taking adjacent houses. This array reminder assumes at least two houses.

  ```python
  dp = [0] * len(nums)
  dp[0], dp[1] = nums[0], max(nums[0], nums[1])
  ```

- **Move:** Skip this house, or take it after the best total two houses back. `[2, 7, 9, 3, 1]` gives 12.

  ```python
  for i in range(2, len(nums)):
      dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
  ```

</details>

<details style="margin-inline-start:1.25rem">
<summary>Word Break</summary>

- **Change:** Track which string prefixes can be made from dictionary words. Start with the empty prefix.

  ```python
  words = set(words)
  dp = [True] + [False] * len(s)
  ```

- **Move:** Extend any valid prefix with a dictionary word. `"cars"` can use `"ca"` then `"rs"`.

  ```python
  for i in range(1, len(s) + 1):
      dp[i] = any(dp[j] and s[j:i] in words for j in range(i))
  ```

</details>
</details>

<details>
<summary>19. Edit Distance</summary>

- Each cell aligns two prefixes and stores their minimum edit count.
- Matching characters use the diagonal unchanged; otherwise choose deletion, insertion or replacement.
- Empty-prefix borders cost the other prefix's length.

```python
dp[i][0], dp[0][j] = i, j
# dp[i][j] = dp[i-1][j-1] when w1[i-1] == w2[j-1]
# else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
#                  delete       insert        replace
```

</details>

#### Group 10: Backtracking & Bit Manipulation

**Idea:** Choose, explore and undo to build possible answers. XOR solves a different task by canceling paired values.

<details>
<summary>20. Combination Sum</summary>

- Build all combinations of positive, distinct candidates that total the target.
- Choose a value, explore its remainder, then undo the choice.
- Reusing index `i` allows repeated values; increasing indices avoid different orders of the same combination.

```python
res = []
def backtrack(start, rem, path):
    if rem == 0:
        res.append(list(path))
        return
    for i in range(start, len(candidates)):
        if candidates[i] > rem:
            continue
        path.append(candidates[i])
        backtrack(i, rem - candidates[i], path)
        path.pop()
```

<details style="margin-inline-start:1.25rem">
<summary>Subsets</summary>

- **Change:** Every partial path is a valid answer, including the empty path. Remove the target-sum condition.

  ```python
  res = []
  ```

- **Move:** Copy each path, then explore later indices only. Advancing to `i + 1` prevents reusing the chosen position.

  ```python
  def backtrack(start, path):
      res.append(list(path))
      for i in range(start, len(nums)):
          path.append(nums[i])
          backtrack(i + 1, path)
          path.pop()
  ```

</details>

<details style="margin-inline-start:1.25rem">
<summary>Single Number</summary>

- **Contrast:** Every value appears twice except one. XOR cancels paired values without exploring possibilities.

  ```python
  ans = 0
  ```

- **Move:** XOR all values together; order does not matter. `[2, 1, 2]` leaves 1.

  ```python
  for n in nums:
      ans ^= n
  ```

</details>
</details>

### Second pass: study nearby variations

Open a related problem and note what changes. Explain why its steps work, then solve it without the snippet.

### Mixed pass: choose the pattern yourself

Hide the pattern heading and ask:

1. What does a simple solution do repeatedly?
2. What data would avoid that repeated work?
3. Why is each step correct, and what are the time and space costs?

Try first, then take a small hint if needed. Hide the hint and try again.

## Review schedule

Try the problem again after 1, 3, 7 and 14 days. Adjust the gaps if it is too easy or difficult.
Record what went wrong:

- Choosing the method.
- What to store.
- Why the step works.
- Writing the code.
- Edge cases.
- Time and space costs.

## Quick pattern map

| Prompt clue | What to track | Method |
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
