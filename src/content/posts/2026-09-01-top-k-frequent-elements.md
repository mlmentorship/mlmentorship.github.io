---
title: "Top K Frequent Elements"
description: "Return the `k` values that appear most often."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Intermediate"
priority: "Core"
aliases: []
prerequisites: []
---

> Return the `k` values that appear most often.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:top-k-frequent-elements-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="top-k-frequent-elements-state-title"><p class="visual-kicker">Counts become buckets</p><p class="visual-title" id="top-k-frequent-elements-state-title">Top K Frequent Elements: Turn frequency into a coordinate you can scan</p><div class="coding-visual coding-visual--frequency" data-coding-visual data-coding-mode="frequency" data-coding-slug="top-k-frequent-elements" role="group" aria-label="Top K Frequent Elements: [1,1,1,2,2,3], k=2 -&gt; buckets 3:[1], 2:[2]. Every value appears in the bucket matching its complete frequency."><div class="coding-visual-example"><span>Concrete trace</span><strong>[1,1,1,2,2,3], k=2 -&gt; buckets 3:[1], 2:[2]</strong></div><div class="coding-visual-sketch coding-visual-sketch--frequency"><div class="coding-sketch-buckets"><span class="coding-sketch-bucket"><b>3</b> value</span><span class="coding-sketch-bucket"><b>2</b> value, value</span><span class="coding-sketch-bucket"><b>1</b> value</span></div><p class="coding-sketch-note">frequency is the bucket coordinate; scan from the largest count</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Count</span><strong>value → frequency</strong><small>Build one count for each distinct value.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Place</span><strong>frequency bucket</strong><small>Put the value at the coordinate named by its count.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Scan</span><strong>high to low</strong><small>Read the buckets in the order the answer needs.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Stop</span><strong>top k values</strong><small>Return as soon as enough values are collected.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Every value appears in the bucket matching its complete frequency.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The buckets replace repeated sorting. Frequency is now position, so walking from the largest bucket exposes the most common values first. For this problem, hold onto the concrete trace: [1,1,1,2,2,3], k=2 -&gt; buckets 3:[1], 2:[2].</figcaption></figure>

**Pattern:** Frequency buckets.

**Simple idea:** Count each value. Put it in a bucket named by its count. Read buckets from
high count to low count.

```python
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
   frequencies = Counter(nums)
   buckets: list[list[int]] = [[] for _ in range(len(nums) + 1)]

   for num, frequency in frequencies.items():
      buckets[frequency].append(num)

   answer: list[int] = []
   for bucket in reversed(buckets):
      answer.extend(bucket)
      if len(answer) >= k:
         return answer[:k]
   return answer
```

**Cost:** $O(n)$ time and $O(n)$ space.
