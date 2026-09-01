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
<figure class="learning-figure coding-visual-figure" aria-labelledby="top-k-frequent-elements-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="top-k-frequent-elements-state-title">Top K Frequent Elements: Use frequency as a bucket coordinate, then scan from the highest bucket.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="top-k-frequent-elements" role="group" aria-label="Top K Frequent Elements: Use frequency as a bucket coordinate, then scan from the highest bucket."><div class="coding-visual-example"><span>Input and goal</span><strong>Return the `k` values that appear most often.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Count values"><div class="coding-trace-frame-heading"><span>Count values</span><strong>The counts are 1 -&gt; 3, 2 -&gt; 2, and 3 -&gt; 1.</strong></div><div class="coding-trace-buckets"><div class="coding-trace-bucket trace-tone-focus"><strong>3</strong><span>1</span></div><div class="coding-trace-bucket"><strong>2</strong><span>2</span></div><div class="coding-trace-bucket"><strong>1</strong><span>3</span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Scan high to low"><div class="coding-trace-frame-heading"><span>Scan high to low</span><strong>Take 1 from bucket 3 and 2 from bucket 2.</strong></div><div class="coding-trace-buckets"><div class="coding-trace-bucket trace-tone-output"><strong>3</strong><span>1</span></div><div class="coding-trace-bucket trace-tone-output"><strong>2</strong><span>2</span></div><div class="coding-trace-bucket"><strong>1</strong><span>3</span></div></div><div class="coding-trace-meta"><span><b>result</b>two values collected</span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Return top k"><div class="coding-trace-frame-heading"><span>Return top k</span><strong>The answer is [1,2]; no global sort is needed.</strong></div><div class="coding-trace-buckets"><div class="coding-trace-bucket trace-tone-output"><strong>3</strong><span>1</span></div><div class="coding-trace-bucket trace-tone-output"><strong>2</strong><span>2</span></div><div class="coding-trace-bucket"><strong>1</strong><span>3</span></div></div><div class="coding-trace-meta"><span><b>result</b>[1, 2]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Count values</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Scan high to low</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Return top k</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Use frequency as a bucket coordinate, then scan from the highest bucket.</p></div><figcaption><strong>Read it this way:</strong> The counts are 1 -&gt; 3, 2 -&gt; 2, and 3 -&gt; 1. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

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
