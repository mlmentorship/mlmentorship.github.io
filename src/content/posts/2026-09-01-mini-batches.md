---
title: "Mini-Batches"
description: "Split examples into batches without dropping the final short batch."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Intermediate"
priority: "Role-specific"
aliases: []
prerequisites: []
---

> Split examples into batches without dropping the final short batch.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:mini-batches-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="mini-batches-state-title"><p class="visual-kicker">A fixed-size cursor</p><p class="visual-title" id="mini-batches-state-title">Mini-Batches: Advance by a slice and keep the final remainder</p><div class="coding-visual coding-visual--batching" data-coding-visual data-coding-mode="batching" data-coding-slug="mini-batches" role="group" aria-label="Mini-Batches: items 0:3, 3:6, 6:end -&gt; the final short batch is still yielded. Every item belongs to exactly one yielded slice, including the final short slice."><div class="coding-visual-example"><span>Concrete trace</span><strong>items 0:3, 3:6, 6:end -&gt; the final short batch is still yielded</strong></div><div class="coding-visual-sketch coding-visual-sketch--batching"><div class="coding-sketch-row"><span class="coding-sketch-pill coding-sketch-pill--state">0 : size</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--active">size : 2size</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill">remainder</span></div><p class="coding-sketch-note">one cursor partitions the input into non-overlapping slices</p></div><div class="coding-visual-flow"><div class="coding-visual-step" data-coding-step="0"><span class="coding-visual-step-label">1. Start</span><strong>cursor = 0</strong><small>Point at the first unprocessed item.</small></div><div class="coding-visual-step" data-coding-step="1"><span class="coding-visual-step-label">2. Slice</span><strong>start : start + size</strong><small>Take a full batch when possible.</small></div><div class="coding-visual-step" data-coding-step="2"><span class="coding-visual-step-label">3. Yield</span><strong>current batch</strong><small>Process the slice without changing its contents.</small></div><div class="coding-visual-step" data-coding-step="3"><span class="coding-visual-step-label">4. Advance</span><strong>cursor += size</strong><small>Repeat until the cursor reaches the end.</small></div></div><p class="coding-visual-invariant"><span>Invariant</span>Every item belongs to exactly one yielded slice, including the final short slice.</p><div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p></div><figcaption><strong>Read it this way:</strong> The range of start positions is the whole algorithm. Python clips the last slice at the sequence end, so no special final-batch branch is needed. For this problem, hold onto the concrete trace: items 0:3, 3:6, 6:end -&gt; the final short batch is still yielded.</figcaption></figure>

**Pattern:** Step through a sequence by batch size.

**Simple idea:** Slice from each start position to `start + batch_size`. Python stops the last
slice at the sequence end.

```python
from collections.abc import Iterator, Sequence

def batches(items: Sequence, batch_size: int) -> Iterator[Sequence]:
   if batch_size <= 0:
      raise ValueError("batch_size must be positive")
   for start in range(0, len(items), batch_size):
      yield items[start : start + batch_size]
```

**Cost:** $O(n)$ total iteration time and $O(batch size)$ output per step.
