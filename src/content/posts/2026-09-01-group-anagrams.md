---
title: "Group Anagrams"
description: "Put words with the same letters into the same group."
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

> Put words with the same letters into the same group.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:group-anagrams-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="group-anagrams-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="group-anagrams-state-title">Group Anagrams: Use one frequency signature as the address for each word group.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="group-anagrams" role="group" aria-label="Group Anagrams: Use one frequency signature as the address for each word group."><div class="coding-visual-example"><span>Input and goal</span><strong>Put words with the same letters into the same group.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Build the first bucket"><div class="coding-trace-frame-heading"><span>Build the first bucket</span><strong>eat and tea have the same sorted signature.</strong></div><div class="coding-trace-buckets"><div class="coding-trace-bucket trace-tone-focus"><strong>[a,e,t]</strong><span>eat</span><span>tea</span></div><div class="coding-trace-bucket"><strong>[a,n,t]</strong></div></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Branch on a new signature"><div class="coding-trace-frame-heading"><span>Branch on a new signature</span><strong>tan belongs under [a,n,t], while ate returns to [a,e,t].</strong></div><div class="coding-trace-buckets"><div class="coding-trace-bucket trace-tone-state"><strong>[a,e,t]</strong><span>eat</span><span>tea</span><span>ate</span></div><div class="coding-trace-bucket trace-tone-focus"><strong>[a,n,t]</strong><span>tan</span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Read the groups"><div class="coding-trace-frame-heading"><span>Read the groups</span><strong>Words sharing a signature are already together.</strong></div><div class="coding-trace-buckets"><div class="coding-trace-bucket"><strong>[a,e,t]</strong><span>eat</span><span>tea</span><span>ate</span></div><div class="coding-trace-bucket"><strong>[a,n,t]</strong><span>tan</span><span>nat</span></div><div class="coding-trace-bucket"><strong>[a,b,t]</strong><span>bat</span></div></div><div class="coding-trace-meta"><span><b>status</b>three buckets</span><span><b>result</b>three groups</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Build the first bucket</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Branch on a new signature</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Read the groups</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Use one frequency signature as the address for each word group.</p></div><figcaption><strong>Read it this way:</strong> eat and tea have the same sorted signature. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Hash map with a shared key.

**Simple idea:** Count each lowercase letter. Anagrams have the same 26 counts, so they use
the same tuple as a map key.

```python
from collections import defaultdict

def group_anagrams(words: list[str]) -> list[list[str]]:
   groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
   for word in words:
      counts = [0] * 26
      for char in word:
         counts[ord(char) - ord("a")] += 1
      groups[tuple(counts)].append(word)
   return list(groups.values())
```

**Cost:** $O(nm)$ time and $O(nm)$ space for $n$ words of average length $m$.
