---
title: "Missing Number"
description: "Cancel expected and observed integer values under an exact one-omission contract, then change validation or omission count."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Mixed"
priority: "Core"
aliases: []
prerequisites: []
---

Find the sole missing integer when every other member of a consecutive domain appears exactly once.

## Problem

**Input.** An integer list `nums` of length `n`, containing distinct values from the inclusive domain `0..n`.

**Output.** Return the one domain value absent from the list.

**Rules.** Exactly one member is missing, and all other members appear once.
Input order is arbitrary. Duplicates, out-of-range values, and multiple omissions violate the contract.
Do not modify the list. Empty input has domain `{0}` and returns zero.

**Example.** `[3, 0, 1]` has length 3, so its domain is `0..3`; return `2`.
For `[0, 1]`, return `2`, the missing upper endpoint.

## Picture

The pictured input `[3,0,1]` omits the value `2`. Pairing equal values explains the result without requiring execution to follow this order:

| Expected | Observed | XOR contribution |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | absent | 2 |
| 3 | 3 | 0 |

<!-- visual:missing-number-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="missing-number-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="missing-number-state-title">Paired values cancel; the missing value remains</p><div class="visual-argument" data-visual-argument="missing-number-state" data-argument-kind="ledger" data-argument-composition="set"><svg viewBox="0 0 360 376" role="img" aria-label="Worked example: paired XOR terms cancel and leave the absent value. For n=3, observations (3,0,1) omit value 2. XORing observations against the full range (0,1,2,3) leaves only that unpaired value."><title>Worked example: paired XOR terms cancel and leave the absent value</title><desc>For n=3, observations (3,0,1) omit value 2. XORing observations against the full range (0,1,2,3) leaves only that unpaired value.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Expected range</tspan></text><g data-proof-token="0" data-proof-row="0" data-proof-position="0" data-proof-box="20,60,74,48"><rect x="20" y="60" width="74" height="48" rx="18" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="57" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="57" dy="0">0</tspan></text></g><g data-proof-token="1" data-proof-row="0" data-proof-position="1" data-proof-box="100,60,74,48"><rect x="100" y="60" width="74" height="48" rx="18" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="137" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="137" dy="0">1</tspan></text></g><g data-proof-token="2" data-proof-row="0" data-proof-position="2" data-proof-box="180,60,74,48"><rect x="180" y="60" width="74" height="48" rx="18" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="217" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="217" dy="0">2</tspan></text></g><g data-proof-token="3" data-proof-row="0" data-proof-position="3" data-proof-box="260,60,74,48"><rect x="260" y="60" width="74" height="48" rx="18" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="297" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="297" dy="0">3</tspan></text></g><text x="20" y="144" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Observed values</tspan></text><g data-proof-token="3" data-proof-row="1" data-proof-position="0" data-proof-box="20,176,100.66666666666667,48"><rect x="20" y="176" width="100.66666666666667" height="48" rx="18" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="70.33333333333334" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="70.33333333333334" dy="0">3</tspan></text></g><g data-proof-token="0" data-proof-row="1" data-proof-position="1" data-proof-box="126.66666666666667,176,100.66666666666667,48"><rect x="126.66666666666667" y="176" width="100.66666666666667" height="48" rx="18" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="177" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="177" dy="0">0</tspan></text></g><g data-proof-token="1" data-proof-row="1" data-proof-position="2" data-proof-box="233.33333333333334,176,100.66666666666667,48"><rect x="233.33333333333334" y="176" width="100.66666666666667" height="48" rx="18" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="283.6666666666667" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="283.6666666666667" dy="0">1</tspan></text></g><text x="20" y="260" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Unpaired XOR remainder</tspan></text><g data-proof-token="2" data-proof-row="2" data-proof-position="0" data-proof-box="20,292,314,48"><rect x="20" y="292" width="314" height="48" rx="18" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="177" y="321.5" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="177" dy="0">2</tspan></text></g></svg></div><details class="visual-derivation" open><summary>Execution trace</summary><div class="visual-derivation-content"><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="missing-number" role="group" tabindex="0" aria-label="Missing Number: Initialize with n, then XOR every expected index and actual value so matched numbers cancel and the missing value survives."><div class="coding-visual-example"><span>Input and goal</span><strong>Values come from 0 through `n`, with one missing. Return the missing value.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize" role="group" aria-label="Seed the unmatched endpoint"><div class="coding-trace-frame-heading"><span>Seed the unmatched endpoint</span><strong>Seed missing with 3 to include the domain endpoint omitted by enumerate.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:3" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="scan">i=0</span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>expectedDomain</b>[0, 1, 2, 3]</span><span><b>accumulator</b>missing = len(nums) = 3 = 0011₂</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>i</dt><dd><strong class="coding-state-after">0</strong></dd></div><div class="coding-state-change"><dt>expected domain</dt><dd><strong class="coding-state-after">[0, 1, 2, 3]</strong></dd></div><div class="coding-state-change"><dt>accumulator</dt><dd><strong class="coding-state-after">missing = len(nums) = 3 = 0011₂</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="index-0" hidden role="group" aria-label="XOR index 0 and value 3"><div class="coding-trace-frame-heading"><span>XOR index 0 and value 3</span><strong>missing = 3 XOR 0 XOR 3 = 0; XOR order does not change the eventual cancellations.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:3" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="scan">i=0</span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>before</b>3 = 0011₂</span><span><b>operation</b>3 xor 0 xor 3 = 0</span><span><b>binaryResult</b>0000₂</span><span><b>cancellationsSeen</b>3 xor 3 = 0</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>before</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3 = 0011₂</strong></dd></div><div class="coding-state-change"><dt>operation</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">3 xor 0 xor 3 = 0</strong></dd></div><div class="coding-state-change"><dt>binary result</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0000₂</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="index-1" hidden role="group" aria-label="XOR index 1 and value 0"><div class="coding-trace-frame-heading"><span>XOR index 1 and value 0</span><strong>missing = 0 XOR 1 XOR 0 = 1; XOR order does not change the eventual cancellations.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:3" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="scan">i=1</span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>before</b>0 = 0000₂</span><span><b>operation</b>0 xor 1 xor 0 = 1</span><span><b>binaryResult</b>0001₂</span><span><b>cancellationsSeen</b>0 xor 0 = 0</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>i</dt><dd><span class="coding-state-before">0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1</strong></dd></div><div class="coding-state-change"><dt>before</dt><dd><span class="coding-state-before">3 = 0011₂</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0 = 0000₂</strong></dd></div><div class="coding-state-change"><dt>operation</dt><dd><span class="coding-state-before">3 xor 0 xor 3 = 0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0 xor 1 xor 0 = 1</strong></dd></div></dl><div class="coding-numeric-transition"><svg viewBox="0 0 360 158" role="img" aria-label="i: 0 before, 1 at the next shown checkpoint." style="display:block;width:100%;max-width:34rem;margin:auto"><title>i: 0 before, 1 at the next shown checkpoint.</title><desc>Each row has its own zero-inclusive numerical scale. Circles mark the preceding authored checkpoint and squares the current checkpoint. Connecting segments compare these shown values, not an invented path of execution.</desc><text x="16" y="20" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text-soft)">SHOWN CHECKPOINTS</text><text x="344" y="20" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text-soft)">circle: before / square: now</text><g data-numeric-transition="pointer:scan" data-before="0" data-after="1" data-minimum="0" data-maximum="1"><text x="16" y="45" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text)">i</text><text x="344" y="45" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text)">0 -&gt; 1</text><path d="M28 96H332" style="stroke:var(--c-rule);fill:none"></path><path d="M28 65V111" style="stroke:var(--c-text-soft);stroke-dasharray:3 3;fill:none"></path><path d="M28 69L332 96" style="stroke:var(--viz-focus-stroke);stroke-width:2;fill:none"></path><circle data-checkpoint-before cx="28" cy="69" r="5" style="fill:var(--c-bg);stroke:var(--c-text-soft);stroke-width:2"></circle><rect data-checkpoint-after x="327" y="91" width="10" height="10" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="28" y="128" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text)">0</text><text x="332" y="128" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text)">1</text></g></svg></div></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="index-2" hidden role="group" aria-label="XOR index 2 and value 1"><div class="coding-trace-frame-heading"><span>XOR index 2 and value 1</span><strong>missing = 1 XOR 2 XOR 1 = 2; XOR order does not change the eventual cancellations.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:3" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">3</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="scan">i=2</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>before</b>1 = 0001₂</span><span><b>operation</b>1 xor 2 xor 1 = 2</span><span><b>binaryResult</b>0010₂</span><span><b>cancellationsSeen</b>1 xor 1 = 0</span><span><b>result</b>2</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>i</dt><dd><span class="coding-state-before">1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">2</strong></dd></div><div class="coding-state-change"><dt>before</dt><dd><span class="coding-state-before">0 = 0000₂</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1 = 0001₂</strong></dd></div><div class="coding-state-change"><dt>operation</dt><dd><span class="coding-state-before">0 xor 1 xor 0 = 1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1 xor 2 xor 1 = 2</strong></dd></div></dl><div class="coding-numeric-transition"><svg viewBox="0 0 360 158" role="img" aria-label="i: 1 before, 2 at the next shown checkpoint." style="display:block;width:100%;max-width:34rem;margin:auto"><title>i: 1 before, 2 at the next shown checkpoint.</title><desc>Each row has its own zero-inclusive numerical scale. Circles mark the preceding authored checkpoint and squares the current checkpoint. Connecting segments compare these shown values, not an invented path of execution.</desc><text x="16" y="20" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text-soft)">SHOWN CHECKPOINTS</text><text x="344" y="20" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text-soft)">circle: before / square: now</text><g data-numeric-transition="pointer:scan" data-before="1" data-after="2" data-minimum="0" data-maximum="2"><text x="16" y="45" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text)">i</text><text x="344" y="45" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text)">1 -&gt; 2</text><path d="M28 96H332" style="stroke:var(--c-rule);fill:none"></path><path d="M28 65V111" style="stroke:var(--c-text-soft);stroke-dasharray:3 3;fill:none"></path><path d="M180 69L332 96" style="stroke:var(--viz-focus-stroke);stroke-width:2;fill:none"></path><circle data-checkpoint-before cx="180" cy="69" r="5" style="fill:var(--c-bg);stroke:var(--c-text-soft);stroke-width:2"></circle><rect data-checkpoint-after x="327" y="91" width="10" height="10" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="28" y="128" style="font:12px var(--font-mono);text-anchor:start;fill:var(--c-text)">0</text><text x="332" y="128" style="font:12px var(--font-mono);text-anchor:end;fill:var(--c-text)">2</text></g></svg></div></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 4</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Seed the unmatched endpoint</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>XOR index 0 and value 3</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>XOR index 1 and value 0</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>XOR index 2 and value 1</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Values should contain every integer from 0 through n exactly once except one, making the actual multiset differ from the expected multiset by a single value.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">After processing indices before i, missing equals n XOR every processed expected index XOR every processed actual value. Equal values may be regrouped and cancel because XOR is associative, commutative, and self-inverse.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">When all values occur in canceling pairs except one, XOR removes order and pair placement from the problem; the same invariant finds a unique element or separates parity-based membership differences.</dd></div></dl></div><div class="visual-derivation-caption">Seed missing with 3 to include the domain endpoint omitted by enumerate. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="missing-number-state"><strong>Check:</strong> Why does seeding with 3 matter for this input? Which value survives all pair cancellations?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="missing-number-state">For n=3, observations (3,0,1) omit value 2. XORing observations against the full range (0,1,2,3) leaves only that unpaired value.</span></figcaption></figure>

## Code

`enumerate(nums)` supplies expected indices `0..n-1`, not endpoint `n`. Seed `missing = n` to complete the expected side.

```python
def missing_number(nums: list[int]) -> int:
   missing = len(nums)
   for index, num in enumerate(nums):
      missing ^= index ^ num
   return missing
```

### Why One Value Survives

After a processed prefix, `missing` XORs the seed, every processed expected index, and every processed observation.

XOR is associative and commutative. Every equal pair cancels because `value ^ value == 0`, and zero changes nothing.

At completion, each present domain value appears twice. Only the absent value appears once.

The actual accumulator trace is `3 -> 0 -> 1 -> 2`. Intermediate accumulators need not be missing values.

### Premise and Cost

Empty input omits zero and returns zero. Input order does not affect the result, and the input is unchanged.

Duplicate `[0,0,2]` violates the premise. The source returns two, which is present; XOR has not validated the observations.

The reference performs $O(n)$ XOR operations with a constant number of integer slots. Python's bit costs depend on integer width.

**Recall:** What information does XOR discard? Equal pairs disappear, so only occurrence parity remains, not full multiplicity.

<!-- learning:core-end -->

## Validate the Universe

**Choose this when:** Duplicate or out-of-range observations should return `None`.

This helper still requires integer elements; it does not parse or coerce arbitrary types.

```python
def missing_number_checked(nums):
   unique = set(nums)
   if len(unique) != len(nums) or any(
      value < 0 or value > len(nums) for value in nums
   ):
      return None
   return missing_number(nums)
```

It rejects `[0,0]` and accepts `[3,0,1]`, returning two. Distinct in-range observations certify exactly one omission because the domain has one extra member.

Validation adds $O(n)$ set entries and expected $O(n)$ hash work. It leaves the input unchanged.

## Recover Two Missing Values

**Choose this when:** `n` distinct observations come from `0..n+1`, leaving exactly two omissions.

The total XOR is `missing_a ^ missing_b`, not either answer alone. Distinct missing values differ in at least one bit.

Use the lowest differing bit to separate them. Equal present values still cancel within the same bucket.

```python
def two_missing_numbers(nums):
   difference = 0
   for value in range(len(nums) + 2):
      difference ^= value
   for value in nums:
      difference ^= value
   distinguishing_bit = difference & -difference
   first = 0
   for value in range(len(nums) + 2):
      if value & distinguishing_bit:
         first ^= value
   for value in nums:
      if value & distinguishing_bit:
         first ^= value
   return sorted([first, difference ^ first])
```

For `[0,1,3]`, it returns `[2,4]`. Empty input returns `[0,1]`. Output is sorted, and input is unchanged.

This helper assumes its new domain contract; it does not validate it. It uses $O(n)$ XOR operations and constant integer slots, excluding width-dependent bit storage.

## Transfer

One XOR cannot reconstruct an arbitrary missing set. Cancellation works because the contract limits which parity differences can remain.
