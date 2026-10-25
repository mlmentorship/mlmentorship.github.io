---
title: "Sum of Two Integers"
description: "Separate sum bits from deferred carries, preserve a finite modular sum, then decode the requested signed or unsigned result."
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

XOR keeps sum bits without carry. Shifted AND records the carry that still needs to be combined.

## Problem

> Add two integers without `+` or `-`.

Use 32-bit signed operands or their unsigned word patterns. The result is signed 32-bit, with overflow wrapping.

## Picture

For one position, XOR retains a single one; two ones instead produce a carry of twice that position's weight.

| First bit | Second bit | XOR sum bit | AND carry bit |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 |
| 1 | 0 | 1 | 0 |
| 1 | 1 | 0 | 1 |

The trace shows four low columns for `7 + 5`. Its mask and computation still use 32 bits.

<!-- visual:sum-of-two-integers-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="sum-of-two-integers-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="sum-of-two-integers-state-title">Sum of Two Integers: XOR writes sum-without-carry bits while shifted AND carries shared 1 bits into the next pass.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="sum-of-two-integers" role="group" tabindex="0" aria-label="Sum of Two Integers: XOR writes sum-without-carry bits while shifted AND carries shared 1 bits into the next pass."><div class="coding-visual-example"><span>Input and goal</span><strong>Add two integers without `+` or `-`.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize-operands" role="group" aria-label="Initialize the two operands"><div class="coding-trace-frame-heading"><span>Initialize the two operands</span><strong>The loop starts with first=0111 and second=0101. Because second is nonzero, another carry-resolution pass is required.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:4" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="active-bit">read</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span></div><div class="coding-trace-meta"><span><b>example</b>first=7, second=5; four low bits of 32</span><span><b>columns</b>weights 8 4 2 1; labels are bit positions</span><span><b>first</b>0111 (7)</span><span><b>second</b>0101 (5)</span><span><b>mask</b>1111…1111 (32 bits)</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="compute-first-pass" hidden role="group" aria-label="Compute pass 1"><div class="coding-trace-frame-heading"><span>Compute pass 1</span><strong>0111 XOR 0101 = 0010. Their shared 1 bits are 0101, and shifting left gives carry 1010.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:4" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="carry-bit">carry</span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="sum-bit">xor</span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span></div><div class="coding-trace-meta"><span><b>example</b>first=7, second=5; four low bits of 32</span><span><b>columns</b>weights 8 4 2 1; labels are bit positions</span><span><b>xor</b>0111 ^ 0101 = 0010</span><span><b>andShift</b>(0111 &amp; 0101) &lt;&lt; 1 = 1010</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="commit-first-pass" hidden role="group" aria-label="Commit pass 1"><div class="coding-trace-frame-heading"><span>Commit pass 1</span><strong>The tuple assignment commits first=0010 and second=1010 together; the old operands supplied both expressions.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:4" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="carry-bit">carry</span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="sum-bit">first</span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span></div><div class="coding-trace-meta"><span><b>example</b>first=7, second=5; four low bits of 32</span><span><b>columns</b>weights 8 4 2 1; labels are bit positions</span><span><b>first</b>0010 (2)</span><span><b>second</b>1010 (10)</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="compute-second-pass" hidden role="group" aria-label="Resolve the remaining shared bit"><div class="coding-trace-frame-heading"><span>Resolve the remaining shared bit</span><strong>0010 XOR 1010 = 1000. The operands share the 2-bit, so (0010 AND 1010) shifted left is 0100.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:4" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="sum-bit">xor</span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="carry-bit">carry</span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span></div><div class="coding-trace-meta"><span><b>example</b>first=7, second=5; four low bits of 32</span><span><b>columns</b>weights 8 4 2 1; labels are bit positions</span><span><b>xor</b>0010 ^ 1010 = 1000</span><span><b>andShift</b>(0010 &amp; 1010) &lt;&lt; 1 = 0100</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="commit-second-pass" hidden role="group" aria-label="Commit pass 2"><div class="coding-trace-frame-heading"><span>Commit pass 2</span><strong>The next loop state is first=1000 and second=0100. Carry is still nonzero, so continue.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:4" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="sum-bit">first</span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="carry-bit">carry</span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span></div><div class="coding-trace-meta"><span><b>example</b>first=7, second=5; four low bits of 32</span><span><b>columns</b>weights 8 4 2 1; labels are bit positions</span><span><b>first</b>1000 (8)</span><span><b>second</b>0100 (4)</span></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="clear-final-carry" hidden role="group" aria-label="Clear the final carry"><div class="coding-trace-frame-heading"><span>Clear the final carry</span><strong>1000 and 0100 have no shared 1 bit: XOR is 1100 and shifted AND is 0000.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:4" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="sum-bit">sum</span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="carry-bit">sum</span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span></div><div class="coding-trace-meta"><span><b>example</b>first=7, second=5; four low bits of 32</span><span><b>columns</b>weights 8 4 2 1; labels are bit positions</span><span><b>xor</b>1000 ^ 0100 = 1100</span><span><b>andShift</b>(1000 &amp; 0100) &lt;&lt; 1 = 0000</span><span><b>first</b>1100 (12)</span><span><b>second</b>0000</span></div></div><div class="coding-trace-frame" data-coding-frame="6" data-frame-key="return-twelve" hidden role="group" aria-label="Return the 32-bit result"><div class="coding-trace-frame-heading"><span>Return the 32-bit result</span><strong>second is zero, so the loop stops. 12 is below the sign bit 0x80000000, therefore return first directly.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:4" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="sum-bit">8</span><span class="coding-trace-array-cell">8</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="carry-bit">4</span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">0</small></span></div><div class="coding-trace-meta"><span><b>example</b>first=7, second=5; four low bits of 32</span><span><b>columns</b>weights 8 4 2 1; labels are bit positions</span><span><b>binary</b>1100</span><span><b>decimal</b>8 + 4 = 12</span><span><b>signCheck</b>12 &lt; 0x80000000</span><span><b>result</b>12</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 7</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize the two operands</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Compute pass 1</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Commit pass 1</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Resolve the remaining shared bit</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Commit pass 2</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Clear the final carry</strong></button><button type="button" data-coding-frame-button="6"><span>7</span><strong>Return the 32-bit result</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">The prompt forbids arithmetic addition and subtraction but permits bitwise operators on fixed-width signed integers.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">At every loop boundary, first plus second modulo 2^32 equals the original sum; each pass moves unresolved carries left until second becomes zero.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Separate a binary operation into local output and deferred carry state; the same decomposition underlies half/full adders, bit-mask arithmetic, and fixed-width overflow reasoning.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> The loop starts with first=0111 and second=0101. Because second is nonzero, another carry-resolution pass is required. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

## Code

Both right-hand expressions read the old operands. Tuple assignment commits their results together.

```python
def get_sum(first: int, second: int) -> int:
   mask = 0xFFFFFFFF
   while second:
      first, second = (first ^ second) & mask, ((first & second) << 1) & mask
   return first if first < 0x80000000 else ~(first ^ mask)
```

### Preserve the Modular Sum

At each loop boundary, `first + second` modulo `2**32` equals the original modular sum.

After the first commit, both stored values are masked words. Any lowest unresolved carry moves left each following pass, eventually vanishing or exiting the word.

The pictured pair progresses `(7,5) -> (2,10) -> (8,4) -> (12,0)`.

Without finite masking, Python sign extension can propagate a carry indefinitely.

### Decode the Completed Word

When carry is zero, the low word is complete. Values below `0x80000000` are already nonnegative signed results.

For larger words, `~(first ^ mask)` decodes the same bits as a negative Python integer.

For example, `2147483647 + 1` wraps to `-2147483648`, not the unbounded mathematical sum.

The source assumes 32-bit operands. Arbitrary large inputs are not normalized when the loop is skipped.

For a fixed 32-bit width, time and space are $O(1)$. Word width must be included when the interface changes.

**Recall:** Why must the two expressions use old operands? Updating `first` early changes which original pairs generate carries.

<!-- learning:core-end -->

## Declare a Signed Width

**Choose this when:** The finite signed representation is an input parameter rather than 32 bits.

Normalize both operands before entering the loop. This also handles arbitrary integer inputs modulo the declared width.

```python
def add_signed_width(first, second, width):
   if width <= 0:
      raise ValueError("positive width required")
   mask = ~(~0 << width)
   first, second = first & mask, second & mask
   while second:
      first, second = (first ^ second) & mask, ((first & second) << 1) & mask
   sign = (mask >> 1) ^ mask
   return first if not first & sign else ~(first ^ mask)
```

Width must be a positive integer. The mask and sign formulas avoid arithmetic addition and subtraction.

At width eight, `127` with `1` gives `-128`; `-1` with `1` gives zero.

Carry resolution takes at most `width` passes. Python's bit operations and retained integer storage depend on width.

## Return an Unsigned Pattern

**Choose this when:** The consumer wants an integer in `0..2**32-1`, not signed decoding.

Normalize both operands, call the reference, then mask the signed result back to its word pattern.

```python
def add_unsigned32(first, second):
   mask = 0xFFFFFFFF
   return get_sum(first & mask, second & mask) & mask
```

`0xFFFFFFFF` with one wraps to zero. `2147483647` with one gives unsigned `2147483648`.

The same completed bits can therefore have different signed and unsigned interpretations. Overflow still wraps modulo `2**32`.

Normalizing before the call matters when `second` becomes zero before any loop iteration. For this fixed width, time and space are constant.

## Transfer

The invariant preserves a modular quantity while moving unresolved work toward a finite boundary. Changing the output interpretation does not change the carry computation.
