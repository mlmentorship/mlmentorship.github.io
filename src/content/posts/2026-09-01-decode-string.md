---
title: "Decode String"
description: "Decode text such as `3[a2[c]]` into `accaccacc`."
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

Expand a valid encoded string whose bracketed pieces may be repeated and nested.

## Problem

**Input.** A valid encoded string `text` containing literal text and expressions of the form `count[contents]`.

**Output.** Return the fully expanded string after applying every repetition, including nested repetitions.

**Rules.** Repeat counts are positive ASCII integers immediately followed by balanced brackets.
Literal text contains no digits or bracket syntax. Adjacent pieces are concatenated in their original order.
Empty input returns `""`. Assume valid input; parsing malformed or untrusted text is a different task.

**Example.** `"3[a2[c]]"` returns `"accaccacc"`.
The inner expression becomes `"acc"`, which is then repeated three times.
For `"2[a]bc"`, return `"aabc"`.

## Picture: Save the Entire Suspended Context

<!-- visual:decode-string-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="decode-string-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="decode-string-state-title">Decode String: Each [ saves its outer string and repeat count so ] can restore exactly one nesting level.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="decode-string" role="group" tabindex="0" aria-label="Decode String: Each [ saves its outer string and repeat count so ] can restore exactly one nesting level."><div class="coding-visual-example"><span>Input and goal</span><strong>Decode text such as `3[a2[c]]` into `accaccacc`.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize" role="group" aria-label="Initialize decoder state"><div class="coding-trace-frame-heading"><span>Initialize decoder state</span><strong>Before scanning 3[a2[c]], stack=[], current=&quot;&quot;, and repeat=0.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>empty</span><small>top saved context</small></span></div><div class="coding-trace-meta"><span><b>input</b>3[a2[c]]</span><span><b>layout</b>outer to inner</span><span><b>current</b>&quot;&quot;</span><span><b>repeat</b>0</span><span><b>action</b>initialize</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="read-three" hidden role="group" aria-label="Build the outer count"><div class="coding-trace-frame-heading"><span>Build the outer count</span><strong>Reading digit 3 applies repeat = 0 * 10 + 3 = 3.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>empty</span><small>top saved context</small></span></div><div class="coding-trace-meta"><span><b>input</b>3[a2[c]]</span><span><b>layout</b>outer to inner</span><span><b>current</b>&quot;&quot;</span><span><b>repeat</b>3</span><span><b>action</b>0 * 10 + 3</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="open-outer" hidden role="group" aria-label="Save the outer context"><div class="coding-trace-frame-heading"><span>Save the outer context</span><strong>At the first [, push (&quot;&quot;, 3), then reset current=&quot;&quot; and repeat=0.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>(&quot;&quot;, 3)</span><small>top saved context</small></span></div><div class="coding-trace-meta"><span><b>input</b>3[a2[c]]</span><span><b>layout</b>outer to inner</span><span><b>current</b>&quot;&quot;</span><span><b>repeat</b>0</span><span><b>action</b>push and reset</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="append-a" hidden role="group" aria-label="Build outer text"><div class="coding-trace-frame-heading"><span>Build outer text</span><strong>Reading a appends it to the current nesting level: &quot;&quot; + &quot;a&quot; = &quot;a&quot;.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>(&quot;&quot;, 3)</span><small>top saved context</small></span></div><div class="coding-trace-meta"><span><b>input</b>3[a2[c]]</span><span><b>layout</b>outer to inner</span><span><b>current</b>&quot;a&quot;</span><span><b>repeat</b>0</span><span><b>action</b>append a</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="read-two" hidden role="group" aria-label="Build the inner count"><div class="coding-trace-frame-heading"><span>Build the inner count</span><strong>Reading digit 2 applies repeat = 0 * 10 + 2 = 2 while current stays &quot;a&quot;.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>(&quot;&quot;, 3)</span><small>top saved context</small></span></div><div class="coding-trace-meta"><span><b>input</b>3[a2[c]]</span><span><b>layout</b>outer to inner</span><span><b>current</b>&quot;a&quot;</span><span><b>repeat</b>2</span><span><b>action</b>0 * 10 + 2</span></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="open-inner" hidden role="group" aria-label="Save the nested context"><div class="coding-trace-frame-heading"><span>Save the nested context</span><strong>At the second [, push (&quot;a&quot;, 2), then reset current=&quot;&quot; and repeat=0.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>(&quot;&quot;, 3)</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>(&quot;a&quot;, 2)</span><small>top saved context</small></span></div><div class="coding-trace-meta"><span><b>input</b>3[a2[c]]</span><span><b>layout</b>outer to inner</span><span><b>current</b>&quot;&quot;</span><span><b>repeat</b>0</span><span><b>action</b>push and reset</span></div></div><div class="coding-trace-frame" data-coding-frame="6" data-frame-key="append-c" hidden role="group" aria-label="Build the inner text"><div class="coding-trace-frame-heading"><span>Build the inner text</span><strong>Reading c appends it to the empty inner string, producing current=&quot;c&quot;.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>(&quot;&quot;, 3)</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>(&quot;a&quot;, 2)</span><small>top saved context</small></span></div><div class="coding-trace-meta"><span><b>input</b>3[a2[c]]</span><span><b>layout</b>outer to inner</span><span><b>current</b>&quot;c&quot;</span><span><b>repeat</b>0</span><span><b>action</b>append c</span></div></div><div class="coding-trace-frame" data-coding-frame="7" data-frame-key="close-inner" hidden role="group" aria-label="Close the inner repeat"><div class="coding-trace-frame-heading"><span>Close the inner repeat</span><strong>The first ] pops (&quot;a&quot;, 2): &quot;a&quot; + &quot;c&quot; * 2 = &quot;acc&quot;.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>(&quot;&quot;, 3)</span><small>top saved context</small></span></div><div class="coding-trace-meta"><span><b>input</b>3[a2[c]]</span><span><b>layout</b>outer to inner</span><span><b>current</b>&quot;acc&quot;</span><span><b>repeat</b>0</span><span><b>action</b>&quot;a&quot; + &quot;c&quot; * 2</span></div></div><div class="coding-trace-frame" data-coding-frame="8" data-frame-key="close-outer" hidden role="group" aria-label="Close the outer repeat"><div class="coding-trace-frame-heading"><span>Close the outer repeat</span><strong>The final ] pops (&quot;&quot;, 3): &quot;&quot; + &quot;acc&quot; * 3 = &quot;accaccacc&quot;.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>empty</span><small>top saved context</small></span></div><div class="coding-trace-meta"><span><b>input</b>3[a2[c]]</span><span><b>layout</b>outer to inner</span><span><b>current</b>&quot;accaccacc&quot;</span><span><b>repeat</b>0</span><span><b>action</b>&quot;&quot; + &quot;acc&quot; * 3</span><span><b>result</b>accaccacc</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 9</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize decoder state</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Build the outer count</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Save the outer context</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Build outer text</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Build the inner count</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Save the nested context</strong></button><button type="button" data-coding-frame-button="6"><span>7</span><strong>Build the inner text</strong></button><button type="button" data-coding-frame-button="7"><span>8</span><strong>Close the inner repeat</strong></button><button type="button" data-coding-frame-button="8"><span>9</span><strong>Close the outer repeat</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use a context stack when repeat counts and bracketed substrings can nest, so finishing an inner region must resume text and count saved before its opening bracket.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Before each character, current is the decoded text for the active nesting level, repeat is the number parsed immediately before its next [, and every stack entry preserves one suspended outer (text, count) pair.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">When nested work temporarily replaces an outer accumulator, push exactly the outer state needed to resume it; this transfers to expression evaluation, nested tags, and recursive-descent simulation.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Before scanning 3[a2[c]], stack=[], current=&quot;&quot;, and repeat=0. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

| Event | Saved stack, outer to inner | Active `current` |
| --- | --- | --- |
| Outer `[` | `("", 3)` | `""` |
| Read a, then inner `[` | `("", 3), ("a", 2)` | `""` |
| Read c | Unchanged | `"c"` |
| Inner `]` | `("", 3)` | `"a" + "c" * 2 = "acc"` |
| Outer `]` | Empty | `"acc" * 3 = "accaccacc"` |

At `[`, save `(current, repeat)` and reset both. Each frame holds exactly what one outer level needs when its child finishes.
At `]`, pop only that frame. Build decimal counts with `repeat*10+digit`; `12[a]` needs twelve copies, not three.

## Code

```python
def decode_string(text: str) -> str:
   stack: list[tuple[str, int]] = []
   current = ""
   repeat = 0

   for char in text:
      if char.isdigit():
         repeat = repeat * 10 + int(char)
      elif char == "[":
         stack.append((current, repeat))
         current, repeat = "", 0
      elif char == "]":
         previous, count = stack.pop()
         current = previous + current * count
      else:
         current += char
   return current
```

Time includes parsing plus **all intermediate copied/materialized characters**, not just final output length.
Nested concatenations can copy a growing result repeatedly. Integer count arithmetic has additional bit costs.
Storage includes output, suspended prefixes, counts and nesting frames.

## Change 1: Compute Length Without Expansion

**New task.** Return decoded size, not text.

**Picture and change.** Replace every active and saved string with its length; keep the same nesting computation.

**Code.**

```python
def decoded_length(text):
   stack = []
   current = repeat = 0
   for character in text:
      if "0" <= character <= "9":
         repeat = repeat * 10 + int(character)
      elif character == "[":
         stack.append((current, repeat))
         current = repeat = 0
      elif character == "]":
         previous, count = stack.pop()
         current = previous + current * count
      else:
         current += 1
   return current
```

**Check.** `3[a2[c]]` gives 9; `1000000[a]` gives 1,000,000 without creating that output.

**Use when.** Only size or a preflight limit is needed. Large-integer arithmetic still costs work.

## Change 2: Cap Active Expansion Before Allocation

**New task.** Return `None` if an active expansion exceeds a nonnegative integer `limit`.

**Picture and change.** Check the resulting size before repetition/concatenation, and check literal growth.

**Code.**

```python
def decode_bounded(text, limit):
   stack = []
   current = ""
   repeat = 0
   for character in text:
      if "0" <= character <= "9":
         repeat = repeat * 10 + int(character)
      elif character == "[":
         stack.append((current, repeat))
         current, repeat = "", 0
      elif character == "]":
         previous, count = stack.pop()
         if len(previous) + len(current) * count > limit:
            return None
         current = previous + current * count
      else:
         if len(current) + 1 > limit:
            return None
         current += character
   return current
```

**Check.** Source input with cap 9 succeeds; `1000000[a]` with cap 10 returns `None` before allocating the repeated string.

**Use when.** Valid input needs a size limit. This does not cap input length, nesting depth, total work or count arithmetic.

## Failure: Saving Only the Count

The inner repeat must restore outer prefix `"a"` as well as count 2. Otherwise `"acc"` cannot be reconstructed correctly.

## Transfer Check

Why cannot one global repeat counter handle nested brackets?

### Reasoned Answer

Entering the inner repeat replaces active state. The outer count and prefix must remain suspended until that child finishes.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Distinguish Output Size From Total Copying

Start with `a` and wrap it repeatedly as `a1[previous]`.
After $r$ levels, output has $r+1$ characters, but closing brackets concatenate lengths $2,3,\ldots,r+1$.
Those nonempty-prefix concatenations repeatedly copy growing strings: their total is quadratic in $r$.
This is an operation-count argument, not a measured performance benchmark or a claim about every interpreter optimization.

A nested repeated group can also make final output exponential in encoded length.
Length-only evaluation avoids materializing it, but still needs nesting state and large integers.
With positive repeats, suspended prefixes contribute to eventual output; peak character/frame storage can be bounded by $O(n+m)$.
Here $n$ is encoded length and $m$ is decoded length, with numeric bit costs accounted separately.

The original `isdigit()` plus `int()` is not a general Unicode numeric grammar.
Malformed brackets, misplaced counts, escapes and non-ASCII digit policies require validation outside the stated contract.
The bounded variant does not provide those checks or a complete memory/time budget for hostile input.
For that use, add grammar, depth, input-size and arithmetic limits before trusting expansion.

</div>
