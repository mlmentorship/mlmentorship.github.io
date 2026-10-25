---
title: "Add parse counts only for valid final tokens"
description: "Separate one-digit and two-digit decoding choices, preserve both rolling dependencies, then enumerate witnesses or change the token vocabulary."
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

Complete decodings split into disjoint groups by final token width; add only groups whose final token is valid.

## Problem

Count decodings of ASCII digit string `2101`, where tokens `1` through `26` map to `A` through `Z`.
No token may start with zero.
The source returns zero for empty input as an external API convention.
It assumes ASCII digits rather than validating arbitrary input.

## Picture

Boundary $j$ means $j$ consumed digits.
Before processing index `i`, `one_back` counts `text[:i]` and `two_back` counts `text[:i-1]`.
A valid current single digit extends every one-back parse.
A valid pair from `10` through `26` extends every two-back parse.
These groups cannot overlap because their last token widths differ.

| Prefix | Valid final-token contributions | Count |
| --- | --- | --- |
| `2` | one initial token | `1` |
| `21` | single `1`, pair `21` | `2` |
| `210` | pair `10` only | `1` |
| `2101` | single `1`; pair `01` invalid | `1` |

The answer is `2 | 10 | 1`, or `BJA`.
Compute `current` before shifting both older values together.
Initialization uses one empty-prefix parse internally, despite the separate external empty-input policy.

<!-- visual:decode-ways-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="decode-ways-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="decode-ways-state-title">Add parse counts only for valid final tokens: At each digit, add one-back decodings for a valid single digit and two-back decodings for a valid 10..26 pair.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="decode-ways" role="group" tabindex="0" aria-label="Add parse counts only for valid final tokens: At each digit, add one-back decodings for a valid single digit and two-back decodings for a valid 10..26 pair."><div class="coding-visual-example"><span>Input and goal</span><strong>Count ways to decode digits where `1` through `26` map to letters.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize" role="group" aria-label="Initialize after the leading 2"><div class="coding-trace-frame-heading"><span>Initialize after the leading 2</span><strong>Seed one empty-prefix parse and one parse for initial digit 2.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 360 230" role="img" aria-label="Prefix boundaries: valid tokens add parse counts; explicitly invalid tokens add nothing"><defs><marker id="decode-ways-edge-0" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><g><line class="coding-trace-edge-line" x1="91" y1="112" x2="266" y2="112" marker-end="url(#decode-ways-edge-0)" /><text class="coding-trace-graph-edge-label" x="180" y="105">2</text></g><g class="coding-trace-graph-node"><circle cx="65" cy="112" r="23" /><text x="65" y="116">0</text></g><g class="coding-trace-graph-node is-focus"><circle cx="295" cy="112" r="23" /><text x="295" y="116">1</text></g></svg></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="current-digit">index 0</span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">3</small></span></div><div class="coding-trace-meta"><span><b>prefix</b>&quot;2&quot;</span><span><b>state</b>two_back = 1; one_back = 1</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="process-one" hidden role="group" aria-label="Process index 1"><div class="coding-trace-frame-heading"><span>Process index 1</span><strong>Digit 1 is nonzero, so current starts with one_back = 1. Pair 21 is valid, so add two_back = 1: current = 2.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 360 230" role="img" aria-label="Prefix boundaries: valid tokens add parse counts; explicitly invalid tokens add nothing"><defs><marker id="decode-ways-edge-1" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><g><line class="coding-trace-edge-line" x1="89.93312103012576" y1="172.6284685650063" x2="267.1899803894751" y2="120.22209275441605" marker-end="url(#decode-ways-edge-1)" /><text class="coding-trace-graph-edge-label" x="180" y="139">1</text></g><g><line class="coding-trace-edge-line" x1="90.23656088015898" y1="61.25427813116983" x2="266.85152824905344" y2="105.02407439215672" marker-end="url(#decode-ways-edge-1)" /><text class="coding-trace-graph-edge-label" x="180" y="76.5">21</text></g><g class="coding-trace-graph-node"><circle cx="65" cy="55" r="23" /><text x="65" y="59">0</text></g><g class="coding-trace-graph-node"><circle cx="65" cy="180" r="23" /><text x="65" y="184">1</text></g><g class="coding-trace-graph-node is-focus"><circle cx="295" cy="112" r="23" /><text x="295" y="116">2</text></g></svg></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="current-digit">index 1</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">3</small></span></div><div class="coding-trace-meta"><span><b>prefix</b>&quot;21&quot;</span><span><b>arithmetic</b>single 1 + pair 1 = 2</span><span><b>update</b>(two_back, one_back) = (1,2)</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="process-zero" hidden role="group" aria-label="Let zero use only pair 10"><div class="coding-trace-frame-heading"><span>Let zero use only pair 10</span><strong>At index 2, digit 0 cannot stand alone, so current starts at 0. Pair 10 is valid and adds two_back = 1: current = 1.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 360 230" role="img" aria-label="Prefix boundaries: valid tokens add parse counts; explicitly invalid tokens add nothing"><defs><marker id="decode-ways-edge-2" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><g data-invalid-transition><line class="coding-trace-edge-line is-rejected" x1="89.93312103012576" y1="172.6284685650063" x2="267.1899803894751" y2="120.22209275441605" marker-end="url(#decode-ways-edge-2)" /><text class="coding-trace-graph-edge-label" x="180" y="139">0 invalid</text></g><g><line class="coding-trace-edge-line" x1="90.23656088015898" y1="61.25427813116983" x2="266.85152824905344" y2="105.02407439215672" marker-end="url(#decode-ways-edge-2)" /><text class="coding-trace-graph-edge-label" x="180" y="76.5">10</text></g><g class="coding-trace-graph-node"><circle cx="65" cy="55" r="23" /><text x="65" y="59">1</text></g><g class="coding-trace-graph-node"><circle cx="65" cy="180" r="23" /><text x="65" y="184">2</text></g><g class="coding-trace-graph-node is-focus"><circle cx="295" cy="112" r="23" /><text x="295" y="116">3</text></g></svg></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="current-digit">index 2</span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">3</small></span></div><div class="coding-trace-meta"><span><b>prefix</b>&quot;210&quot;</span><span><b>arithmetic</b>single 0 + pair 1 = 1</span><span><b>update</b>(two_back, one_back) = (2,1)</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="process-final-one" hidden role="group" aria-label="Reject the leading-zero pair"><div class="coding-trace-frame-heading"><span>Reject the leading-zero pair</span><strong>At index 3, digit 1 is nonzero and contributes one_back = 1. Pair 01 is invalid, so it adds 0; the only decoding is 2|10|1.</strong></div><div class="coding-trace-graph"><svg viewBox="0 0 360 230" role="img" aria-label="Prefix boundaries: valid tokens add parse counts; explicitly invalid tokens add nothing"><defs><marker id="decode-ways-edge-3" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs><g><line class="coding-trace-edge-line" x1="89.93312103012576" y1="172.6284685650063" x2="267.1899803894751" y2="120.22209275441605" marker-end="url(#decode-ways-edge-3)" /><text class="coding-trace-graph-edge-label" x="180" y="139">1</text></g><g data-invalid-transition><line class="coding-trace-edge-line is-rejected" x1="90.23656088015898" y1="61.25427813116983" x2="266.85152824905344" y2="105.02407439215672" marker-end="url(#decode-ways-edge-3)" /><text class="coding-trace-graph-edge-label" x="180" y="76.5">01 invalid</text></g><g class="coding-trace-graph-node"><circle cx="65" cy="55" r="23" /><text x="65" y="59">2</text></g><g class="coding-trace-graph-node"><circle cx="65" cy="180" r="23" /><text x="65" y="184">3</text></g><g class="coding-trace-graph-node is-focus"><circle cx="295" cy="112" r="23" /><text x="295" y="116">4</text></g></svg></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="current-digit">index 3</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">3</small></span></div><div class="coding-trace-meta"><span><b>prefix</b>&quot;2101&quot;</span><span><b>arithmetic</b>single 1 + invalid pair 0 = 1</span><span><b>result</b>1</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 4</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize after the leading 2</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Process index 1</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Let zero use only pair 10</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Reject the leading-zero pair</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when each encoded token consumes one or two adjacent symbols, validity depends on the local symbol or pair, and the task asks for the number of complete parses.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">Before index i, one_back counts decodings through i-1 and two_back through i-2; current sums exactly the valid single-digit and two-digit ways ending at i.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">For variable-width parsing, sum counts from earlier boundaries only when the token ending at the current boundary is valid; zeros often eliminate the width-one transition.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Seed one empty-prefix parse and one parse for initial digit 2. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

## Code

```python
def num_decodings(text: str) -> int:
   if not text or text[0] == "0":
      return 0

   two_back = one_back = 1
   for index in range(1, len(text)):
      current = one_back if text[index] != "0" else 0
      if 10 <= int(text[index - 1 : index + 1]) <= 26:
         current += two_back
      two_back, one_back = one_back, current
   return one_back
```

The loop uses $O(n)$ additions and a constant number of integer slots.
Growing counts mean neither constant bit storage nor constant-time additions is guaranteed.

### Failure Check

`06` and `100` have zero complete decodings; `10` has one.
Accepting zero alone invents parses.
Overwriting an older count before both contributions are computed loses a dependency.

<!-- learning:core-end -->

## Change to Decoded Witnesses

**New task:** Return every decoded letter string for small inputs.

**Picture and change:** Branch on valid widths one and two, then prepend the corresponding letter to completed suffix witnesses.

```python
def decoded_strings(text):
   if not text or not text.isascii() or not text.isdecimal():
      return []
   def visit(index):
      if index == len(text):
         return [""]
      result = []
      for width in (1, 2):
         token = text[index:index + width]
         if len(token) == width and token[0] != "0" and 1 <= int(token) <= 26:
            for suffix in visit(index + width):
               result.append(chr(64 + int(token)) + suffix)
      return result
   return visit(0)
```

**Check:** `"226"` returns `["BBF","BZ","VF"]`; `"06"` returns `[]`.
The internal completed suffix contributes one empty string, not zero witnesses.
Enumeration can require exponentially many output characters and recursive calls.
It does not inherit the source's constant integer-slot storage or avoid Python's recursion limit.

**Use when:** Actual decodings are needed and their output is small.

## Change the Token Vocabulary

**New task:** Count parses from an arbitrary finite set of nonempty strings.

**Picture and change:** Keep one count per prefix boundary and add earlier counts only for valid ending slices.

```python
def codebook_parse_count(text, tokens):
   tokens = set(tokens)
   if "" in tokens:
      raise ValueError("empty token")
   ways = [1] + [0] * len(text)
   for end in range(1, len(text) + 1):
      for token in tokens:
         if len(token) <= end and text[end - len(token):end] == token:
            ways[end] += ways[end - len(token)]
   return ways[-1]
```

**Check:** `"123"` with `["1","2","3","12","123"]` has three parses: `1|2|3`, `12|3` and `123`.
Duplicate tokens do not multiply parses; empty tokens are rejected.
This generic API counts empty text as one empty parse.
Its cost includes token comparisons, slicing, growing integer additions and the full boundary-count row.

**Use when:** Validity no longer means decimal values from one through twenty-six.

### Transfer Check

What changes when only parse existence is needed?

### Answer

Replace count addition with Boolean OR while retaining the same valid-token boundary edges.
Witnesses and counts need different output evidence.
