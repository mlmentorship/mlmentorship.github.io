---
title: "Encode and Decode Strings"
description: "Frame payloads with lengths, preserve empty and hash-containing strings, and distinguish trusted character decoding from validated and byte-counted formats."
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

Read a length before interpreting payload content. The length, not the next delimiter, fixes where that payload ends.

## Problem

> Convert a list of any strings into one string and recover the exact list.

Preserve order, repeated strings, empty strings, and every Python string character. The reference decoder assumes complete output from `encode_strings`.

`[]` encodes as `""`. `[""]` encodes as `"0#"`. These are different lists.

## Picture

Write each record as `length#text`:

| Input text | Record | Payload range in complete data |
| --- | --- | --- |
| `"lint"` | `4#lint` | `[2:6]` |
| `"#"` | `1##` | `[8:9]` |
| `""` | `0#` | `[11:11]` |

Complete data is `4#lint1##0#`, with eleven character positions.

Before each decode loop, `index` starts the next header. `separator` finds that header's `#`; `length` reads its preceding decimal digits.

Move `index` after the delimiter, consume exactly `length` characters, then advance by that length.

At index six, header `1#` ends at index eight. The next `#` is one payload character, not another delimiter.

Length zero still consumes a header, appends one empty string, and advances to the next record or end.

Labels distinguish length digits (`len`), header delimiters (`sep`), and payload positions (`text`).

Length and slicing both use Python code points, not UTF-8 bytes or displayed graphemes. Header parsing never searches inside a framed payload.

<!-- visual:encode-and-decode-strings-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="encode-and-decode-strings-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="encode-and-decode-strings-state-title">Encode and Decode Strings: A decimal length before # makes each payload boundary explicit, even when the payload contains # or is empty.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="encode-and-decode-strings" role="group" tabindex="0" aria-label="Encode and Decode Strings: A decimal length before # makes each payload boundary explicit, even when the payload contains # or is empty."><div class="coding-visual-example"><span>Input and goal</span><strong>Convert a list of any strings into one string and recover the exact list.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="encode-lint" role="group" aria-label="Encode lint"><div class="coding-trace-frame-heading"><span>Encode lint</span><strong>The first input string has length 4, so encoding emits 4#lint.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="codec-cursor">encode item 0</span><span class="coding-trace-array-cell">lint</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">&quot;&quot;</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>emitted</b>4#lint</span></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="encode-hash" hidden role="group" aria-label="Encode #"><div class="coding-trace-frame-heading"><span>Encode #</span><strong>The second payload is the delimiter character itself, but its length prefix makes 1## unambiguous.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">lint</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="codec-cursor">encode item 1</span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">&quot;&quot;</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>emitted</b>4#lint1##</span></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="encode-empty" hidden role="group" aria-label="Encode the empty string"><div class="coding-trace-frame-heading"><span>Encode the empty string</span><strong>The empty payload has length 0, so it emits 0# and the complete data is 4#lint1##0#.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">lint</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="codec-cursor">encode item 2</span><span class="coding-trace-array-cell">&quot;&quot;</span><small class="coding-trace-array-index">2</small></span></div><div class="coding-trace-meta"><span><b>emitted</b>4#lint1##0#</span></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="decode-length-4" hidden role="group" aria-label="Decode length 4"><div class="coding-trace-frame-heading"><span>Decode length 4</span><strong>At index 0, separator=1 and length=int(data[0:1])=4. Move index to 2.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:6" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="decoder-index">index=0</span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">0 len</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">1 sep</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">l</span><small class="coding-trace-array-index">2 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">i</span><small class="coding-trace-array-index">3 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">n</span><small class="coding-trace-array-index">4 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">t</span><small class="coding-trace-array-index">5 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">6 len</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-7"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">7 sep</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-8"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">8 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-9"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">9 len</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-10"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">10 sep</small></span></div><div class="coding-trace-meta"><span><b>encoded</b>4#lint1##0#</span><span><b>separator</b>1</span><span><b>length</b>4</span><span><b>nextPayload</b>data[2:6]</span></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="decode-lint" hidden role="group" aria-label="Consume lint"><div class="coding-trace-frame-heading"><span>Consume lint</span><strong>Append data[2:6] = lint, then index += 4 moves from 2 to 6.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:6" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">0 len</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">1 sep</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">l</span><small class="coding-trace-array-index">2 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">i</span><small class="coding-trace-array-index">3 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">n</span><small class="coding-trace-array-index">4 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">t</span><small class="coding-trace-array-index">5 text</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-pointer" data-motion-key="decoder-index">index=6</span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">6 len</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-7"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">7 sep</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-8"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">8 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-9"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">9 len</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-10"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">10 sep</small></span></div><div class="coding-trace-meta"><span><b>encoded</b>4#lint1##0#</span><span><b>decoded</b>[&quot;lint&quot;]</span><span><b>consumed</b>[2:6] -&gt; lint</span></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="decode-hash" hidden role="group" aria-label="Decode payload #"><div class="coding-trace-frame-heading"><span>Decode payload #</span><strong>From index 6, separator=7 and length=1. Move to 8, append data[8:9] = #, then advance to 9.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:6" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">0 len</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">1 sep</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">l</span><small class="coding-trace-array-index">2 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">i</span><small class="coding-trace-array-index">3 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">n</span><small class="coding-trace-array-index">4 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">t</span><small class="coding-trace-array-index">5 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">6 len</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-7"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">7 sep</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-8"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">8 text</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-9"><span class="coding-trace-array-pointer" data-motion-key="decoder-index">index=9</span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">9 len</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-10"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">10 sep</small></span></div><div class="coding-trace-meta"><span><b>encoded</b>4#lint1##0#</span><span><b>decoded</b>[&quot;lint&quot;, &quot;#&quot;]</span><span><b>parse</b>length data[6:7]=1; payload data[8:9]=#</span></div></div><div class="coding-trace-frame" data-coding-frame="6" data-frame-key="decode-empty" hidden role="group" aria-label="Decode the empty payload"><div class="coding-trace-frame-heading"><span>Decode the empty payload</span><strong>From index 9, separator=10 and length=0. Move to 11, append data[11:11] = empty, and the loop ends.</strong></div><div class="coding-trace-array is-fitted" style="--trace-items:6" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">4</span><small class="coding-trace-array-index">0 len</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">1 sep</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">l</span><small class="coding-trace-array-index">2 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">i</span><small class="coding-trace-array-index">3 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">n</span><small class="coding-trace-array-index">4 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-5"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">t</span><small class="coding-trace-array-index">5 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-6"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span><small class="coding-trace-array-index">6 len</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-7"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">7 sep</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-8"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">8 text</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-9"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">0</span><small class="coding-trace-array-index">9 len</small></span><span class="coding-trace-array-item trace-tone-output" role="listitem" data-motion-key="value-10"><span class="coding-trace-array-pointer" data-motion-key="decoder-index">separator=10</span><span class="coding-trace-array-cell">#</span><small class="coding-trace-array-index">10 sep</small></span></div><div class="coding-trace-meta"><span><b>encoded</b>4#lint1##0#</span><span><b>decoded</b>[&quot;lint&quot;, &quot;#&quot;, &quot;&quot;]</span><span><b>parse</b>length data[9:10]=0; payload data[11:11]=&quot;&quot;</span><span><b>finalIndex</b>11 = len(data)</span><span><b>result</b>[&quot;lint&quot;, &quot;#&quot;, &quot;&quot;]</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 7</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Encode lint</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Encode #</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Encode the empty string</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Decode length 4</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Consume lint</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Decode payload #</strong></button><button type="button" data-coding-frame-button="6"><span>7</span><strong>Decode the empty payload</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Arbitrary strings, including empty strings and delimiter characters, must be concatenated and later recovered without escaping ambiguity.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">At the start of each decode loop, index points to the first decimal length digit. After locating # and consuming exactly length payload characters, index points to the next length or end of data.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">Prefix each record with a parseable size in the same unit used for slicing. Byte transport must count encoded bytes, while complete-input validation and streaming state require additional checks.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> The first input string has length 4, so encoding emits 4#lint. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

## Code

```python
def encode_strings(strings: list[str]) -> str:
   return "".join(f"{len(text)}#{text}" for text in strings)


def decode_strings(data: str) -> list[str]:
   strings = []
   index = 0

   while index < len(data):
      separator = data.index("#", index)
      length = int(data[index:separator])
      index = separator + 1
      strings.append(data[index : index + length])
      index += length
   return strings
```

Let $E$ count encoded characters, including headers. With bounded-size header arithmetic, scanning and copying take $O(E)$ work and output space.

This decoder is an inverse, not a validator. `decode_strings("3#ab")` returns `["ab"]` because Python permits short slices.
Missing delimiters or invalid numbers can raise exceptions. Signed headers are not rejected explicitly.

<!-- learning:core-end -->

## Change Validation or Units

Start from `decode_strings`. Two adjustable parts are input trust and the unit counted by each header.

1. **Reject incomplete or malformed records.**

   **New task:** Return `None` for a missing decimal header or insufficient payload.

   **Picture and change:** Keep the record-start cursor. Require ASCII decimal digits and check the endpoint before slicing.

   **Code:** Complete replacement for a complete input, not partial network chunks:

   ```python
   def decode_checked(data):
      strings = []
      index = 0
      while index < len(data):
         separator = data.find("#", index)
         if separator < 0:
            return None
         header = data[index:separator]
         if not header.isascii() or not header.isdecimal():
            return None
         try:
            length = int(header)
         except ValueError:
            return None
         start = separator + 1
         end = start + length
         if end > len(data):
            return None
         strings.append(data[start:end])
         index = end
      return strings
   ```

   **Check:** `3#ab` and `-1#a` return `None`. `4#lint1##0#` returns `["lint","#",""]`.

   Leading zeros remain valid: `00#` encodes one empty record. Conversion failures return `None`.

   **Use when:** Encoder provenance is absent. Production still needs header, record-count, payload, and total-size caps before costly conversion or allocation.

   Valid bounded-header parsing costs $O(E)$. Rejected input can require scanning its complete malformed header.

2. **Count transport bytes.**

   **New task:** Frame UTF-8 payload bytes instead of Python code points.

   **Picture and change:** Encode each payload first, count bytes, and slice bytes before decoding text.

   | Python literal | Code points | UTF-8 bytes |
   | --- | --- | --- |
   | `"\u00e9"` | 1 | 2 |
   | `"#"` | 1 | 1 |

   **Code:** This complete helper changes both sides and decodes only its own encoder output.
   Strings must be UTF-8 encodable.

   ```python
   def utf8_framed_roundtrip(strings):
      payloads = [text.encode("utf-8") for text in strings]
      data = b"".join(
         str(len(payload)).encode("ascii") + b"#" + payload
         for payload in payloads
      )
      strings = []
      index = 0
      while index < len(data):
         separator = data.index(b"#", index)
         length = int(data[index:separator])
         start = separator + 1
         strings.append(data[start:start + length].decode("utf-8"))
         index = start + length
      return strings
   ```

   **Check:** `["\u00e9","#",""]` roundtrips unchanged. Counting one code point but slicing one byte would split the first payload incorrectly.

   **Use when:** The external medium measures bytes. Work and storage scale with encoded bytes, under bounded-header assumptions.

## Rebuild the Boundary

What must survive when a chunk ends inside a payload?

**Answer:** The parse phase, remaining payload size, and partial data. Chunk boundaries are not record boundaries.

Neither format provides authentication, normalization, corruption recovery, or a streaming parser.
