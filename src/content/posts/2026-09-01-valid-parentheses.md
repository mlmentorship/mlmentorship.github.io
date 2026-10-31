---
title: "Valid Parentheses"
description: "Check whether all brackets close in the correct order."
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

Determine whether a bracket string has matching types, correct nesting, and no unmatched brackets.

## Problem

**Input.** A string `text` containing only the six bracket characters `()[]{}`.

**Output.** Return `True` if every bracket is correctly matched and nested; otherwise return `False`.

**Rules.** Each opener needs a later closer of the same type.
Pairs may be nested or adjacent, but may not cross. Equal counts alone are insufficient.
Leave the string unchanged. Empty text is valid; other characters require the filtering variant below.

**Example.** `"([{}])"` and `"()[]"` return `True`.
For `"([)]"`, return `False` because the pairs cross; a lone `"("` also returns `False`.

## Picture: Close From Newest to Oldest

<!-- visual:valid-parentheses-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="valid-parentheses-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="valid-parentheses-state-title">Each closing bracket must match the most recent unmatched opening bracket</p><div class="visual-argument" data-visual-argument="valid-parentheses-state" data-argument-kind="ledger" data-argument-composition="compare"><svg viewBox="0 0 360 376" role="img" aria-label="Worked example: a closing bracket must match the most recent opener. For ([)], the ) encounters [ on top. A matching ( deeper in the stack does not make the sequence valid."><title>Worked example: a closing bracket must match the most recent opener</title><desc>For ([)], the ) encounters [ on top. A matching ( deeper in the stack does not make the sequence valid.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Openers seen so far; top at right</tspan></text><g data-proof-token="(" data-proof-row="0" data-proof-position="0" data-proof-box="20,60,154,48"><rect x="20" y="60" width="154" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="97" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="97" dy="0">(</tspan></text></g><g data-proof-token="[" data-proof-row="0" data-proof-position="1" data-proof-box="180,60,154,48"><rect x="180" y="60" width="154" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="257" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="257" dy="0">[</tspan></text></g><text x="20" y="144" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Incoming closer</tspan></text><g data-proof-token=")" data-proof-row="1" data-proof-position="0" data-proof-box="20,176,314,48"><rect x="20" y="176" width="314" height="48" rx="0" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="177" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="177" dy="0">)</tspan></text></g><text x="20" y="260" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Required top opener versus actual</tspan></text><g data-proof-token="(" data-proof-row="2" data-proof-position="0" data-proof-box="20,292,100.66666666666667,48"><rect x="20" y="292" width="100.66666666666667" height="48" rx="0" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="70.33333333333334" y="319" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="70.33333333333334" dy="0">(</tspan></text></g><g data-proof-token="[" data-proof-row="2" data-proof-position="1" data-proof-box="126.66666666666667,292,100.66666666666667,48"><rect x="126.66666666666667" y="292" width="100.66666666666667" height="48" rx="0" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="177" y="319" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="177" dy="0">[</tspan></text></g><g data-proof-token="REJECT" data-proof-row="2" data-proof-position="2" data-proof-box="233.33333333333334,292,100.66666666666667,48"><rect x="233.33333333333334" y="292" width="100.66666666666667" height="48" rx="0" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="283.6666666666667" y="319" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="283.6666666666667" dy="0">REJECT</tspan></text></g></svg></div><details class="visual-derivation" open><summary>Execution trace</summary><div class="visual-derivation-content"><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="valid-parentheses" role="group" tabindex="0" aria-label="Valid Parentheses: A closing bracket is valid only when it removes its matching newest unmatched opening bracket."><div class="coding-visual-example"><span>Input and goal</span><strong>Check whether all brackets close in the correct order.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize" role="group" aria-label="Initialize an empty stack"><div class="coding-trace-frame-heading"><span>Initialize an empty stack</span><strong>Before scanning ([{}]), there is no unmatched opening bracket.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>empty</span><small>top</small></span></div><div class="coding-trace-meta"><span><b>input</b>([{}])</span><span><b>layout</b>bottom to top</span><span><b>current</b>before index 0</span><span><b>action</b>initialize</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>top</dt><dd><strong class="coding-state-after">(0, 0)</strong></dd></div><div class="coding-state-change"><dt>input</dt><dd><strong class="coding-state-after">([{}])</strong></dd></div><div class="coding-state-change"><dt>layout</dt><dd><strong class="coding-state-after">bottom to top</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="push-round" hidden role="group" aria-label="Push the first opening"><div class="coding-trace-frame-heading"><span>Push the first opening</span><strong>At index 0, ( is not a closing key, so the implementation appends it.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>(</span><small>top</small></span></div><div class="coding-trace-meta"><span><b>input</b>([{}])</span><span><b>layout</b>bottom to top</span><span><b>current</b>index 0: (</span><span><b>action</b>push (</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>current</dt><dd><span class="coding-state-before">before index 0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">index 0: (</strong></dd></div><div class="coding-state-change"><dt>action</dt><dd><span class="coding-state-before">initialize</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">push (</strong></dd></div><div class="coding-state-change"><dt>column 0 / row 0</dt><dd><span class="coding-state-before">empty</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="push-square" hidden role="group" aria-label="Push the nested opening"><div class="coding-trace-frame-heading"><span>Push the nested opening</span><strong>At index 1, [ is newer unfinished work and sits above (.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>(</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>[</span><small>top</small></span></div><div class="coding-trace-meta"><span><b>input</b>([{}])</span><span><b>layout</b>bottom to top</span><span><b>current</b>index 1: [</span><span><b>action</b>push [</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>top</dt><dd><span class="coding-state-before">(0, 0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 0)</strong></dd></div><div class="coding-state-change"><dt>current</dt><dd><span class="coding-state-before">index 0: (</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">index 1: [</strong></dd></div><div class="coding-state-change"><dt>action</dt><dd><span class="coding-state-before">push (</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">push [</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="push-curly" hidden role="group" aria-label="Push the innermost opening"><div class="coding-trace-frame-heading"><span>Push the innermost opening</span><strong>At index 2, { becomes the only opening that the next closer may finish.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>(</span></span><span class="coding-trace-grid-cell" data-motion-key="grid-1-0"><span>[</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>{</span><small>top</small></span></div><div class="coding-trace-meta"><span><b>input</b>([{}])</span><span><b>layout</b>bottom to top</span><span><b>current</b>index 2: {</span><span><b>action</b>push {</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>top</dt><dd><span class="coding-state-before">(1, 0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(2, 0)</strong></dd></div><div class="coding-state-change"><dt>current</dt><dd><span class="coding-state-before">index 1: [</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">index 2: {</strong></dd></div><div class="coding-state-change"><dt>action</dt><dd><span class="coding-state-before">push [</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">push {</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="pop-curly" hidden role="group" aria-label="Match the curly pair"><div class="coding-trace-frame-heading"><span>Match the curly pair</span><strong>At index 3, } expects {. pop() returns {, so the scan may continue.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell" data-motion-key="grid-0-0"><span>(</span></span><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>[</span><small>top</small></span></div><div class="coding-trace-meta"><span><b>input</b>([{}])</span><span><b>layout</b>bottom to top</span><span><b>current</b>index 3: }</span><span><b>action</b>pop {; expected {</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>top</dt><dd><span class="coding-state-before">(2, 0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(1, 0)</strong></dd></div><div class="coding-state-change"><dt>current</dt><dd><span class="coding-state-before">index 2: {</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">index 3: }</strong></dd></div><div class="coding-state-change"><dt>action</dt><dd><span class="coding-state-before">push {</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">pop {; expected {</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="pop-square" hidden role="group" aria-label="Match the square pair"><div class="coding-trace-frame-heading"><span>Match the square pair</span><strong>At index 4, ] expects [. pop() returns [, preserving correct nesting.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>(</span><small>top</small></span></div><div class="coding-trace-meta"><span><b>input</b>([{}])</span><span><b>layout</b>bottom to top</span><span><b>current</b>index 4: ]</span><span><b>action</b>pop [; expected [</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>top</dt><dd><span class="coding-state-before">(1, 0)</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">(0, 0)</strong></dd></div><div class="coding-state-change"><dt>current</dt><dd><span class="coding-state-before">index 3: }</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">index 4: ]</strong></dd></div><div class="coding-state-change"><dt>action</dt><dd><span class="coding-state-before">pop {; expected {</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">pop [; expected [</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="6" data-frame-key="pop-round" hidden role="group" aria-label="Match the outer pair"><div class="coding-trace-frame-heading"><span>Match the outer pair</span><strong>At index 5, ) expects (. pop() returns (, leaving no unfinished opening.</strong></div><div class="coding-trace-grid" style="--trace-cols:1" role="group" aria-label="Grid state"><span class="coding-trace-grid-cell trace-tone-focus" data-motion-key="stack-top"><span>empty</span><small>top</small></span></div><div class="coding-trace-meta"><span><b>input</b>([{}])</span><span><b>layout</b>bottom to top</span><span><b>current</b>index 5: )</span><span><b>action</b>pop (; expected (</span><span><b>result</b>true</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>current</dt><dd><span class="coding-state-before">index 4: ]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">index 5: )</strong></dd></div><div class="coding-state-change"><dt>action</dt><dd><span class="coding-state-before">pop [; expected [</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">pop (; expected (</strong></dd></div><div class="coding-state-change"><dt>result</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">true</strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 7</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize an empty stack</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Push the first opening</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Push the nested opening</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Push the innermost opening</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Match the curly pair</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Match the square pair</strong></button><button type="button" data-coding-frame-button="6"><span>7</span><strong>Match the outer pair</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use this pattern when nested delimiters must close in reverse opening order and each closer must validate the most recent unfinished opener.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">After each processed character, the stack contains exactly the unmatched opening brackets from the processed prefix, ordered oldest at the bottom and newest at the top.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">For tags, expression delimiters, and nested scopes, store only unfinished openers and let each closer inspect the newest one; an empty stack at the end proves completion.</dd></div></dl></div><div class="visual-derivation-caption">Before scanning ([{}]), there is no unmatched opening bracket. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="valid-parentheses-state"><strong>Check:</strong> Why can equal counts of each bracket type still fail for ([)]? Why must the stack also be empty after the scan?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="valid-parentheses-state">For ([)], the ) encounters [ on top. A matching ( deeper in the stack does not make the sequence valid.</span></figcaption></figure>

| Read | Stack, oldest to newest | Action |
| --- | --- | --- |
| `(` | `(` | Push |
| `[` | `(`, `[` | Push |
| `{` | `(`, `[`, `{` | Push |
| `}` | `(`, `[` | Match and pop `{` |
| `]` | `(` | Match and pop `[` |
| `)` | Empty | Match and pop `(` |

After every accepted prefix, `opening` contains exactly the unmatched openers in their original order.
Its top is the only opener the next closer may finish; searching deeper would permit crossing pairs.
Reject a closer with no opener or the wrong top. Final emptiness proves completion only after those checks passed.

## Code

```python
def valid_parentheses(text: str) -> bool:
   opening: list[str] = []
   matching = {")": "(", "]": "[", "}": "{"}

   for char in text:
      if char not in matching:
         opening.append(char)
      elif not opening or opening.pop() != matching[char]:
         return False

   return not opening
```

Time is $O(n)$ with amortized constant-time list pushes and pops.
Space is $O(h)$ for maximum nesting depth, $O(n)$ in the worst case.
The original pushes every noncloser; it does not ignore ordinary text.

## Change 1: Ignore Nonbracket Text

**New task.** Ordinary characters are irrelevant; bracket nesting still matters.

**Picture and change.** Push only explicit openers. Ignore other nonclosers.

**Code.**

```python
def balanced_brackets_in_text(text):
   stack = []
   matching = {")": "(", "]": "[", "}": "{"}
   for character in text:
      if character in "([{":
         stack.append(character)
      elif character in matching:
         if not stack or stack.pop() != matching[character]:
            return False
   return not stack
```

**Check.** `"a(b[c])"` passes; `"a([)]"` fails. Quotes and comments have no special treatment.

**Use when.** The declared input grammar makes nonbrackets irrelevant, not for parsing a programming language.

## Change 2: Permit Only Round Parentheses

**New task.** The alphabet is only `()`; reject every other character.

**Picture and change.** One type needs only a depth counter. Every prefix must stay nonnegative; final depth must be zero.

**Code.**

```python
def balanced_round(text):
   depth = 0
   for character in text:
      if character == "(":
         depth += 1
      elif character == ")":
         depth -= 1
         if depth < 0:
            return False
      else:
         return False
   return depth == 0
```

**Check.** `"(())()"` passes; `")("` fails despite final balance zero. This variant uses constant auxiliary space.

**Use when.** Removing type distinctions makes ordered obligations reducible to one count.

## Failure: Counts Without Nesting

`"([)]"` fails as soon as `)` meets `[`. Also reject early `")"` and unfinished `"("`.

## Transfer Check

Why does a queue fail?

### Reasoned Answer

A closer finishes the most recently opened unfinished scope, not the oldest. First-in-first-out order contradicts nesting.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Separate Matching From Completion

The short-circuit `not opening or ...` prevents popping an empty stack.
A nonempty stack after the scan means unfinished work remains, even if no mismatch occurred.
An empty final stack is not enough if earlier closers were never validated.

For mixed brackets, depth alone cannot distinguish `([)]` from `([])`.
The ordered type sequence is necessary state. For one type, the counter retains everything needed and removes unnecessary type storage.

The text-ignoring variant still treats bracket characters inside quoted strings or comments as delimiters.
It is not an HTML/XML parser, a Python parser, or a tokenizer that recognizes escapes.
Use a grammar-aware parser when those constructs affect whether a bracket is active.

</div>
