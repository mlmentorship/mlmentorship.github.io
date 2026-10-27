---
title: "Minimum Window Substring"
description: "Track missing copies, save each valid cover before destructive shrinking, and adapt to required-only counters or counting every covering range."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Advanced"
priority: "Specialist"
aliases: []
prerequisites: []
---

Find the shortest contiguous piece of text containing every required character with its requested multiplicity.

## Problem

**Input.** Two Python strings: `text` to search and `required` specifying the needed characters and their counts.

**Output.** Return the shortest substring of `text` containing at least every character count found in `required`.

**Rules.** Required character order does not matter, but repetitions do.
The selected range must be contiguous. Compare exact Python code points and preserve both strings.
An empty requirement or no qualifying range returns `""`. Equal-length ties retain the first shortest occurrence.

**Example.** For `text = "ABAAC"` and `required = "AAC"`, return `"AAC"` from half-open range `[2, 5)`.
The shorter `"AC"` does not qualify because it lacks a second A.

## Picture

For each character, `need = required count - current window count`.
Positive values are missing copies; zero is exact; negative values are surplus.
The scalar `missing` sums positive deficits, not missing distinct symbols.

| Transition | `need[A]` | `need[B]` | `need[C]` | `missing` |
| --- | --- | --- | --- | --- |
| Before reading | `2` | `0` | `1` | `3` |
| After `ABAA` | `-1` | `-1` | `1` | `1` |
| Add `C`, save `ABAAC` | `-1` | `-1` | `0` | `0` |
| Remove surplus `A`, save `BAAC` | `0` | `-1` | `0` | `0` |
| Remove surplus `B`, save `AAC` | `0` | `0` | `0` | `0` |
| Remove required `A`, stop | `1` | `0` | `0` | `1` |

Adding a character decreases `missing` only when its **old** deficit is positive. Always decrement its deficit afterward.
While valid, measure before removal. Increment the outgoing deficit first; increase `missing` only when its **new** value is positive.

Expansion cannot destroy coverage. Removed starts never need revisiting: a later right boundary would make those windows longer than already-considered candidates.
The saved bounds retain the shortest valid slice even when the current window becomes invalid.
In the visual, `L/R` mark the current inclusive window; the underline and final `bL/bR` preserve saved bounds.

<!-- visual:minimum-window-substring-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="minimum-window-substring-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="minimum-window-substring-state-title">Minimum Window Substring: Save valid covers before each left removal. Removing the first essential copy makes the window invalid again.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="minimum-window-substring" role="group" tabindex="0" aria-label="Minimum Window Substring: Save valid covers before each left removal. Removing the first essential copy makes the window invalid again."><div class="coding-visual-example"><span>Input and goal</span><strong>Find the shortest substring that contains all required characters and counts.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="initialize" role="group" aria-label="Initialize required counts"><div class="coding-trace-frame-heading"><span>Initialize required counts</span><strong>For ABAAC and required AAC, start L at zero. Need two A copies and one C, so missing is three.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">C</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>mapLabel</b>need = required count minus window count</span><span><b>range</b>empty before reading index 0</span><span><b>required</b>need A:2, C:1</span><span><b>missing</b>3</span><span><b>best</b>none</span></div><div class="coding-trace-map"><span class="coding-trace-label">need = required count minus window count</span><span class="coding-trace-map-entry"><b>A</b><span>2</span></span><span class="coding-trace-map-entry"><b>C</b><span>1</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="expand-a-0" hidden role="group" aria-label="Expand through A at index 0"><div class="coding-trace-frame-heading"><span>Expand through A at index 0</span><strong>need[A] was positive, so this A fills one requirement: missing becomes 2 and need[A] becomes 1.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state trace-tone-focus" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">C</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>mapLabel</b>need = required count minus window count</span><span><b>range</b>[0..0] = &quot;A&quot;</span><span><b>direction</b>R reads index 0</span><span><b>missing</b>from 3 to 2</span></div><div class="coding-trace-map"><span class="coding-trace-label">need = required count minus window count</span><span class="coding-trace-map-entry"><b>A</b><span>1</span></span><span class="coding-trace-map-entry"><b>C</b><span>1</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="expand-b-1" hidden role="group" aria-label="Expand through irrelevant B"><div class="coding-trace-frame-heading"><span>Expand through irrelevant B</span><strong>B has no positive deficit, so missing stays two. Decrementing need[B] to negative one records surplus.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">C</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>mapLabel</b>need = required count minus window count</span><span><b>range</b>[0..1] = &quot;AB&quot;</span><span><b>direction</b>R reads index 1</span><span><b>missing</b>2</span></div><div class="coding-trace-map"><span class="coding-trace-label">need = required count minus window count</span><span class="coding-trace-map-entry"><b>A</b><span>1</span></span><span class="coding-trace-map-entry"><b>B</b><span>-1</span></span><span class="coding-trace-map-entry"><b>C</b><span>1</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="expand-a-2" hidden role="group" aria-label="Expand through the second A"><div class="coding-trace-frame-heading"><span>Expand through the second A</span><strong>need[A] was 1, so index 2 fills the second required A: missing becomes 1 and need[A] becomes 0.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">C</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>mapLabel</b>need = required count minus window count</span><span><b>range</b>[0..2] = &quot;ABA&quot;</span><span><b>direction</b>R reads index 2</span><span><b>missing</b>from 2 to 1</span></div><div class="coding-trace-map"><span class="coding-trace-label">need = required count minus window count</span><span class="coding-trace-map-entry"><b>A</b><span>0</span></span><span class="coding-trace-map-entry"><b>B</b><span>-1</span></span><span class="coding-trace-map-entry"><b>C</b><span>1</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="expand-a-3" hidden role="group" aria-label="Record a surplus A"><div class="coding-trace-frame-heading"><span>Record a surplus A</span><strong>need[A] is already 0, so the A at index 3 is surplus: missing stays 1 and need[A] becomes -1.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-2"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-focus" role="listitem" data-motion-key="value-3"><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-4"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">C</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>mapLabel</b>need = required count minus window count</span><span><b>range</b>[0..3] = &quot;ABAA&quot;</span><span><b>direction</b>R reads index 3</span><span><b>missing</b>1</span></div><div class="coding-trace-map"><span class="coding-trace-label">need = required count minus window count</span><span class="coding-trace-map-entry"><b>A</b><span>-1</span></span><span class="coding-trace-map-entry"><b>B</b><span>-1</span></span><span class="coding-trace-map-entry"><b>C</b><span>1</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="5" data-frame-key="expand-c-4" hidden role="group" aria-label="Complete the first valid window"><div class="coding-trace-frame-heading"><span>Complete the first valid window</span><strong>C fills the final deficit, making missing = 0. Save &quot;ABAAC&quot; as the first valid window of length 5.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item trace-tone-state is-saved-range" role="listitem" data-motion-key="value-0" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-1" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-2" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-3" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-focus is-saved-range" role="listitem" data-motion-key="value-4" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">C</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>mapLabel</b>need = required count minus window count</span><span><b>range</b>[0..4] = &quot;ABAAC&quot;</span><span><b>direction</b>R reads index 4</span><span><b>missing</b>from 1 to 0</span><span><b>best</b>&quot;ABAAC&quot; (length 5)</span></div><div class="coding-trace-map"><span class="coding-trace-label">need = required count minus window count</span><span class="coding-trace-map-entry"><b>A</b><span>-1</span></span><span class="coding-trace-map-entry"><b>B</b><span>-1</span></span><span class="coding-trace-map-entry"><b>C</b><span>0</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="6" data-frame-key="shrink-a-0" hidden role="group" aria-label="Remove surplus A, then measure again"><div class="coding-trace-frame-heading"><span>Remove surplus A, then measure again</span><strong>Restore need[A] from negative one to zero and move L to one. Coverage remains complete. Next, save BAAC with length four.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item trace-tone-state is-saved-range" role="listitem" data-motion-key="value-1" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-2" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-3" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-focus is-saved-range" role="listitem" data-motion-key="value-4" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">C</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>mapLabel</b>need = required count minus window count</span><span><b>range</b>[1..4] = &quot;BAAC&quot;</span><span><b>direction</b>L moves from 0 to 1</span><span><b>reason</b>removed surplus A; still valid</span><span><b>best</b>&quot;BAAC&quot; (length 4)</span></div><div class="coding-trace-map"><span class="coding-trace-label">need = required count minus window count</span><span class="coding-trace-map-entry"><b>A</b><span>0</span></span><span class="coding-trace-map-entry"><b>B</b><span>-1</span></span><span class="coding-trace-map-entry"><b>C</b><span>0</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="7" data-frame-key="shrink-b-1" hidden role="group" aria-label="Remove surplus B, then measure again"><div class="coding-trace-frame-heading"><span>Remove surplus B, then measure again</span><strong>Restore need[B] from negative one to zero and move L to two. Next, save AAC with length three.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-state is-saved-range" role="listitem" data-motion-key="value-2" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-3" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-focus is-saved-range" role="listitem" data-motion-key="value-4" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">C</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>mapLabel</b>need = required count minus window count</span><span><b>range</b>[2..4] = &quot;AAC&quot;</span><span><b>direction</b>L moves from 1 to 2</span><span><b>reason</b>removed surplus B; still valid</span><span><b>best</b>&quot;AAC&quot; (length 3)</span><span><b>update</b>best = &quot;AAC&quot; (length 3)</span></div><div class="coding-trace-map"><span class="coding-trace-label">need = required count minus window count</span><span class="coding-trace-map-entry"><b>A</b><span>0</span></span><span class="coding-trace-map-entry"><b>B</b><span>0</span></span><span class="coding-trace-map-entry"><b>C</b><span>0</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="8" data-frame-key="break-a-2" hidden role="group" aria-label="Stop after removing a required A"><div class="coding-trace-frame-heading"><span>Stop after removing a required A</span><strong>Increment need[A] from 0 to 1, so missing becomes 1. L moves to 3 and shrinking stops because [3..4] lacks one A.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-2" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item trace-tone-state is-saved-range" role="listitem" data-motion-key="value-3" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="left-pointer">L</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-focus is-saved-range" role="listitem" data-motion-key="value-4" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="right-pointer">R</span><span class="coding-trace-array-cell">C</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>mapLabel</b>need = required count minus window count</span><span><b>range</b>[3..4] = &quot;AC&quot;</span><span><b>direction</b>L moves from 2 to 3</span><span><b>reason</b>required A removed; validity breaks</span><span><b>missing</b>from 0 to 1</span></div><div class="coding-trace-map"><span class="coding-trace-label">need = required count minus window count</span><span class="coding-trace-map-entry"><b>A</b><span>1</span></span><span class="coding-trace-map-entry"><b>B</b><span>0</span></span><span class="coding-trace-map-entry"><b>C</b><span>0</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="9" data-frame-key="result" hidden role="group" aria-label="Return the shortest saved window"><div class="coding-trace-frame-heading"><span>Return the shortest saved window</span><strong>The scan is complete and the saved slice starts at 2 with length 3, so the result is &quot;AAC&quot;.</strong></div><div class="coding-trace-array-map"><div class="coding-trace-array is-fitted" style="--trace-items:5" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem" data-motion-key="value-0"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">0</small></span><span class="coding-trace-array-item" role="listitem" data-motion-key="value-1"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">B</span><small class="coding-trace-array-index">1</small></span><span class="coding-trace-array-item trace-tone-output is-saved-range" role="listitem" data-motion-key="value-2" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="best-left">bL</span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">2</small></span><span class="coding-trace-array-item is-saved-range" role="listitem" data-motion-key="value-3" data-saved-range><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">A</span><small class="coding-trace-array-index">3</small></span><span class="coding-trace-array-item trace-tone-output is-saved-range" role="listitem" data-motion-key="value-4" data-saved-range><span class="coding-trace-array-pointer" data-motion-key="best-right">bR</span><span class="coding-trace-array-cell">C</span><small class="coding-trace-array-index">4</small></span></div><div class="coding-trace-meta"><span><b>mapLabel</b>saved best bounds</span><span><b>range</b>[2..4] = &quot;AAC&quot;</span><span><b>result</b>&quot;AAC&quot;</span></div><div class="coding-trace-map"><span class="coding-trace-label">saved best bounds</span><span class="coding-trace-map-entry"><b>best_start</b><span>2</span></span><span class="coding-trace-map-entry"><b>best_length</b><span>3</span></span></div></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 10</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Initialize required counts</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Expand through A at index 0</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Expand through irrelevant B</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Expand through the second A</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Record a surplus A</strong></button><button type="button" data-coding-frame-button="5"><span>6</span><strong>Complete the first valid window</strong></button><button type="button" data-coding-frame-button="6"><span>7</span><strong>Remove surplus A, then measure again</strong></button><button type="button" data-coding-frame-button="7"><span>8</span><strong>Remove surplus B, then measure again</strong></button><button type="button" data-coding-frame-button="8"><span>9</span><strong>Stop after removing a required A</strong></button><button type="button" data-coding-frame-button="9"><span>10</span><strong>Return the shortest saved window</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when a contiguous result must cover a multiset of required values and validity can be updated by adding or removing one boundary value.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">need[x] is the remaining deficit for x (negative means surplus), missing is the total number of required copies absent from [L..R], and best is the shortest valid window seen before the current transition.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">When validity is monotone under expansion, grow until feasible and record before each left removal. Surplus removals preserve validity; removing the first essential copy deliberately breaks it. Exact deficits support constant-time boundary updates without comparing full multisets.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> For ABAAC and required AAC, start L at zero. Need two A copies and one C, so missing is three. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

## Code

```python
from collections import Counter

def min_window(text: str, required: str) -> str:
   if not required:
      return ""

   need = Counter(required)
   missing = len(required)
   left = 0
   best_start = 0
   best_length = len(text) + 1

   for right, char in enumerate(text):
      if need[char] > 0:
         missing -= 1
      need[char] -= 1

      while missing == 0:
         length = right - left + 1
         if length < best_length:
            best_start, best_length = left, length

         left_char = text[left]
         need[left_char] += 1
         if need[left_char] > 0:
            missing += 1
         left += 1

   if best_length > len(text):
      return ""
   return text[best_start : best_start + best_length]
```

**Cost:** $O(N+M)$ expected hash work for text and requirement lengths, since each boundary moves forward at most $N$ times.
The Counter retains irrelevant text keys too: map storage depends on distinct characters in their union, not only the requirement.
Only the final answer is sliced; copying it adds storage proportional to its length.

<!-- learning:core-end -->

## Change Stored Keys or the Result

Start from `min_window`. Keep exact deficit transitions; change either the map's keys or how feasible ranges contribute.

1. **Return bounds with required-only state.**

   **New task:** Avoid irrelevant map keys and defer substring allocation.

   **Picture and change:** Update counts only when a character is a required key.
   Keep the same measure-before-remove rule, returning half-open bounds instead of a copied string.

   **Code:** Complete replacement:

   ```python
   from collections import Counter


   def min_window_range(text, required):
      if not required:
         return 0, 0
      need = Counter(required)
      missing = len(required)
      left, best = 0, None
      for right, char in enumerate(text):
         if char in need:
            if need[char] > 0:
               missing -= 1
            need[char] -= 1
         while missing == 0:
            if best is None or right + 1 - left < best[1] - best[0]:
               best = left, right + 1
            outgoing = text[left]
            if outgoing in need:
               need[outgoing] += 1
               if need[outgoing] > 0:
                  missing += 1
            left += 1
      return best
   ```

   **Check:** `ABAAC`, `AAC` gives `(2,5)`. Input `ab`, `aa` gives `None`; empty requirement gives `(0,0)`.

   **Use when:** A large input alphabet or an index-based caller makes required-only state useful.
   Expected time stays $O(N+M)$; map storage becomes $O(K)$ for distinct required characters.

2. **Count every covering substring.**

   **New task:** Count nonempty ranges that cover all required copies, not only a shortest one.

   **Picture and change:** Shrink until invalid as before. Exactly the `left` earlier starts give valid windows ending at this right boundary.
   Add that count rather than keeping a minimum.

   **Code:** Complete replacement:

   ```python
   from collections import Counter


   def count_cover_windows(text, required):
      if not required:
         return len(text) * (len(text) + 1) // 2
      need = Counter(required)
      missing = len(required)
      left = answer = 0
      for char in text:
         if char in need:
            if need[char] > 0:
               missing -= 1
            need[char] -= 1
         while missing == 0:
            outgoing = text[left]
            if outgoing in need:
               need[outgoing] += 1
               if need[outgoing] > 0:
                  missing += 1
            left += 1
         answer += left
      return answer
   ```

   **Check:** `AAA`, `AA` gives three: two length-two ranges and one length-three range.
   Input `AB`, `A` gives two. Empty requirement counts all $N(N+1)/2$ nonempty ranges.

   **Use when:** The number of feasible occurrences matters.
   Expected time remains $O(N+M)$ and map storage $O(K)$; answer bit width grows with input length.

## Rebuild the Removal Test

Why inspect the outgoing deficit after incrementing it?

**Answer:** A transition from negative to zero removes surplus. A transition from zero to one creates a missing required copy and stops shrinking.
