---
title: "Streaming automatic speech recognition"
description: "Emit transcripts with bounded latency using chunked encoders, monotonic alignment, endpointing, and stability-aware evaluation."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Show tentative words before speech ends, but define when they may change and when the transcript becomes final.

## Problem

One synthetic trace uses a shared clock, not measured service guarantees:

| Event | Time |
| --- | ---: |
| Speech starts | 0 ms |
| First word ends | 300 ms |
| First displayed partial includes that word | 460 ms |
| Speech ends | 1,200 ms |
| Final text appears | 1,550 ms |

The first partial is "play the"; later audio changes it to "play their song".
Compute first-text, word-emission and finalization delays without changing their anchors.
Separately, a ten-second clip needs two seconds of processing.
Does that compute ratio certify the live experience?

## Picture: Useful Early Text Can Still Change

<!-- visual:streaming-asr-provisional-to-final -->
<figure class="learning-figure plot-panel" aria-labelledby="streaming-asr-timeline-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="streaming-asr-timeline-title">Visible partial words remain revisable until the finalization rule permits commitment</p><div class="visual-argument" data-visual-argument="streaming-asr-provisional-to-final" data-argument-kind="ledger"><p class="visual-argument-premise">Visible partial words remain revisable until commitment</p><svg viewBox="0 0 360 322" role="img" aria-label="Visible partial words remain revisable until commitment. The partial hypothesis changes as audio arrives. Only the declared finalization rule turns a provisional token sequence into committed output."><title>Visible partial words remain revisable until commitment</title><desc>The partial hypothesis changes as audio arrives. Only the declared finalization rule turns a provisional token sequence into committed output.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Early partial</tspan></text><rect data-proof-token="I" x="20" y="58" width="156" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="98" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="98" dy="0">I</tspan></text><rect data-proof-token="scream" x="180" y="58" width="156" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="258" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="258" dy="0">scream</tspan></text><text x="20" y="126" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Revised partial</tspan></text><rect data-proof-token="ice" x="20" y="156" width="156" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="98" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="98" dy="0">ice</tspan></text><rect data-proof-token="cream" x="180" y="156" width="156" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="258" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="258" dy="0">cream</tspan></text><text x="20" y="224" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Finalization receipt</tspan></text><rect data-proof-token="ice cream" x="20" y="254" width="156" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="98" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="98" dy="0">ice cream</tspan></text><rect data-proof-token="committed" x="180" y="254" width="156" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="258" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="258" dy="0">committed</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 352" role="img" aria-labelledby="streaming-asr-svg-title streaming-asr-svg-desc">
		<title id="streaming-asr-svg-title">A streaming transcript changes from provisional to final as audio arrives</title>
		<desc id="streaming-asr-svg-desc">Audio arrives from left to right as the words play, their, and song, followed by a pause. After a bounded chunk and right-context wait, the recognizer emits the provisional words play the. Once the word song arrives, the recognizer revises the to their and adds song. Only after an endpointing wait during the pause does it mark play their song final. Thus first-token latency ends before speech ends, while finalization latency includes endpointing.</desc>
		<text class="viz-axis-label" x="12" y="18">AUDIO ARRIVES LEFT TO RIGHT · CONCEPTUAL, NOT TO SCALE</text>
		<path class="viz-axis" d="M18 72H342"></path>
		<path d="M334 68L342 72L334 76" style="fill:none;stroke:var(--c-text-soft);stroke-width:1.4"></path>
		<rect class="viz-node viz-node--input" x="25" y="37" width="62" height="27" rx="4"></rect>
		<rect class="viz-node viz-node--input" x="94" y="37" width="62" height="27" rx="4"></rect>
		<rect class="viz-node viz-node--input" x="163" y="37" width="62" height="27" rx="4"></rect>
		<rect class="viz-node" x="232" y="37" width="94" height="27" rx="4" style="stroke-dasharray:4 3"></rect>
		<text class="viz-node-value" x="56" y="55">/play/</text>
		<text class="viz-node-value" x="125" y="55">/their/</text>
		<text class="viz-node-value" x="194" y="55">/song/</text>
		<text class="viz-node-value" x="279" y="55">pause / wait</text>
		<path d="M156 68V326" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:4 4"></path>
		<path d="M225 68V326" style="fill:none;stroke:var(--viz-edge);stroke-width:1.2;stroke-dasharray:4 4"></path>
		<path d="M326 68V326" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.5;stroke-dasharray:4 4"></path>
		<text class="viz-axis-label" x="12" y="102">CHECKPOINT 1 · EARLY PARTIAL · 460 ms</text>
		<text class="viz-label" x="12" y="119">chunk + bounded right context are available</text>
		<rect class="viz-node viz-node--input" x="44" y="132" width="72" height="32" rx="4"></rect>
		<rect class="viz-node viz-node--focus" x="124" y="132" width="72" height="32" rx="4" style="stroke-dasharray:4 3"></rect>
		<text class="viz-node-label" x="80" y="153">play</text>
		<text class="viz-node-label" x="160" y="153">the…</text>
		<text class="viz-callout" x="205" y="153">PROVISIONAL</text>
		<path d="M160 170V195M156 187L160 195L164 187" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.8"></path>
		<text class="viz-label" x="172" y="184">future audio may revise text</text>
		<text class="viz-axis-label" x="12" y="215">CHECKPOINT 2 · LATER PARTIAL</text>
		<rect class="viz-node viz-node--input" x="44" y="226" width="72" height="32" rx="4"></rect>
		<rect class="viz-node viz-node--focus" x="124" y="226" width="72" height="32" rx="4"></rect>
		<rect class="viz-node viz-node--input" x="204" y="226" width="72" height="32" rx="4"></rect>
		<text class="viz-node-label" x="80" y="247">play</text>
		<text class="viz-node-label" x="160" y="247">their</text>
		<text class="viz-node-label" x="240" y="247">song</text>
		<text class="viz-label" x="284" y="239">the → their</text>
		<text class="viz-axis-label" x="284" y="254">REVISED</text>
		<path d="M240 264V284M236 276L240 284L244 276" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.8"></path>
		<text class="viz-label" x="12" y="278">speech stops; endpoint must fire</text>
		<text class="viz-axis-label" x="12" y="301">FINAL RESULT · 1550 ms</text>
		<rect class="viz-node viz-node--output" x="44" y="309" width="232" height="32" rx="4"></rect>
		<rect x="48" y="313" width="224" height="24" rx="2" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1"></rect>
		<text class="viz-node-label" x="160" y="330">play their song</text>
		<text class="viz-callout" x="286" y="330">FINAL</text>
	</svg>
	<div class="visual-derivation-caption">compare partial, revised and final text, not geometric distances on this schematic clock. First text precedes speech end; finalization includes endpointing and remaining delivery.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="streaming-asr-provisional-to-final"><strong>Check:</strong> Why can play their song be visible before the system is allowed to mark it final?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="streaming-asr-provisional-to-final">The partial hypothesis changes as audio arrives. Only the declared finalization rule turns a provisional token sequence into committed output.</span></figcaption>
</figure>

## Calculate Distinct Delays

| Measurement | Calculation | Result |
| --- | --- | ---: |
| First text from speech start | 460 - 0 | 460 ms |
| This word's emission from word end | 460 - 300 | 160 ms |
| Finalization from speech end | 1,550 - 1,200 | 350 ms |
| Separate clip's real-time factor | 2 / 10 | 0.2 |

Real-time factor is processing duration divided by audio duration.
It does not bound queueing, network delivery, revision frequency or finalization tails.
Measure stable-prefix time separately; an early partial is not necessarily stable.

## Endpointing and partial stability

For word-token indices, keep "play" stable and replace range $[1,2)$ of the first partial with ["their", "song"].
Advance the version; later mark the unchanged full transcript final.
Declare replacement ranges, stable-prefix length, endpoint state and reconnect/reset behavior.
Reject stale updates or changes to committed prefixes.

Aggressive endpointing can clip pauses; conservative endpointing delays finalization.
Keep tentative state reversible and measure false endpoints alongside delay.
Provisional words do not authorize consequential actions.
Even final text can require separate semantic, permission and confirmation checks.

## Change 1: More Future Context

**New task.** Additional lookahead helps recognition but threatens the deadline.

**Picture and change.** Measure actual critical-path delay, quality and cache cost under live load.

**Check.** Overlap changes incremental delay; model runtime alone is insufficient.

**Use when.** Context and responsiveness conflict.

## Change 2: Pauses Cause Premature Finalization

**New task.** The endpoint closes an unfinished request.

**Picture and change.** Adjust endpoint/stability policy and test pauses, rushed speech, noise and interruption.

**Check.** Lower mean delay can hide worse false-endpoint and task-completion rates.

**Use when.** Commitment errors dominate.

## Evaluation

| Group | Required observation |
| --- | --- |
| Recognition | Final/streaming WER with fixed normalization |
| Delay | Named first-text, word, stable-prefix and final anchors, including tails |
| Stability | Revisions/edit overhead and false endpoints |
| Resources | Real-time factor, peak compute, queue/cache behavior |
| Slices | Accent, noise, device, language and speech rate |

## Failure: Good Final WER Arrives Too Late

Accurate final text can miss a live task's deadline.
Acting on a subsequently revised provisional word creates a different failure.

## Transfer Check

Does streaming require batch size one?

### Reasoned Answer

No. Independent live streams may be batched while preserving bounded future context, state and output deadlines.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Architecture choices

| Route | Runtime contract |
| --- | --- |
| Chunked encoder | Bounded windows/lookahead, persistent caches |
| RNN-T | Time/token decoding with streaming-compatible encoder/state |
| CTC | Frame objective with compatible front end, encoder and search |
| Streaming attention | Explicitly constrained/chunked context, not full-utterance dependence |

## Latency components

Trace chunk buffering, feature extraction, lookahead, compute, decoding, endpointing, queueing, network and client rendering.
Overlap matters; measure end-to-end distributions from joint traces, not sums of stage percentiles.
Offline processing still has resource and deadline constraints even without live-context restrictions.

```text
PartialUpdate
	utterance_id, stream_epoch, version, token_replace_range
	tokens, stable_prefix_length, final, endpoint_state
	audio_event_time, display_time, model_policy_version
```

Bind replacements to the utterance/epoch and increasing version.
On reconnect, explicitly resume acknowledged state or reset; do not mix caches across utterances.
Specify stale/missing updates, bounded queues and safe timeout/fallback behavior.
Later offline rescoring needs a distinct result identity and cannot silently rewrite an already consumed final result.

## In an interview

1. Fix latency anchors and quality/stability targets.
2. Declare context, chunk, cache and decoding policy.
3. Trace buffering through client display.
4. Define tentative, stable, endpoint and final states.
5. Test fallback, rescoring, interruptions and slice monitoring.

## Common confusions

Low WER is not a delay bound; extra context changes the actual critical path, not necessarily every frame additively.

## Sources

- [He et al., 2019](https://arxiv.org/abs/1811.06621).
- [Shangguan et al., 2020](https://www.isca-archive.org/interspeech_2020/shangguan20_interspeech.html).
- [Li et al., 2020](https://research.google/pubs/towards-fast-and-accurate-streaming-end-to-end-asr/).

*Related: [automatic speech recognition](/concepts/automatic-speech-recognition/), [RNN-T](/concepts/rnn-transducer/), and [CTC](/concepts/connectionist-temporal-classification/).*

</div>
