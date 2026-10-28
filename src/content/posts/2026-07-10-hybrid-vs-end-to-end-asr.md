---
title: "Hybrid versus end-to-end speech recognition"
description: "Compare modular acoustic-pronunciation-language pipelines with CTC, attention, and transducer systems across data, control, latency, and operations."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Locate the changing knowledge source before choosing acoustic retraining. Text, pronunciation, acoustics and endpointing need different update evidence.

## Problem

Product names change weekly; the team has text and pronunciations, but little new audio.
For two hypothetical transcript candidates, use fixed acoustic and external-LM log scores:

| Candidate | Acoustic score $A$ | LM score $L$ |
| --- | ---: | ---: |
| Rare name | -5 | -1 |
| Common word | -2 | -4 |

Decode with $A+\lambda L$.
Can text-side preferences change the winner without changing acoustic weights?

## Picture: Find Where New Knowledge Enters

<!-- visual:asr-adaptation-boundaries -->
<figure class="learning-figure plot-panel" aria-labelledby="asr-boundaries-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="asr-boundaries-title">See which ASR knowledge sources can change without retraining the acoustic core.</p>
	<svg viewBox="0 0 360 468" role="img" aria-labelledby="asr-boundaries-svg-title asr-boundaries-svg-desc">
		<title id="asr-boundaries-svg-title">Hybrid and end-to-end ASR adaptation boundaries</title>
		<desc id="asr-boundaries-svg-desc">A hybrid path passes acoustic scores into HMM, lexicon and language-model decoding. Compatible pronunciation/text updates can change those sources without retraining acoustic weights, while graph rebuilding and regression checks may still be required. A jointly trained audio-to-token core can also use external LM, contextual bias and endpoint controls. The bottom strip contrasts explicit updates with core adaptation and allows blends; neither data volume nor an architecture label determines quality.</desc>
		<defs><marker id="asr-boundary-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
		<rect class="viz-plot-bg" x="8" y="8" width="344" height="174" rx="5"></rect>
		<text class="viz-axis-label" x="18" y="28">HYBRID · EXPLICIT MODULE BOUNDARIES</text>
		<rect class="viz-node viz-node--input" x="18" y="55" width="54" height="36" rx="3"></rect>
		<text class="viz-callout" x="45" y="77" text-anchor="middle">audio</text>
		<path d="M72 73H91" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#asr-boundary-arrow)"></path>
		<rect class="viz-node" x="98" y="48" width="86" height="50" rx="3"></rect>
		<text class="viz-callout" x="141" y="68" text-anchor="middle">acoustic model</text>
		<text class="viz-label" x="141" y="84" text-anchor="middle">frame/state scores</text>
		<path d="M184 73H203" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#asr-boundary-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="210" y="40" width="78" height="66" rx="3"></rect>
		<text class="viz-callout" x="249" y="61" text-anchor="middle">WFST decode</text>
		<text class="viz-label" x="249" y="78" text-anchor="middle">HMM + lexicon</text>
		<text class="viz-label" x="249" y="93" text-anchor="middle">+ language model</text>
		<path d="M288 73H307" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#asr-boundary-arrow)"></path>
		<rect class="viz-node viz-node--output" x="314" y="55" width="29" height="36" rx="3"></rect>
		<text class="viz-callout" x="328.5" y="77" text-anchor="middle">text</text>
		<rect class="viz-node viz-node--input" x="106" y="126" width="91" height="34" rx="3"></rect>
		<text class="viz-label" x="151.5" y="139" text-anchor="middle">pronunciations</text>
		<text class="viz-label" x="151.5" y="155" text-anchor="middle">lexicon update</text>
		<rect class="viz-node viz-node--input" x="213" y="126" width="75" height="34" rx="3"></rect>
		<text class="viz-label" x="250.5" y="139" text-anchor="middle">domain text</text>
		<text class="viz-label" x="250.5" y="155" text-anchor="middle">LM update</text>
		<path d="M197 143H205V109H221M250 126V109" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:4 3;marker-end:url(#asr-boundary-arrow)"></path>
		<text class="viz-label" x="18" y="171">Swap explicit knowledge sources; keep the acoustic model fixed.</text>
		<rect class="viz-plot-bg" x="8" y="192" width="344" height="174" rx="5"></rect>
		<text class="viz-axis-label" x="18" y="212">END TO END · JOINTLY LEARNED ACOUSTIC-TO-TOKEN CORE</text>
		<rect class="viz-node viz-node--input" x="18" y="239" width="54" height="36" rx="3"></rect>
		<text class="viz-callout" x="45" y="261" text-anchor="middle">audio</text>
		<path d="M72 257H91" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#asr-boundary-arrow)"></path>
		<rect class="viz-node viz-node--focus" x="98" y="226" width="129" height="62" rx="3"></rect>
		<text class="viz-callout" x="162.5" y="247" text-anchor="middle">encoder + token model</text>
		<text class="viz-label" x="162.5" y="264" text-anchor="middle">CTC, RNN-T, or attention</text>
		<text class="viz-label" x="162.5" y="279" text-anchor="middle">joint paired-data objective</text>
		<path d="M227 257H246" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#asr-boundary-arrow)"></path>
		<rect class="viz-node" x="253" y="239" width="48" height="36" rx="3"></rect>
		<text class="viz-callout" x="277" y="261" text-anchor="middle">decode</text>
		<path d="M301 257H314" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#asr-boundary-arrow)"></path>
		<rect class="viz-node viz-node--output" x="321" y="239" width="22" height="36" rx="3"></rect>
		<text class="viz-callout" x="332" y="261" text-anchor="middle">text</text>
		<rect class="viz-node viz-node--input" x="31" y="310" width="86" height="34" rx="3"></rect>
		<text class="viz-label" x="74" y="323" text-anchor="middle">audio + text</text>
		<text class="viz-label" x="74" y="339" text-anchor="middle">retrain core</text>
		<path d="M117 327H143V292" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.5;marker-end:url(#asr-boundary-arrow)"></path>
		<rect class="viz-node" x="160" y="310" width="173" height="34" rx="3" style="stroke-dasharray:4 3"></rect>
		<text class="viz-label" x="246.5" y="323" text-anchor="middle">LM / bias / endpointing</text>
		<text class="viz-label" x="246.5" y="339" text-anchor="middle">optional core controls</text>
		<path d="M277 310V280" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:4 3;marker-end:url(#asr-boundary-arrow)"></path>
		<text class="viz-label" x="18" y="355">End to end does not mean “no decoder” or “no external controls.”</text>
		<text class="viz-axis-label" x="18" y="386">ARCHITECTURE FOLLOWS THE CONSTRAINT</text>
		<path d="M36 419H324" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
		<path d="M36 413V425M180 413V425M324 413V425" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
		<text class="viz-callout" x="36" y="407">explicit control</text>
		<text class="viz-callout" x="324" y="407" text-anchor="end">joint optimization</text>
		<text class="viz-label" x="36" y="442">text/pronunciation update</text>
		<text class="viz-label" x="180" y="459" text-anchor="middle">valid blend</text>
		<text class="viz-label" x="324" y="442" text-anchor="end">core adaptation</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> trace text and pronunciation into decoding, not acoustic weights. A jointly trained core can also retain external controls; compatible updates still require validation. <span class="visual-reader-check" data-reader-check="asr-adaptation-boundaries"><strong>Check:</strong> For the marked vocabulary change, which knowledge source can change without retraining the acoustic core?</span></figcaption>
</figure>

## Calculate the Changed Preference

| Fusion weight | Rare name | Common word | Selected candidate |
| --- | --- | --- | --- |
| $\lambda=0.5$ | $-5-0.5=-5.5$ | $-2-2=-4$ | Common |
| $\lambda=1.5$ | $-5-1.5=-6.5$ | $-2-6=-8$ | Rare |

The acoustic evidence and parameters stayed fixed.
Only the scoring preference changed; this is not measured accuracy improvement or proof the rare name was spoken.
If that candidate never enters a fixed n-best list, rescoring cannot select it.

<span id="choosing-between-them"></span>

## Decision: Version the Whole Update

Specify acoustic/token units, compatible phone/token inventory, lexicon, LM score convention, fusion weight, search and artifact versions.
Graph rebuilding, publication, regression tests and compatible rollback can remain necessary when acoustic weights do not change.
Compare rare-term recall, false insertions, WER slices, delay, maintenance and update time under matched data and workloads.

## Change 1: Only Text and Pronunciations Are Available

**New task.** Add domain terms without paired-audio retraining data.

**Picture and change.** Test compatible lexicon, LM or contextual-bias updates with candidate-support and negative checks.

**Check.** New-term recall must not hide unrelated false insertions.

**Use when.** Vocabulary changes faster than labeled audio.

## Change 2: The Microphone Changed

**New task.** Errors follow channel conditions, not new vocabulary.

**Picture and change.** Inspect front-end/acoustic coverage and collect representative adaptation and evaluation audio.

**Check.** A stronger term bias cannot restore missing acoustic information.

**Use when.** Failure tracks noise, channel or device.

## Failure: Fixed Acoustic Weights Are Called a Free Update

An incompatible graph, missing candidate or untested bias can break delivery despite unchanged acoustic weights.

## Transfer Check

Is an end-to-end acoustic core with external LM rescoring a contradiction?

### Reasoned Answer

No. Joint core training and modular inference controls describe different boundaries.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Hybrid ASR

Acoustic scores combine with HMM topology, pronunciation lexicon and LM, often through WFST decoding.
Explicit modules enable local diagnosis/control but cost alignment, graph and component maintenance.
Separate objectives do not automatically optimize final transcript quality jointly.

## End-to-end ASR

CTC, RNN-T and attention objectives learn audio-to-token mappings with different alignment/history assumptions.
Joint representations can reduce some boundaries; external fusion, biasing, endpointing, confidence and safety controls can remain.
Data needs, multilingual quality, hallucination and diagnosis depend on the actual system, not its label.

## In an interview

1. Name whether acoustics, pronunciation, frequency or endpoint policy changes.
2. Locate the update in both decompositions.
3. Check labeled audio, domain text, control, languages and existing infrastructure.
4. Validate streaming, rare terms, false biasing and update rollback.
5. Compare matched beyond-WER outcomes and operating cost.

## Common confusions

Neither "hybrid always needs less data" nor "end-to-end has no decoder" is a reliable selection rule.
Accuracy and adaptation speed require workload-specific evidence.

## Sources

- [Kaldi HCLG construction](https://kaldi-asr.org/doc/graph.html).
- [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211).
- [RNN-T](https://arxiv.org/abs/1211.3711).

*Related: [CTC](/concepts/connectionist-temporal-classification/), [RNN-T](/concepts/rnn-transducer/), [encoder-decoder architectures](/concepts/encoder-decoder-architectures/), and [streaming ASR](/concepts/streaming-asr/).*

</div>
