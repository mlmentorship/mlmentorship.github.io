---
title: "Streaming automatic speech recognition"
description: "Emit transcripts with bounded latency using chunked encoders, monotonic alignment, endpointing, and stability-aware evaluation."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Streaming ASR is the difference between a voice assistant that feels responsive and one that feels broken: it has to emit transcript while the user is still speaking, with bounded latency. The catch is that the model cannot attend to unlimited future context, so it trades recognition quality against emission latency, compute, and transcript stability. Offline ASR has none of those constraints, which is why a system that scores well offline can still feel terrible live.

**Learning objective:** trace why a token can appear before speech ends while remaining revisable, and distinguish first-token latency from finalization latency.

<!-- visual:streaming-asr-provisional-to-final -->
<figure class="learning-figure plot-panel" aria-labelledby="streaming-asr-timeline-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="streaming-asr-timeline-title">How can streaming ASR respond early without committing early?</p>
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
		<text class="viz-node-value" x="279" y="55">pause · endpoint wait</text>
		<path d="M156 68V326" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:4 4"></path>
		<path d="M225 68V326" style="fill:none;stroke:var(--viz-edge);stroke-width:1.2;stroke-dasharray:4 4"></path>
		<path d="M326 68V326" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.5;stroke-dasharray:4 4"></path>
		<text class="viz-axis-label" x="12" y="102">CHECKPOINT 1 · EARLY PARTIAL</text>
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
		<text class="viz-axis-label" x="12" y="301">FINAL RESULT</text>
		<rect class="viz-node viz-node--output" x="44" y="309" width="232" height="32" rx="4"></rect>
		<rect x="48" y="313" width="224" height="24" rx="2" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1"></rect>
		<text class="viz-node-label" x="160" y="330">play their song</text>
		<text class="viz-callout" x="286" y="330">FINAL</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> the first partial result appears once a chunk and its bounded right context are available, so <strong>time to first token</strong> can end while speech is still arriving. That text remains provisional: later audio changes <code>the</code> to <code>their</code>. <strong>Finalization latency</strong> ends only after the pause triggers endpointing and remaining processing completes. Measure both milestones and the revision between them. Original schematic informed by <a href="https://arxiv.org/abs/1811.06621">He et al. (2019)</a>, <a href="https://www.isca-archive.org/interspeech_2020/shangguan20_interspeech.html">Shangguan et al. (2020)</a>, and <a href="https://research.google/pubs/towards-fast-and-accurate-streaming-end-to-end-asr/">Li et al. (2020)</a>.</figcaption>
</figure>

## Architecture choices

- **Chunked encoder:** processes bounded windows with cached state or limited lookahead.
- **RNN-T / transducer:** learns a monotonic alignment and emits tokens incrementally.
- **CTC:** a simple monotonic objective, often paired with streaming beam search and a language model.
- **Streaming attention:** constrains or chunks encoder-decoder attention so it does not need the whole utterance.

## Latency components

End-to-end latency is more than model runtime: audio chunk size, feature extraction, model compute, lookahead, decoding, endpointing, network, and client rendering all add up. Reporting model runtime alone hides the delay the user actually feels.

## Endpointing and partial stability

The system has to decide when speech has ended. Aggressive endpointing lowers latency but clips pauses; conservative endpointing feels sluggish. Partial hypotheses may also revise earlier words, so measure flicker or edit overhead alongside final word error rate.

## Evaluation

- Final and streaming word error rate
- Time to first token and finalization latency
- Partial stability / revision rate
- Real-time factor and peak compute
- Slice quality by accent, noise, device, language, and speech rate

## In an interview

1. Clarify the latency and quality targets.
2. Choose a monotonic or chunked architecture.
3. Account for every latency component, not just the model.
4. Discuss endpointing and partial-transcript stability.
5. Design fallback, offline rescoring, and slice monitoring.

## Common confusions

- **"Streaming means batch size one."** It means bounded future context and incremental output; you can still batch across streams.
- **"WER captures latency."** It says nothing about when words appear or how often they change.
- **"More lookahead is free quality."** Every frame of lookahead is user-visible latency.

*Related: [automatic speech recognition](/concepts/automatic-speech-recognition/), [RNN-T](/concepts/rnn-transducer/), and [CTC](/concepts/connectionist-temporal-classification/).*
