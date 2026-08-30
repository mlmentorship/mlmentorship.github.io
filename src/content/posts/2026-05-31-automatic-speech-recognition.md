---
title: "Automatic speech recognition (ASR)"
description: "The end-to-end map from a waveform to text: features, the three modeling paradigms (CTC, RNN-T, attention), language-model fusion, and how the field moved from HMM-GMM pipelines to a single neural model."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

ASR maps an audio waveform to a text transcript. Modern systems are **end-to-end neural models** (CTC, RNN-T, or attention encoder-decoders) trained directly on (audio, text) pairs, replacing the old multi-stage HMM-GMM + pronunciation-lexicon + n-gram pipeline.

ASR is the canonical "sequence in, sequence out, unknown alignment" problem, and it pulls together several interview-favorite ideas: feature extraction, alignment-free losses, language-model fusion, and streaming-vs-accuracy tradeoffs. It's also a frequent applied-scientist domain (voice assistants, captioning, call-center analytics, medical scribing).

## The pipeline

```text
waveform → features → acoustic model → (decoder + LM) → text
```

### 1. Audio features

Raw audio is ~16 kHz samples. Models rarely consume raw samples directly; they use **frames**:

- Window the signal (e.g. 25 ms windows, 10 ms hop → 100 frames/sec).
- Compute a **log-mel spectrogram**: short-time Fourier transform → mel filterbank → log. This mimics human pitch perception and is the de-facto standard input.
- **MFCCs** (mel-frequency cepstral coefficients) add a DCT on top; common in classical/HMM systems, less needed for deep nets which prefer raw log-mel.

**Augmentation:** **SpecAugment** applies time and frequency masking, with optional time warping, to the spectrogram. Speed perturbation and noise mixing are also common ASR augmentations.

### 2. Acoustic / sequence model

The three end-to-end paradigms:

| Paradigm | Idea | Streams? | Built-in LM? |
| --- | --- | --- | --- |
| **CTC** | Frame classifier + blank, marginalize alignments | Yes | No |
| **RNN-T** | CTC + label-conditioned prediction net | Yes | Yes |
| **Attention enc-dec** (LAS, Whisper) | Decoder attends over encoded audio | Hard | Yes |

<!-- visual:asr-alignment-decoder-choices -->
<figure class="learning-figure plot-panel" aria-labelledby="asr-alignment-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="asr-alignment-title">What advances when each ASR model emits a token?</p>
	<svg viewBox="0 0 360 500" role="img" aria-labelledby="asr-alignment-svg-title asr-alignment-svg-desc">
		<title id="asr-alignment-svg-title">Comparison of CTC, RNN-T, and attention alignment paths</title>
		<desc id="asr-alignment-svg-desc">Three stacked panels map six encoded audio frames to the transcript CAT. CTC makes exactly one label-or-blank decision at each frame and collapses the path without using output history. RNN-T follows a two-dimensional path: blank steps move forward in audio time, while token steps move upward in output position, with a prediction network that sees the emitted prefix. Standard offline attention emits tokens autoregressively; each output uses prior tokens and a weighted context over the encoded frames.</desc>
		<defs><marker id="arrow-forward" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
		<rect class="viz-plot-bg" x="8" y="8" width="344" height="142" rx="5"></rect>
		<text class="viz-axis-label" x="20" y="28">CTC · ONE DECISION PER FRAME</text>
		<text class="viz-label" x="20" y="47">encoded frames</text>
		<path class="viz-forward" d="M42 67H318"></path>
		<g>
			<rect class="viz-node viz-node--input" x="38" y="55" width="36" height="25" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="88" y="55" width="36" height="25" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="138" y="55" width="36" height="25" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="188" y="55" width="36" height="25" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="238" y="55" width="36" height="25" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="288" y="55" width="36" height="25" rx="2"></rect>
			<text class="viz-callout" x="56" y="72" text-anchor="middle">1</text>
			<text class="viz-callout" x="106" y="72" text-anchor="middle">2</text>
			<text class="viz-callout" x="156" y="72" text-anchor="middle">3</text>
			<text class="viz-callout" x="206" y="72" text-anchor="middle">4</text>
			<text class="viz-callout" x="256" y="72" text-anchor="middle">5</text>
			<text class="viz-callout" x="306" y="72" text-anchor="middle">6</text>
		</g>
		<text class="viz-label" x="20" y="100">frame path</text>
		<text class="viz-callout" x="56" y="100" text-anchor="middle">blank</text>
		<text class="viz-callout" x="106" y="100" text-anchor="middle">C</text>
		<text class="viz-callout" x="156" y="100" text-anchor="middle">blank</text>
		<text class="viz-callout" x="206" y="100" text-anchor="middle">A</text>
		<text class="viz-callout" x="256" y="100" text-anchor="middle">T</text>
		<text class="viz-callout" x="306" y="100" text-anchor="middle">blank</text>
		<path class="viz-forward" d="M180 106V119"></path>
		<rect class="viz-node viz-node--output" x="137" y="120" width="86" height="22" rx="3"></rect>
		<text class="viz-callout" x="180" y="135" text-anchor="middle">collapse → CAT</text>
		<text class="viz-label" x="232" y="135">no output history</text>
		<rect class="viz-plot-bg" x="8" y="158" width="344" height="176" rx="5"></rect>
		<text class="viz-axis-label" x="20" y="178">RNN-T · MOVE THROUGH TIME OR OUTPUT</text>
		<text class="viz-label" x="53" y="318">audio time t →</text>
		<text class="viz-label" x="18" y="257" transform="rotate(-90 18 257)">output position u →</text>
		<g style="fill:none;stroke:var(--c-rule);stroke-width:1">
			<path d="M54 202H254M54 234H254M54 266H254M54 298H254"></path>
			<path d="M54 202V298M94 202V298M134 202V298M174 202V298M214 202V298M254 202V298"></path>
		</g>
		<path d="M54 298H94V266H134H174V234H214V202H254" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:4;stroke-linecap:round;stroke-linejoin:round"></path>
		<g class="viz-callout">
			<text x="74" y="292" text-anchor="middle">blank</text>
			<text x="100" y="285">C</text>
			<text x="154" y="260" text-anchor="middle">blank</text>
			<text x="180" y="253">A</text>
			<text x="220" y="221">T</text>
			<text x="234" y="196" text-anchor="middle">blank</text>
		</g>
		<text class="viz-label" x="274" y="220">token step:</text>
		<text class="viz-callout" x="274" y="236">u + 1</text>
		<text class="viz-label" x="274" y="261">blank step:</text>
		<text class="viz-callout" x="274" y="277">t + 1</text>
		<text class="viz-label" x="274" y="302">prefix seen:</text>
		<text class="viz-callout" x="274" y="318">C, then CA</text>
		<rect class="viz-plot-bg" x="8" y="342" width="344" height="150" rx="5"></rect>
		<text class="viz-axis-label" x="20" y="362">OFFLINE ATTENTION · CHOOSE CONTEXT PER TOKEN</text>
		<text class="viz-label" x="20" y="382">all encoded frames available</text>
		<g>
			<rect class="viz-node viz-node--input" x="40" y="390" width="36" height="24" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="88" y="390" width="36" height="24" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="136" y="390" width="36" height="24" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="184" y="390" width="36" height="24" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="232" y="390" width="36" height="24" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="280" y="390" width="36" height="24" rx="2"></rect>
		</g>
		<g style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:4 3">
			<path d="M58 414L108 452M106 414L108 452M154 414L180 452M202 414L180 452M250 414L252 452M298 414L252 452"></path>
		</g>
		<rect class="viz-node viz-node--output" x="88" y="452" width="40" height="25" rx="3"></rect>
		<rect class="viz-node viz-node--output" x="160" y="452" width="40" height="25" rx="3"></rect>
		<rect class="viz-node viz-node--output" x="232" y="452" width="40" height="25" rx="3"></rect>
		<text class="viz-callout" x="108" y="469" text-anchor="middle">C</text>
		<text class="viz-callout" x="180" y="469" text-anchor="middle">A</text>
		<text class="viz-callout" x="252" y="469" text-anchor="middle">T</text>
		<path class="viz-forward" d="M128 464H157M200 464H229"></path>
		<text class="viz-label" x="340" y="459" text-anchor="end">prior tokens</text>
		<text class="viz-label" x="340" y="475" text-anchor="end">+ attended audio</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> CTC must make one label-or-blank choice at every frame, then collapse those choices; it does not condition on emitted labels. RNN-T can move right with a blank or emit upward at the same audio time, and its prediction network sees the prefix. Standard offline attention generates left to right from prior tokens while choosing a weighted context over the encoded utterance, which is why native streaming is hardest.</figcaption>
</figure>

Encoders are usually **Conformer** or transformer stacks over log-mel frames. Whisper is a plain transformer encoder-decoder trained on ~680k hours of weakly-supervised web audio.

### 3. Language model fusion

Acoustic models benefit from an external LM, especially CTC (which has no internal LM):

- **Shallow fusion**: add $\lambda \log p_{LM}(\mathbf{y})$ to the acoustic score during beam search.
- **Deep / cold fusion**: integrate LM hidden states into the decoder.
- Rescoring: generate an n-best list / lattice, then **re-score** with a large LM (often a transformer LM).

## The historical contrast (why end-to-end won)

The classical pipeline was **HMM-GMM** (later HMM-DNN): a pronunciation lexicon mapped words → phones, an HMM modeled phone-state transitions, a GMM/DNN modeled acoustics, and a separate n-gram LM handled language. It required **forced alignment** and lots of expert-built components.

End-to-end models collapse all of this into one network trained on (audio, text) pairs. They win on simplicity and, with enough data, on accuracy, though at the cost of needing more data and giving up some modularity.

## Evaluation

The primary metric is **Word Error Rate (WER)**:

$$
\text{WER} = \frac{S + D + I}{N},
$$

where $S, D, I$ are substitutions, deletions, insertions (via edit distance to the reference) and $N$ is the number of reference words. **Character Error Rate (CER)** is the analog for languages without clear word boundaries. Note WER can exceed 100% (insertions).

## What an interviewer expects you to say

1. Describe the pipeline: **log-mel features → neural encoder → decoder/LM → text**, and that SpecAugment is the key augmentation.
2. Compare the three paradigms (**CTC / RNN-T / attention**) on streaming, internal LM, and accuracy.
3. Explain **external LM fusion** (shallow fusion, rescoring) and why CTC needs it most.
4. Know **WER** and how it's computed.
5. Bonus: explain why the field moved from **HMM-GMM** to end-to-end, and when you'd still prefer a streaming RNN-T (on-device, low latency) over an offline attention model (Whisper, max accuracy).

## Common confusions

- **"Models eat raw waveforms."** Usually log-mel spectrogram frames; raw-waveform front-ends (wav2vec 2.0, SincNet) exist but features are still the norm.
- **"WER ≤ 100%."** False. Insertions can push it above 100%.
- **"Whisper streams."** It's an offline attention encoder-decoder; it needs (chunks of) the full utterance. Streaming use requires chunking hacks. RNN-T is the native streaming choice.
- **"Self-supervised pretraining is irrelevant."** wav2vec 2.0 / HuBERT-style self-supervised pretraining on unlabeled audio is now standard for low-resource ASR.

---

*Related: [Connectionist Temporal Classification (CTC)](/concepts/connectionist-temporal-classification/), [RNN-Transducer (RNN-T)](/concepts/rnn-transducer/), [Transformer architecture](/concepts/transformer-architecture/), [Tokenization](/concepts/tokenization/).*
