---
title: "Automatic speech recognition (ASR)"
description: "Connect audio timing to transcript alignment, compare CTC/RNN-T/attention context, and evaluate text errors separately from delivery delay."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Audio frames arrive on a clock; transcript labels lack supplied frame alignment. Choose alignment, output history and available audio context separately.

## Problem

One second contains 16,000 samples.
Use uncentered 400-sample windows, a 160-sample hop and no padding.
How many complete frames exist?

Separately, a reference says "turn left" while a hypothesis says "please turn right now".
Compute word error rate, then choose an architecture for captions that must appear before speech ends.

## Calculate the Feature Clock

At this sample rate, the window is 25 milliseconds and the hop is ten.
The exact full-window count is:

$$
1+\left\lfloor\frac{16000-400}{160}\right\rfloor=98.
$$

The last frame starts at sample 15,520 and ends before sample 15,920 under half-open indexing.
The approximate 100-hops/second rate is not this clip's frame count.
Record padding, centering, normalization and encoder subsampling before interpreting output lengths.

## Picture: What Advances When a Label Is Emitted?

<!-- visual:asr-alignment-decoder-choices -->
<figure class="learning-figure plot-panel" aria-labelledby="asr-alignment-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="asr-alignment-title">What advances when each ASR model emits a token?</p>
	<svg viewBox="0 0 360 500" role="img" aria-labelledby="asr-alignment-svg-title asr-alignment-svg-desc">
		<title id="asr-alignment-svg-title">Comparison of CTC, RNN-T, and attention alignment paths</title>
		<desc id="asr-alignment-svg-desc">Three illustrative panels produce CAT. CTC uses six frames, one label-or-blank decision per frame and collapse without explicit emitted-label conditioning. The RNN-T schematic shows five blank time moves and three token moves with prefix conditioning. Offline attention uses prior tokens and weighted context over six available encoded frames. These paths are schematic, not outputs of a trained model on the worked waveform.</desc>
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
		<text class="viz-label" x="20" y="118">frame path</text>
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
		<text class="viz-label" x="14" y="257" transform="rotate(-90 14 257)">output position u →</text>
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
	<figcaption><strong>Read it this way:</strong> compare frame, time/token and attention moves. CTC lacks explicit emitted-label conditioning; RNN-T and autoregressive attention use a prefix. Streaming still depends on available encoder context.</figcaption>
</figure>

| Family | Alignment and output history | Streaming condition |
| --- | --- | --- |
| CTC | Frame-label/blank paths summed; no explicit autoregressive label prefix | Streaming-compatible features and encoder |
| RNN-T | Blank advances time; token advances output with prediction-network prefix | Bounded-context encoder, state and decoding |
| Attention encoder-decoder | Prefix plus attended audio; ordinary offline version sees encoded utterance | Changed context/chunk policy and measured delivery |

CTC's factorization does not prevent its encoder learning linguistic regularities.
An output-history network is also not proof of complete language knowledge.

## Evaluation

Align the worked text under a fixed normalization/reference policy:

| Reference | Hypothesis | Edit |
| --- | --- | --- |
| - | please | Insertion |
| turn | turn | Match |
| left | right | Substitution |
| - | now | Insertion |

Word error rate divides edits by reference words:

$$
\text{WER} = \frac{S + D + I}{N},
$$

Here $S=1$, $D=0$, $I=2$ and $N=2$, giving $3/2=150\%$.
Insertions allow WER above 100%.
Specify empty-reference handling; ordinary WER has a zero denominator there.
Use CER when character units fit, and report critical entities alongside aggregate errors.

## Change 1: Captions Must Arrive During Speech

**New task.** Deliver stable text before the utterance ends.

**Picture and change.** Bound future context, preserve streaming state and measure token/finalization delay from named audio events.

**Check.** CTC or RNN-T loss alone cannot make a full-utterance encoder stream.
Chunking an offline model changes quality and boundary behavior.

**Use when.** Live delivery is required.

## Change 2: Rare Names Drive Failures

**New task.** Domain entities matter more than average WER.

**Picture and change.** Check acoustic/token coverage and contextual bias/fusion with held-out terms and false-insertion tests.

**Check.** A stronger LM can prefer plausible words unsupported by audio.

**Use when.** Domain terms determine usefulness.

## Failure: Fluent Text Is Called Acoustically Verified

A better language score can mask wrong words.
Report audio grounding, text error, stability, delay and resources separately.

## Transfer Check

Can rescoring a fixed n-best list recover a transcript absent from that list?

### Reasoned Answer

No. Candidate generation or the search must change; rescoring only reorders existing candidates.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## The pipeline

```text
waveform -> features -> encoder -> decoder/LM -> transcript
```

### 1. Audio features

Log-mel features apply short-time Fourier analysis, mel filtering and logarithms.
MFCCs add a discrete cosine transform to log-mel energies.
Learned waveform front ends are another option; specify sample rate, feature shape and time stride rather than assume universality.
SpecAugment masks time/frequency, optionally warps time; noise mixing and speed perturbation require task-valid labels.
Self-supervised wav2vec/HuBERT-style pretraining offers another data path, not guaranteed gains.

### 2. Acoustic / sequence model

Transformer/Conformer encoders can have different future-context policies.
LAS/Whisper exemplify attention-based decoding; inspect the particular model's context and deployment wrapper.
Link [CTC](/concepts/connectionist-temporal-classification/) and [RNN-T](/concepts/rnn-transducer/) for objective derivations.
Declare vocabulary, blank/EOS conventions and true post-subsampling lengths.

### 3. Language model fusion

| Route | What changes |
| --- | --- |
| Shallow fusion | Add $\lambda\log p_{LM}(\mathbf y)$ during beam search under a specified score convention |
| Deep/cold fusion | Integrate LM hidden state into decoding |
| Rescoring | Reorder an existing n-best list/lattice; missing candidates stay missing |

External LMs are optional; compare entity gains, false insertions, costs and acoustic support.

<span id="the-historical-contrast-why-end-to-end-won"></span>

## Historical Decomposition

Hybrid systems combine lexicon/pronunciation, HMM state transitions, GMM/DNN acoustic estimates and language scores, often using alignment training.
End-to-end objectives reduce some component boundaries, not every lexicon/LM or decoder choice.
Neither family wins every data, resource, customization and latency setting.

<span id="what-an-interviewer-expects-you-to-say"></span>

## Delivery Checklist

1. State waveform, feature and subsampling units.
2. Trace alignment, label history and audio context.
3. Specify fusion/search and missing-candidate limits.
4. Fix reference normalization, slices, WER/CER and entity checks.
5. Measure onset/word-end/utterance-end delay, revision stability, resources and fallback.

## Common confusions

Loss factorization, encoder context, linguistic evidence and runtime delivery are distinct contracts.

---

*Related: [Connectionist Temporal Classification (CTC)](/concepts/connectionist-temporal-classification/), [RNN-Transducer (RNN-T)](/concepts/rnn-transducer/), [Transformer architecture](/concepts/transformer-architecture/), [Tokenization](/concepts/tokenization/).*

</div>
