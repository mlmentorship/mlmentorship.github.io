---
title: "RNN-Transducer (RNN-T)"
description: "The streaming-ASR workhorse. RNN-T fixes CTC's biggest weakness (its frame-independence assumption) by adding a prediction network that conditions on previously emitted tokens, while staying naturally streamable."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

RNN-T extends CTC with a **prediction network** (an internal language model over emitted tokens) and a **joint network** that combines acoustic and label context, producing a streamable model that marginalizes over alignments while conditioning each output on the tokens already produced.

RNN-T is the model behind most **on-device, streaming** speech recognition (Google's Gboard/Assistant dictation, many production ASR systems). It is the natural "what fixes CTC?" follow-up in any speech interview.

The key selling points:

- **Streaming by construction.** Unlike attention encoder-decoder models, RNN-T emits tokens left-to-right as audio arrives, with bounded latency.
- **No frame-independence assumption.** The prediction network conditions on output history, so RNN-T has a built-in language model, the thing CTC lacks.
- **Still alignment-free.** Like CTC, it marginalizes over all valid alignments during training.

## The three components

| Component | Input | Role |
| --- | --- | --- |
| **Encoder (transcription net)** | Acoustic frames $x_{1:t}$ | Acoustic representation $f_t$ (the "audio" tower) |
| **Prediction net** | Previously emitted non-blank tokens $y_{1:u-1}$ | Label-history representation $g_u$ (an internal LM) |
| **Joint net** | $f_t, g_u$ | Combine → distribution over $V \cup \{\varnothing\}$ |

The joint network is typically a small feed-forward net:

$$
h_{t,u} = \psi(W_f f_t + W_g g_u + b), \qquad p(k \mid t, u) = \mathrm{softmax}(W_h h_{t,u}).
$$

## The output lattice and blank

RNN-T defines a 2D grid indexed by acoustic frame $t$ and label position $u$. At each node it predicts either:

- a **real token** → move "up" ($u \to u+1$), staying on the same frame, or
- the **blank** $\varnothing$ → move "right" ($t \to t+1$), advancing the audio.

A path from bottom-left to top-right is one alignment. The training loss sums over **all** monotonic paths through this lattice:

<!-- visual:rnnt-two-axis-alignment-lattice -->
<figure class="learning-figure" aria-labelledby="rnnt-lattice-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="rnnt-lattice-title">How can different RNN-T paths emit the same transcript?</p>
	<div class="visual-grid--two" role="group" aria-label="Two RNN-T lattice paths that both emit A then B">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 245" role="img" aria-labelledby="rnnt-early-title rnnt-early-desc">
				<title id="rnnt-early-title">Alignment one emits A before advancing acoustic time</title>
				<desc id="rnnt-early-desc">Starting at time zero and output position zero, the path moves up to emit A, right twice on blanks, up to emit B, and right once on a blank. It ends at time three and output position two.</desc>
				<rect class="viz-plot-bg" x="26" y="30" width="248" height="188" rx="5"></rect>
				<text class="viz-axis-label" x="150" y="18" text-anchor="middle">PATH 1 · EMIT A EARLY</text>
				<path d="M45 196H255M45 121H255M45 46H255M45 46V196M115 46V196M185 46V196M255 46V196" style="fill:none;stroke:var(--c-rule);stroke-width:1"></path>
				<text class="viz-axis-label" x="45" y="232" text-anchor="middle">t=0</text>
				<text class="viz-axis-label" x="115" y="232" text-anchor="middle">t=1</text>
				<text class="viz-axis-label" x="185" y="232" text-anchor="middle">t=2</text>
				<text class="viz-axis-label" x="255" y="232" text-anchor="middle">t=3</text>
				<text class="viz-axis-label" x="17" y="200" text-anchor="middle">u=0</text>
				<text class="viz-axis-label" x="17" y="125" text-anchor="middle">u=1</text>
				<text class="viz-axis-label" x="17" y="50" text-anchor="middle">u=2</text>
				<path d="M45 196V121H185V46H255" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:4;stroke-linejoin:round"></path>
				<path d="M41 130L45 121L49 130M106 117L115 121L106 125M176 117L185 121L176 125M181 55L185 46L189 55M246 42L255 46L246 50" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
				<text class="viz-node-label" x="55" y="161">A</text>
				<text class="viz-node-label" x="80" y="111">∅</text>
				<text class="viz-node-label" x="150" y="111">∅</text>
				<text class="viz-node-label" x="195" y="86">B</text>
				<text class="viz-node-label" x="220" y="36">∅</text>
				<circle cx="45" cy="196" r="5" style="fill:var(--viz-surface);stroke:var(--viz-edge);stroke-width:2"></circle>
				<circle cx="255" cy="46" r="5" style="fill:var(--viz-output-bg);stroke:var(--viz-edge);stroke-width:2"></circle>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 245" role="img" aria-labelledby="rnnt-late-title rnnt-late-desc">
				<title id="rnnt-late-title">Alignment two advances acoustic time before emitting A</title>
				<desc id="rnnt-late-desc">Starting at time zero and output position zero, the path moves right on a blank, up to emit A, right on a blank, up to emit B, and right once more on a blank. It reaches the same endpoint and emits the same A B transcript as alignment one.</desc>
				<rect class="viz-plot-bg" x="26" y="30" width="248" height="188" rx="5"></rect>
				<text class="viz-axis-label" x="150" y="18" text-anchor="middle">PATH 2 · EMIT A LATER</text>
				<path d="M45 196H255M45 121H255M45 46H255M45 46V196M115 46V196M185 46V196M255 46V196" style="fill:none;stroke:var(--c-rule);stroke-width:1"></path>
				<text class="viz-axis-label" x="45" y="232" text-anchor="middle">t=0</text>
				<text class="viz-axis-label" x="115" y="232" text-anchor="middle">t=1</text>
				<text class="viz-axis-label" x="185" y="232" text-anchor="middle">t=2</text>
				<text class="viz-axis-label" x="255" y="232" text-anchor="middle">t=3</text>
				<text class="viz-axis-label" x="17" y="200" text-anchor="middle">u=0</text>
				<text class="viz-axis-label" x="17" y="125" text-anchor="middle">u=1</text>
				<text class="viz-axis-label" x="17" y="50" text-anchor="middle">u=2</text>
				<path d="M45 196H115V121H185V46H255" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:4;stroke-dasharray:8 4;stroke-linejoin:round"></path>
				<path d="M106 192L115 196L106 200M111 130L115 121L119 130M176 117L185 121L176 125M181 55L185 46L189 55M246 42L255 46L246 50" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
				<text class="viz-node-label" x="80" y="186">∅</text>
				<text class="viz-node-label" x="125" y="161">A</text>
				<text class="viz-node-label" x="150" y="111">∅</text>
				<text class="viz-node-label" x="195" y="86">B</text>
				<text class="viz-node-label" x="220" y="36">∅</text>
				<circle cx="45" cy="196" r="5" style="fill:var(--viz-surface);stroke:var(--viz-edge);stroke-width:2"></circle>
				<circle cx="255" cy="46" r="5" style="fill:var(--viz-output-bg);stroke:var(--viz-edge);stroke-width:2"></circle>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> move right only on <strong>blank ∅</strong> to consume an acoustic frame; move up only on a <strong>token</strong> to extend the transcript without consuming time. Both labelled paths end at (t=3, u=2) and emit <code>A B</code>, so training adds both path probabilities rather than choosing one alignment. This original schematic was checked against <a href="https://arxiv.org/abs/1211.3711">Graves (2012)</a> and <a href="https://arxiv.org/abs/1811.06621">He et al. (2019)</a>.</figcaption>
</figure>

$$
p(\mathbf{y} \mid X) = \sum_{\text{paths}} \prod p(\cdot),
$$

computed exactly with a forward-backward DP over the $T \times U$ lattice (cost $O(T \cdot U)$, heavier than CTC's $O(T)$ and notoriously memory-hungry, which is why fused/streaming RNN-T loss kernels exist).

## RNN-T vs CTC vs attention seq2seq

| | Internal LM? | Streamable? | Alignment | Training cost |
| --- | --- | --- | --- | --- |
| **CTC** | No (frame-independent) | Yes | Monotonic, marginalized | $O(T)$ |
| **RNN-T** | Yes (prediction net) | Yes | Monotonic, marginalized | $O(T \cdot U)$ |
| **Attention enc-dec (LAS / Whisper)** | Yes (decoder) | Hard (needs full utterance) | Soft, learned attention | $O(T \cdot U)$ |

The mental model: **RNN-T = CTC + an internal language model, at the cost of a 2D alignment lattice.** Attention models are more accurate offline but stream poorly; RNN-T is the streaming sweet spot.

## Practical notes

- The encoder is usually the largest component; modern systems use **Conformer** (convolution-augmented transformer) encoders.
- The prediction network can be surprisingly small: even **stateless** (one-token context) variants work well, because the joint acoustic+text signal carries most of the information.
- RNN-T loss is memory-heavy; production training uses **function-merging / pruned RNN-T** losses to fit the $T \times U \times |V|$ tensor.
- Latency is tuned by limiting the encoder's right-context (how far into the future it looks).

## What an interviewer expects you to say

1. State that RNN-T fixes CTC's **conditional-independence** weakness with a **prediction network** conditioned on emitted tokens.
2. Name the **three nets**: encoder, prediction, joint.
3. Describe the **blank = advance time, token = advance label** lattice and that training marginalizes over all monotonic paths.
4. Explain *why it streams*: outputs are produced left-to-right with bounded right-context, unlike global attention.
5. Bonus: mention Conformer encoders, pruned/streaming RNN-T loss, and the accuracy-vs-latency tradeoff against attention models like Whisper.

## Common confusions

- **"The prediction network sees audio."** It does not. It only sees previously emitted *labels*; it is a language model. The joint net fuses it with the encoder's audio features.
- **"RNN-T is just CTC with an LSTM."** The architectural difference is the label-conditioned prediction net and the 2D lattice; that's what removes frame independence.
- **"Attention models can't beat RNN-T."** Offline, full-context attention models (Whisper) are typically more accurate. RNN-T wins on streaming latency and on-device deployment.
- **"Blank means silence."** As in CTC, blank is a structural token meaning "advance to the next frame," not acoustic silence.

---

*Related: [Connectionist Temporal Classification (CTC)](/concepts/connectionist-temporal-classification/), [Automatic speech recognition](/concepts/automatic-speech-recognition/), [Transformer architecture](/concepts/transformer-architecture/), [LSTM and GRU](/concepts/lstm-and-gru/).*
