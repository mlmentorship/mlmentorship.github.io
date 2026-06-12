---
title: "RNN-Transducer (RNN-T)"
description: "The streaming-ASR workhorse. RNN-T fixes CTC's biggest weakness — its frame-independence assumption — by adding a prediction network that conditions on previously emitted tokens, while staying naturally streamable."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

RNN-T extends CTC with a **prediction network** (an internal language model over emitted tokens) and a **joint network** that combines acoustic and label context, producing a streamable model that marginalizes over alignments while conditioning each output on the tokens already produced.

## Why it matters

RNN-T is the model behind most **on-device, streaming** speech recognition (Google's Gboard/Assistant dictation, many production ASR systems). It is the natural "what fixes CTC?" follow-up in any speech interview.

The key selling points:

- **Streaming by construction.** Unlike attention encoder-decoder models, RNN-T emits tokens left-to-right as audio arrives, with bounded latency.
- **No frame-independence assumption.** The prediction network conditions on output history, so RNN-T has a built-in language model — the thing CTC lacks.
- **Still alignment-free.** Like CTC, it marginalizes over all valid alignments during training.

## The three components

| Component | Input | Role |
|-----------|-------|------|
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

$$
p(\mathbf{y} \mid X) = \sum_{\text{paths}} \prod p(\cdot),
$$

computed exactly with a forward-backward DP over the $T \times U$ lattice (cost $O(T \cdot U)$ — heavier than CTC's $O(T)$ and notoriously memory-hungry, which is why fused/streaming RNN-T loss kernels exist).

## RNN-T vs CTC vs attention seq2seq

| | Internal LM? | Streamable? | Alignment | Training cost |
|---|---|---|---|---|
| **CTC** | No (frame-independent) | Yes | Monotonic, marginalized | $O(T)$ |
| **RNN-T** | Yes (prediction net) | Yes | Monotonic, marginalized | $O(T \cdot U)$ |
| **Attention enc-dec (LAS / Whisper)** | Yes (decoder) | Hard (needs full utterance) | Soft, learned attention | $O(T \cdot U)$ |

The mental model: **RNN-T = CTC + an internal language model, at the cost of a 2D alignment lattice.** Attention models are more accurate offline but stream poorly; RNN-T is the streaming sweet spot.

## Practical notes

- The encoder is usually the largest component; modern systems use **Conformer** (convolution-augmented transformer) encoders.
- The prediction network can be surprisingly small — even **stateless** (one-token context) variants work well, because the joint acoustic+text signal carries most of the information.
- RNN-T loss is memory-heavy; production training uses **function-merging / pruned RNN-T** losses to fit the $T \times U \times |V|$ tensor.
- Latency is tuned by limiting the encoder's right-context (how far into the future it looks).

## What an interviewer expects you to say

1. State that RNN-T fixes CTC's **conditional-independence** weakness with a **prediction network** conditioned on emitted tokens.
2. Name the **three nets**: encoder, prediction, joint.
3. Describe the **blank = advance time, token = advance label** lattice and that training marginalizes over all monotonic paths.
4. Explain *why it streams*: outputs are produced left-to-right with bounded right-context, unlike global attention.
5. Bonus: mention Conformer encoders, pruned/streaming RNN-T loss, and the accuracy-vs-latency tradeoff against attention models like Whisper.

## Common confusions

- **"The prediction network sees audio."** It does not. It only sees previously emitted *labels* — it is a language model. The joint net fuses it with the encoder's audio features.
- **"RNN-T is just CTC with an LSTM."** The architectural difference is the label-conditioned prediction net and the 2D lattice; that's what removes frame independence.
- **"Attention models can't beat RNN-T."** Offline, full-context attention models (Whisper) are typically more accurate. RNN-T wins on streaming latency and on-device deployment.
- **"Blank means silence."** As in CTC, blank is a structural token meaning "advance to the next frame," not acoustic silence.

---

*Related: [Connectionist Temporal Classification (CTC)](/concepts/connectionist-temporal-classification/), [Automatic speech recognition](/concepts/automatic-speech-recognition/), [Transformer architecture](/concepts/transformer-architecture/), [LSTM and GRU](/concepts/lstm-and-gru/).*
