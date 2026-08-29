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
