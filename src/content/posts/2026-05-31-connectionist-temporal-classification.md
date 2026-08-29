---
title: "Connectionist Temporal Classification (CTC)"
description: "How you train a sequence model to map audio (or pixels) to text without knowing the alignment. CTC marginalizes over every possible alignment with a blank symbol and a forward-backward sum."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

CTC is a loss function that trains a frame-level classifier to output a shorter label sequence **without requiring a frame-to-label alignment**, by introducing a special **blank** symbol and summing the probability of *all* alignments that collapse to the target.

The canonical interview topic for speech, handwriting, and any **monotonic, unaligned** sequence-to-sequence task. It answers the question every ASR interviewer eventually asks: *"You have 1000 audio frames and a 5-word transcript. How do you train without per-frame labels?"*

CTC matters because:

- It removes the need for a separate alignment model (the old HMM-GMM pipeline forced-aligned audio to phones first).
- It is the foundation that RNN-T and many streaming ASR systems build on or contrast against.
- The forward-backward dynamic program is the same idea as the HMM forward-backward algorithm, a clean way to show you understand marginalization over latent structure.

## The setup

The network emits, for each of $T$ input frames, a probability distribution over the vocabulary $V$ plus a blank token $\varnothing$:

$$
y_t \in \Delta^{|V|+1}, \quad t = 1 \dots T.
$$

A **path** (or alignment) $\pi$ is one label per frame, e.g. for target `CAT`:

```text
C C ∅ A ∅ T T   →  collapse  →  CAT
∅ C A A ∅ T ∅   →  collapse  →  CAT
```

The **collapse function** $\mathcal{B}$ does two things, in order:

1. Merge consecutive repeated labels.
2. Remove all blanks.

The blank is what lets the model emit the same letter twice (`L L` in `HELLO`): insert a blank between them, `L ∅ L`, and the merge step won't collapse them.

## The loss

The probability of a target sequence $\mathbf{l}$ is the sum over every path that collapses to it:

$$
p(\mathbf{l} \mid X) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} \prod_{t=1}^{T} y_t^{\pi_t}.
$$

The CTC loss is $-\log p(\mathbf{l} \mid X)$. The number of valid paths is exponential in $T$, so the sum is computed with a **forward-backward dynamic program** over an augmented label sequence $\mathbf{l}'$ (the target with a blank inserted before, after, and between every label).

Define $\alpha_t(s)$ = total probability of all paths ending in symbol $\mathbf{l}'_s$ at frame $t$:

$$
\alpha_t(s) = \big(\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \alpha_{t-1}(s-2)\big)\, y_t^{\mathbf{l}'_s},
$$

where the $s-2$ term is only allowed when moving between two distinct non-blank labels (it skips a blank). The total is $\alpha_T$ summed over the final two states. Gradients flow through this DP via the backward variables $\beta$, giving an exact $O(T \cdot |\mathbf{l}|)$ gradient.

## The conditional-independence assumption

CTC factorizes $p(\mathbf{l} \mid X) = \sum_\pi \prod_t y_t^{\pi_t}$: each frame's output depends only on $X$, **not on previously emitted labels**. There is no internal language model. This is CTC's defining limitation and the main reason RNN-T exists.

Practically, CTC ASR systems are decoded with an **external language model** (shallow fusion / beam search with a KenLM or neural LM) to recover the linguistic dependencies CTC ignores.

## Decoding

| Method | What it does | When |
| --- | --- | --- |
| **Greedy / best-path** | argmax per frame, then collapse | Fast, approximate; no LM |
| **Prefix beam search** | Beam over collapsed prefixes, merging paths | Standard with an external LM |
| **CTC + LM (shallow fusion)** | Add $\lambda \log p_{LM}$ during beam search | Production ASR |

Greedy decoding is *not* the argmax over label sequences (best path ≠ best labeling), because many paths can collapse to the same string. Beam search approximates the true argmax.

## What an interviewer expects you to say

1. Frame the problem: unknown alignment between $T$ frames and a shorter label sequence.
2. Introduce the **blank** symbol and the **collapse rule** (merge repeats, then drop blanks).
3. State that the loss **marginalizes over all alignments** via forward-backward DP (exact, not sampled).
4. Name the **conditional-independence-across-frames** assumption and its consequence: CTC has no built-in LM, so you fuse an external one.
5. Bonus: contrast with attention-based seq2seq (no monotonicity assumption, but harder to stream) and RNN-T (adds a label-dependent prediction network).

## Common confusions

- **"The blank means silence."** No. Blank means "emit nothing / no label transition here." Silence is just a region the acoustic model maps to blanks, but blank is a structural token, not a phoneme.
- **"Greedy decoding gives the most likely transcript."** It gives the most likely *path*, which after collapsing may not be the most likely *transcript*.
- **"CTC needs aligned data."** The whole point is that it doesn't; it learns the alignment implicitly.
- **"CTC models language."** It doesn't; it is conditionally independent across frames. Linguistic structure comes from an external LM at decode time.
- **"CTC only works for speech."** It works for any monotonic alignment task: handwriting recognition, OCR, lip reading, keyword spotting.

---

*Related: [RNN-Transducer (RNN-T)](/concepts/rnn-transducer/), [Automatic speech recognition](/concepts/automatic-speech-recognition/), [Forward-backward and Viterbi](/concepts/forward-backward-and-viterbi/), [Hidden Markov models](/concepts/hidden-markov-models/).*
