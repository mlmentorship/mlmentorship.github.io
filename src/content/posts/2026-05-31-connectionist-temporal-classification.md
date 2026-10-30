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

<!-- visual:ctc-blank-preserves-repeat -->
<figure class="learning-figure" aria-labelledby="ctc-blank-repeat-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="ctc-blank-repeat-title">A blank separates repeated letters so both survive CTC collapse</p><div class="visual-argument" data-visual-argument="ctc-blank-preserves-repeat" data-argument-kind="ledger"><p class="visual-argument-premise">The blank keeps two equal letters from merging</p><svg viewBox="0 0 360 322" role="img" aria-label="The blank keeps two equal letters from merging. First merge consecutive identical emissions, then remove blanks. A blank between the a emissions preserves the repeated output aa."><title>The blank keeps two equal letters from merging</title><desc>First merge consecutive identical emissions, then remove blanks. A blank between the a emissions preserves the repeated output aa.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Alignment path</tspan></text><rect data-proof-token="a" x="20" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="58" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="58" dy="0">a</tspan></text><rect data-proof-token="a" x="100" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="138" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="138" dy="0">a</tspan></text><rect data-proof-token="_" x="180" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="218" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="218" dy="0">_</tspan></text><rect data-proof-token="a" x="260" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="298" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="298" dy="0">a</tspan></text><text x="20" y="126" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Merge consecutive repeats</tspan></text><rect data-proof-token="a" x="20" y="156" width="102.66666666666667" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="71.33333333333334" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="71.33333333333334" dy="0">a</tspan></text><rect data-proof-token="_" x="126.66666666666667" y="156" width="102.66666666666667" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="178" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">_</tspan></text><rect data-proof-token="a" x="233.33333333333334" y="156" width="102.66666666666667" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="284.6666666666667" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="284.6666666666667" dy="0">a</tspan></text><text x="20" y="224" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Remove blank</tspan></text><rect data-proof-token="a" x="20" y="254" width="156" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="98" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="98" dy="0">a</tspan></text><rect data-proof-token="a" x="180" y="254" width="156" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="258" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="258" dy="0">a</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-grid--two" role="group" aria-label="Two CTC paths traced through merge repeats and remove blanks">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 215" role="img" aria-labelledby="ctc-adjacent-title ctc-adjacent-desc">
				<title id="ctc-adjacent-title">Adjacent repeated labels collapse to one label</title>
				<desc id="ctc-adjacent-desc">The frame path L, L first merges consecutive repeats into one L. Removing blanks changes nothing, so the output contains one L.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="182" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">NO SEPARATOR · ONE L</text>
				<text class="viz-axis-label" x="20" y="57">PATH</text>
				<rect class="viz-node viz-node--input" x="116" y="38" width="30" height="28" rx="4"></rect>
				<rect class="viz-node viz-node--input" x="154" y="38" width="30" height="28" rx="4"></rect>
				<text class="viz-node-label" x="131" y="57" text-anchor="middle">L</text>
				<text class="viz-node-label" x="169" y="57" text-anchor="middle">L</text>
				<path d="M150 72V86M146 82L150 86L154 82" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<text class="viz-axis-label" x="20" y="107">1 · MERGE REPEATS</text>
				<rect class="viz-node" x="135" y="89" width="30" height="28" rx="4"></rect>
				<text class="viz-node-label" x="150" y="108" text-anchor="middle">L</text>
				<text class="viz-label" x="176" y="107">adjacent pair merged</text>
				<path d="M150 123V137M146 133L150 137L154 133" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<text class="viz-axis-label" x="20" y="158">2 · REMOVE BLANKS</text>
				<rect class="viz-node viz-node--output" x="135" y="140" width="30" height="28" rx="4"></rect>
				<text class="viz-node-label" x="150" y="159" text-anchor="middle">L</text>
				<text class="viz-callout" x="150" y="192" text-anchor="middle">output: one L</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 215" role="img" aria-labelledby="ctc-separated-title ctc-separated-desc">
				<title id="ctc-separated-title">A blank between repeated labels preserves both labels</title>
				<desc id="ctc-separated-desc">The frame path L, blank, L has no consecutive repeated labels, so the merge step leaves all three tokens. The blank is then removed without running merge again, so the output contains two L labels.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="182" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">BLANK SEPARATOR · TWO Ls</text>
				<text class="viz-axis-label" x="20" y="57">PATH</text>
				<rect class="viz-node viz-node--input" x="104" y="38" width="30" height="28" rx="4"></rect>
				<path d="M150 38L165 52L150 66L135 52Z" style="fill:var(--viz-neutral-bg);stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:3 2"></path>
				<rect class="viz-node viz-node--input" x="166" y="38" width="30" height="28" rx="4"></rect>
				<text class="viz-node-label" x="119" y="57" text-anchor="middle">L</text>
				<text class="viz-node-label" x="150" y="56" text-anchor="middle">∅</text>
				<text class="viz-node-label" x="181" y="57" text-anchor="middle">L</text>
				<path d="M150 72V86M146 82L150 86L154 82" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<text class="viz-axis-label" x="20" y="107">1 · MERGE REPEATS</text>
				<text class="viz-label" x="150" y="107">L</text>
				<text class="viz-label" x="168" y="107">∅</text>
				<text class="viz-label" x="186" y="107">L</text>
				<text class="viz-label" x="209" y="107">unchanged</text>
				<path d="M150 123V137M146 133L150 137L154 133" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<text class="viz-axis-label" x="20" y="158">2 · REMOVE BLANKS</text>
				<rect class="viz-node viz-node--output" x="116" y="140" width="30" height="28" rx="4"></rect>
				<rect class="viz-node viz-node--output" x="154" y="140" width="30" height="28" rx="4"></rect>
				<text class="viz-node-label" x="131" y="159" text-anchor="middle">L</text>
				<text class="viz-node-label" x="169" y="159" text-anchor="middle">L</text>
				<text class="viz-callout" x="150" y="192" text-anchor="middle">output: two Ls</text>
			</svg>
		</section>
	</div>
	<div class="visual-derivation-caption">CTC merges repeats before it removes blanks. <code>L L</code> merges immediately, but <code>L ∅ L</code> has no adjacent repeat at step 1; removing <code>∅</code> at step 2 leaves both Ls.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="ctc-blank-preserves-repeat"><strong>Check:</strong> Why must repeated labels be merged before blanks are removed?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="ctc-blank-preserves-repeat">First merge consecutive identical emissions, then remove blanks. A blank between the a emissions preserves the repeated output aa.</span></figcaption>
</figure>

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
