---
title: "Conditional random fields (CRFs)"
description: "A CRF models labels for a whole sequence and scores transitions jointly. Linear-chain CRFs improve taggers when neighboring labels constrain each other."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A CRF is a **discriminative**, undirected graphical model that defines $p(\mathbf{y} \mid \mathbf{x})$ over a structured output $\mathbf{y}$ (e.g. a label sequence), scoring entire labelings jointly through feature functions over cliques, most commonly a **linear-chain CRF** that couples adjacent labels.

CRFs are the classic answer to **structured prediction**: when your outputs are interdependent (the label of token $t$ depends on token $t-1$), independent per-token softmax classification is wrong because it can produce **illegal or incoherent label sequences** (e.g. an `I-PER` tag right after an `O` tag in BIO tagging). A CRF layer fixes this by modeling transitions.

They remain interview-relevant because:

- A **linear-chain CRF on top of a BiLSTM or transformer encoder** was the standard NER / POS / chunking architecture and still appears in production sequence taggers.
- CRF vs HMM is the cleanest way to show you understand **generative vs discriminative** modeling of sequences.
- The training objective is a clean example of a globally-normalized log-likelihood with a forward-algorithm partition function.

## The model

A linear-chain CRF scores a full label sequence $\mathbf{y} = (y_1, \dots, y_T)$ given input $\mathbf{x}$:

$$
p(\mathbf{y} \mid \mathbf{x}) = \frac{1}{Z(\mathbf{x})} \exp\!\Big( \sum_{t} \psi_t(y_t, \mathbf{x}) + \sum_{t} A(y_{t-1}, y_t) \Big),
$$

where $\psi_t$ is the **emission / unary** score (how well label $y_t$ fits position $t$, often the logits from a neural encoder), $A(y_{t-1}, y_t)$ is a learned **transition** score between adjacent labels, and

$$
Z(\mathbf{x}) = \sum_{\mathbf{y}'} \exp(\cdots)
$$

is the **partition function**: a sum over all $|V|^T$ possible labelings. $Z$ couples the whole sequence through **global normalization**. Per-token softmax instead normalizes each position locally and independently.

<!-- visual:crf-global-path-scoring -->
<figure class="learning-figure plot-panel" aria-labelledby="crf-path-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="crf-path-visual-title">Transition scores can overturn the locally best labels</p><div class="visual-argument" data-visual-argument="crf-global-path-scoring" data-argument-kind="bars"><p class="visual-argument-premise">Transition penalties can defeat the locally best labels</p><svg viewBox="0 0 360 394" role="img" aria-label="Transition penalties can defeat the locally best labels. Local emissions favor AA, but its transition penalty is -5. Globally, AB or BA scores 3 and beats AA at -1."><title>Transition penalties can defeat the locally best labels</title><desc>Local emissions favor AA, but its transition penalty is -5. Globally, AB or BA scores 3 and beats AA at -1.</desc><text x="18" y="23" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">AA: emissions 4; transition -5</tspan></text><path d="M28 64L332 64" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="-1" data-proof-min="-1" data-proof-max="3" data-proof-zero="104" x="28" y="46" width="76" height="35" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></rect><text x="28" y="106" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="28" dy="0">-1</tspan></text><text x="18" y="140" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">AB: emissions 3; transition 0</tspan></text><path d="M28 181L332 181" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="3" data-proof-min="-1" data-proof-max="3" data-proof-zero="104" x="104" y="163" width="228" height="35" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><text x="28" y="223" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="28" dy="0">3</tspan></text><text x="18" y="257" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">BB: emissions 2; transition 0</tspan></text><path d="M28 298L332 298" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="2" data-proof-min="-1" data-proof-max="3" data-proof-zero="104" x="104" y="280" width="152" height="35" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></rect><text x="28" y="340" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text-soft)"><tspan x="28" dy="0">2</tspan></text><text x="28" y="374" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="28" dy="0">-1</tspan></text><text x="332" y="374" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="332" dy="0">3</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 350" role="img" aria-labelledby="crf-path-svg-title crf-path-svg-desc">
		<title id="crf-path-svg-title">Independent emissions compared with CRF whole-path scoring</title>
		<desc id="crf-path-svg-desc">For the two tokens Ada Lovelace, independent selection picks O with emission 3 and I-PER with emission 3, totaling 6 but forming an incoherent BIO sequence. In the CRF example, the O to I-PER transition has score minus 5, so that complete path scores 1. The B-PER to I-PER path combines emissions 2 and 3 with transition score plus 1, scores 6, and is selected as coherent. These are illustrative learned scores rather than universal hard constraints.</desc>
		<defs>
			<marker id="crf-path-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path class="viz-arrow-forward" d="M0 0L10 5L0 10Z"></path></marker>
		</defs>
		<rect class="viz-plot-bg" x="5" y="5" width="350" height="152" rx="6"></rect>
		<text class="viz-axis-label" x="15" y="25">INDEPENDENT SOFTMAX · MAXIMIZE EACH EMISSION</text>
		<text class="viz-node-value" x="105" y="50">Ada</text>
		<text class="viz-node-value" x="255" y="50">Lovelace</text>
		<rect class="viz-node viz-node--input" x="55" y="62" width="100" height="55" rx="8"></rect>
		<rect class="viz-node viz-node--input" x="205" y="62" width="100" height="55" rx="8"></rect>
		<text class="viz-node-label" x="105" y="85">O</text>
		<text class="viz-node-value" x="105" y="103">emission = 3</text>
		<text class="viz-node-label" x="255" y="85">I-PER</text>
		<text class="viz-node-value" x="255" y="103">emission = 3</text>
		<text class="viz-callout" x="180" y="137" text-anchor="middle">3 + 3 = 6 · INVALID BIO: O → I-PER</text>
		<rect class="viz-plot-bg" x="5" y="170" width="350" height="175" rx="6"></rect>
		<text class="viz-axis-label" x="15" y="190">LINEAR-CHAIN CRF · SCORE COMPLETE PATHS</text>
		<text class="viz-node-value" x="105" y="215">Ada</text>
		<text class="viz-node-value" x="255" y="215">Lovelace</text>
		<path class="viz-forward" style="marker-end:url(#crf-path-arrow)" d="M155 255H202"></path>
		<text class="viz-edge-label" x="180" y="244">+1</text>
		<rect class="viz-node viz-node--focus" x="55" y="227" width="100" height="55" rx="8"></rect>
		<rect class="viz-node viz-node--focus" x="205" y="227" width="100" height="55" rx="8"></rect>
		<text class="viz-node-label" x="105" y="250">B-PER</text>
		<text class="viz-node-value" x="105" y="268">emission = 2</text>
		<text class="viz-node-label" x="255" y="250">I-PER</text>
		<text class="viz-node-value" x="255" y="268">emission = 3</text>
		<text class="viz-callout" x="180" y="303" text-anchor="middle">chosen: 2 + 3 + 1 = 6 · VALID BIO</text>
		<text class="viz-gradient-label" x="180" y="327">rejected O → I-PER: 3 + 3 − 5 = 1</text>
	</svg>
	<div class="visual-derivation-caption">the token emissions stay fixed. Softmax takes each column's maximum and cannot penalize the invalid <code>O → I-PER</code> pair. The CRF adds a learned score for each adjacent-label transition, compares complete paths, and selects <code>B-PER → I-PER</code> in this example. The forward algorithm sums the exponentiated scores of all paths to compute <var>Z</var>; Viterbi finds the highest-scoring path.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="crf-global-path-scoring"><strong>Check:</strong> Why does the locally preferred O-to-I-PER path lose after transition scoring?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="crf-global-path-scoring">Local emissions favor AA, but its transition penalty is -5. Globally, AB or BA scores 3 and beats AA at -1.</span></figcaption>
</figure>

## Training and inference

The partition function looks intractable ($|V|^T$ terms) but factorizes over the chain:

- **Training**: maximize $\log p(\mathbf{y} \mid \mathbf{x})$. The gradient needs $Z(\mathbf{x})$ and the marginals, both computed by the **forward algorithm** (the same dynamic program as HMM forward-backward) in $O(T |V|^2)$.
- **Decoding**: find $\arg\max_\mathbf{y} p(\mathbf{y} \mid \mathbf{x})$ with the **Viterbi algorithm**, also $O(T |V|^2)$.

So a CRF reuses exactly the [forward-backward and Viterbi](/concepts/forward-backward-and-viterbi/) machinery, but on a *discriminatively trained, globally normalized* model.

## CRF vs HMM vs softmax tagger

| | Models | Normalization | Features |
| --- | --- | --- | --- |
| **HMM** | $p(\mathbf{x}, \mathbf{y})$ generative | local (per emission/transition) | tied to generative story |
| **MEMM** | $p(\mathbf{y}\mid\mathbf{x})$, per-step | local (per step) → **label bias** | rich, but biased |
| **Linear-chain CRF** | $p(\mathbf{y}\mid\mathbf{x})$ | **global** (one $Z$ per sequence) | rich, no label bias |
| **Independent softmax** | $\prod_t p(y_t\mid\mathbf{x})$ | local, independent | no transition modeling |

The CRF's global normalization is what cures the **label-bias problem** of MEMMs (locally normalized models that can't redistribute probability mass once committed at a step).

## The neural CRF (BiLSTM-CRF / Transformer-CRF)

In modern systems the encoder (BiLSTM or transformer) produces the **emission scores** $\psi_t$, and a small learned **transition matrix** $A$ sits on top. The whole stack is trained end-to-end with the CRF negative log-likelihood. The encoder captures rich context; the CRF enforces valid, coherent label transitions. This combination reliably beats a softmax-per-token head on tasks with strong output structure (NER, slot filling).

## What an interviewer expects you to say

1. State that a CRF models $p(\mathbf{y}\mid\mathbf{x})$ **over the whole sequence**, with emission + transition scores and a **global partition function $Z$**.
2. Explain *why* it beats independent softmax: it models **label dependencies / transitions** and avoids illegal sequences.
3. Know that **training uses the forward algorithm** (for $Z$) and **decoding uses Viterbi**, both $O(T|V|^2)$.
4. Place it on the **generative-vs-discriminative** map (HMM is the generative cousin) and mention the **label-bias** problem CRFs fix relative to MEMMs.
5. Bonus: the **BiLSTM-CRF / encoder-CRF** pattern (neural encoder for emissions, CRF layer for structure).

## Common confusions

- **"CRF = HMM."** HMM is generative and locally normalized; CRF is discriminative and globally normalized. CRFs can use arbitrary, overlapping input features.
- **"You need a CRF whenever you tag sequences."** Only when output structure matters. With a strong contextual encoder (large transformer), the marginal gain of a CRF head shrinks because the encoder already captures most dependencies, but it still helps enforce hard constraints.
- **"The partition function is intractable."** For a chain it's an $O(T|V|^2)$ forward pass. It's only intractable for general (loopy) graph structures.
- **"CRFs are obsolete."** The CRF *layer* is still a standard, cheap way to enforce coherent label sequences on top of any encoder.

---

*Related: [Forward-backward and Viterbi](/concepts/forward-backward-and-viterbi/), [Hidden Markov models](/concepts/hidden-markov-models/), [Belief propagation](/concepts/belief-propagation/), [Graphical models](/concepts/graphical-models/).*
