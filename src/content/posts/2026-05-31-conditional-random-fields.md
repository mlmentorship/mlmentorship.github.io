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
