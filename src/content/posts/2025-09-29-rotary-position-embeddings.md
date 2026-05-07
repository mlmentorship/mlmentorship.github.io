---
title: "Rotary position embeddings (RoPE)"
description: "The dominant position encoding for modern LLMs. Encodes relative position by rotating Q and K in 2D subspaces, enabling clean context extrapolation."
date: "2025-09-29"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

RoPE encodes token position by rotating each pair of dimensions of the query and key vectors by an angle proportional to position, so that the inner product $Q_m^\top K_n$ depends only on the relative offset $m - n$.

## Why it matters

Standard absolute position embeddings (sinusoidal in original transformer; learned in BERT/GPT-2) are added to token embeddings at the input. They couple position with content additively and don't extend cleanly past the training context.

RoPE [(Su et al., 2021)](https://arxiv.org/abs/2104.09864) is a multiplicative scheme applied inside attention. It became the default in modern decoder LLMs: Llama 1/2/3, Mistral, Qwen, DeepSeek, GPT-NeoX. Its main practical advantage is that the same formula works for any sequence length, and several context-extension tricks (NTK scaling, YaRN, position interpolation) operate directly on RoPE's frequency base.

## The mechanism

Split the head dimension $d$ into $d/2$ pairs. For each pair $i \in \{0, \dots, d/2 - 1\}$ pick a frequency $\theta_i = 10000^{-2i/d}$ (same base as sinusoidal). For a token at position $m$, rotate the $i$-th 2D pair by angle $m\theta_i$:

$$
\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} =
\begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
$$

Apply the same rotation to keys (with their position $n$). The inner product satisfies $\langle Q'_m, K'_n \rangle = f(Q_m, K_n, m - n)$. Depends only on the relative offset.

In code, RoPE is implemented as elementwise multiplies with precomputed `cos` and `sin` tables; no extra parameters.

## Why it works

- **Relative**: attention scores depend on $m - n$, not absolute positions, matching what attention should care about.
- **Decay with distance**: high-frequency pairs ($i$ small) rotate fast; their inner product decays with $|m-n|$, providing implicit locality bias.
- **Length-flexible**: rotations are well-defined at any position, so RoPE doesn't have a hard cap like learned embeddings.

## Context extension

To run a RoPE model past its training length:
- **Position interpolation** [(Chen et al., 2023)](https://arxiv.org/abs/2306.15595): linearly compress positions so the new max length maps to the original training range.
- **NTK-aware scaling**: increase the RoPE base $10000$ to a larger value so high-frequency components don't alias.
- **YaRN** [(Peng et al., 2023)](https://arxiv.org/abs/2309.00071): per-frequency interpolation tuned by training length statistics.

All operate directly on the frequency base; no architectural change.

## Common pitfalls

- **Applying RoPE to V.** Only Q and K are rotated; V is not.
- **Confusing with ALiBi.** ALiBi adds a fixed slope to attention scores; RoPE rotates Q/K. Both encode relative position but are different mechanisms.
- **Forgetting the base when extending context.** Naively running a 4K model at 32K without scaling produces garbage past 4K.
