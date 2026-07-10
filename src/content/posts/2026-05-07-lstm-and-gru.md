---
title: "LSTM and GRU: gating as Hadamard products"
description: "Recurrent networks fail because gradients vanish through repeated matmul. Gates fix this by using elementwise multiplication to control information flow. Then transformers replaced them anyway."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

**LSTM** and **GRU** are recurrent units with learned gates that decide, per timestep, what to keep and what to forget. Each gate is a sigmoid output, applied via Hadamard (elementwise) product to the hidden state. The state is no longer multiplied by a weight matrix at every step, so gradients can flow.

## Why it matters

A vanilla RNN updates its hidden state as $h_t = \tanh(W h_{t-1} + U x_t)$. Backpropagating through $T$ timesteps multiplies the gradient by $W$ a total of $T$ times. Eigenvalues of $W$ less than 1 vanish; eigenvalues greater than 1 explode. Both kill learning ([Pascanu et al., 2013](https://arxiv.org/abs/1211.5063)).

LSTMs ([Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf)) replaced repeated matmul on the cell state with a Hadamard-product update, opening a "gradient highway." GRUs ([Cho et al., 2014](https://arxiv.org/abs/1406.1078)) simplified this to two gates with similar empirical performance.

Both are now mostly historical. Transformers replaced them everywhere. But the gating idea persists in attention masking, gating in mixture-of-experts, and residual gating in modern architectures.

## The LSTM cell

State: hidden state $h_t$ and cell state $c_t$.

Gates (each computed from $[h_{t-1}, x_t]$):

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t]) \quad \text{(forget)} \\
i_t &= \sigma(W_i [h_{t-1}, x_t]) \quad \text{(input)} \\
o_t &= \sigma(W_o [h_{t-1}, x_t]) \quad \text{(output)} \\
\tilde{c}_t &= \tanh(W_c [h_{t-1}, x_t]) \quad \text{(candidate)}
\end{aligned}
$$

Update:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \qquad h_t = o_t \odot \tanh(c_t).
$$

The forget gate controls what fraction of the previous cell state survives. The input gate controls what fraction of the new candidate is written. The output gate controls what fraction of the cell is exposed.

## Why Hadamard products fix gradients

The cell-state recurrence is $c_t = f_t \odot c_{t-1} + (\dots)$. The Jacobian $\partial c_t / \partial c_{t-1} = \text{diag}(f_t)$ has bounded eigenvalues. If the forget gate stays near 1, gradients pass through unchanged; if it goes near 0, the cell forgets cleanly. No exponential blow-up or decay.

## The GRU cell

Two gates instead of three:

$$
\begin{aligned}
z_t &= \sigma(W_z [h_{t-1}, x_t]) \quad \text{(update)} \\
r_t &= \sigma(W_r [h_{t-1}, x_t]) \quad \text{(reset)} \\
\tilde{h}_t &= \tanh(W_h [r_t \odot h_{t-1}, x_t]) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t.
\end{aligned}
$$

Single hidden state, no separate cell. Slightly fewer parameters. Empirically comparable to LSTM on most tasks.

## Tradeoffs vs transformers

- **Sequential**: must process tokens one at a time. Cannot parallelize across the sequence dimension at training time.
- **Linear in sequence length** at inference (vs $O(n^2)$ for vanilla attention). The advantage that makes RNN-style models attractive again at very long context (Mamba, RWKV, linear attention).
- **No explicit attention**. Information from token $i$ to token $j$ has to survive $j - i$ gate updates. Long-range dependencies are still hard in practice.

## Common pitfalls

- **Treating LSTM and GRU as interchangeable**. They are close empirically but the cell state in LSTM gives sharper control over long-range memory.
- **Using vanilla RNNs in 2025**. Almost never the right choice. Either go LSTM/GRU or, more likely, transformer.
- **Forgetting truncated BPTT**. Backpropagating through 100k tokens is infeasible; truncate at a window (typically 64 to 256 tokens) and cut gradients.

## Related

- [Backpropagation](/concepts/backpropagation/).
- [The attention mechanism](/concepts/attention-mechanism/).
- [Vanishing and exploding gradients](/concepts/exploding-vanishing-gradients/).
