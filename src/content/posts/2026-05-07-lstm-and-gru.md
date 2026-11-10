---
title: "LSTM and GRU: gating as Hadamard products"
description: "Recurrent networks fail because gradients vanish through repeated matmul. Gates fix this by using elementwise multiplication to control information flow. Then transformers replaced them anyway."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**LSTM** and **GRU** are recurrent units with learned gates that decide, per timestep, what to keep and what to forget. Each gate is a sigmoid output, applied via Hadamard (elementwise) product to state or candidate vectors. The LSTM's direct cell-state path is scaled coordinate by coordinate instead of transformed by another dense matrix, so gradients have a controlled route through time.

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

**Learning objective:** compute one LSTM cell update coordinate by coordinate, then read the direct backward multiplier from the same forget gate.

<!-- visual:lstm-coordinate-gated-state-update -->
<figure class="learning-figure" aria-labelledby="lstm-coordinate-title">
	<p class="visual-kicker">Worked state update</p>
	<p class="visual-title" id="lstm-coordinate-title">Forget and input gates scale separate contributions before they add</p><div class="visual-argument" data-visual-argument="lstm-coordinate-gated-state-update" data-argument-kind="ledger" data-argument-composition="weighted-values"><svg viewBox="0 0 360 580" role="img" aria-label="Two gates scale two different contributions. For one coordinate, forget gate 0.8 retains 1.6 from old state 2; input gate 0.5 admits 1.5 from candidate 3. Their sum is 3.1."><title>Two gates scale two different contributions</title><desc>For one coordinate, forget gate 0.8 retains 1.6 from old state 2; input gate 0.5 admits 1.5 from candidate 3. Their sum is 3.1.</desc><text x="20" y="26" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Width: gate. Height: state.</tspan></text><text x="20" y="66" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">gate 0.8</tspan></text><rect data-picture-contribution="0" data-picture-source-member="0" x="20" y="224" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="0" data-picture-source-member="1" x="36" y="224" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="0" data-picture-source-member="2" x="52" y="224" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="0" data-picture-source-member="3" x="68" y="224" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="0" data-picture-source-member="4" x="84" y="224" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="0" data-picture-source-member="5" x="100" y="224" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="0" data-picture-source-member="6" x="116" y="224" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="0" data-picture-source-member="7" x="132" y="224" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="0" data-picture-source-member="8" x="20" y="204" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="0" data-picture-source-member="9" x="36" y="204" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="0" data-picture-source-member="10" x="52" y="204" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="0" data-picture-source-member="11" x="68" y="204" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="0" data-picture-source-member="12" x="84" y="204" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="0" data-picture-source-member="13" x="100" y="204" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="0" data-picture-source-member="14" x="116" y="204" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="0" data-picture-source-member="15" x="132" y="204" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="20" y="278" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">state 2</tspan></text><text x="20" y="313" style="font-size:20px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">area 1.6</tspan></text><path d="M50 332L98 362" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="210" y="66" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="210" dy="0">gate 0.5</tspan></text><rect data-picture-contribution="1" data-picture-source-member="16" x="210" y="224" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="17" x="226" y="224" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="18" x="242" y="224" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="19" x="258" y="224" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="20" x="274" y="224" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="21" x="210" y="204" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="22" x="226" y="204" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="23" x="242" y="204" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="24" x="258" y="204" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="25" x="274" y="204" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="26" x="210" y="184" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="27" x="226" y="184" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="28" x="242" y="184" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="29" x="258" y="184" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="30" x="274" y="184" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="210" y="278" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="210" dy="0">state 3</tspan></text><text x="210" y="313" style="font-size:20px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="210" dy="0">area 1.5</tspan></text><path d="M240 332L242 362" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><rect data-picture-collected-member="0" x="20" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="1" x="36" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="2" x="52" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="3" x="68" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="4" x="84" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="5" x="100" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="6" x="116" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="7" x="132" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="8" x="148" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="9" x="164" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="10" x="180" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="11" x="196" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="12" x="212" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="13" x="228" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="14" x="244" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="15" x="260" y="376" width="16" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="16" x="276" y="376" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="17" x="292" y="376" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="18" x="308" y="376" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="19" x="324" y="376" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="20" x="20" y="396" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="21" x="36" y="396" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="22" x="52" y="396" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="23" x="68" y="396" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="24" x="84" y="396" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="25" x="100" y="396" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="26" x="116" y="396" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="27" x="132" y="396" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="28" x="148" y="396" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="29" x="164" y="396" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="30" x="180" y="396" width="16" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="20" y="454" style="font-size:20px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Combined contribution: 3.1</tspan></text><text x="20" y="517" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Same tiles, regrouped; each</tspan><tspan x="20" dy="18">contributes 0.1</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 430" role="img" aria-labelledby="lstm-coordinate-svg-title lstm-coordinate-svg-desc">
			<title id="lstm-coordinate-svg-title">Three-coordinate LSTM cell-state update and direct gradient path</title>
			<desc id="lstm-coordinate-svg-desc">A worked example has three columns labeled coordinate A, B, and C. The previous cell state 2, negative 1, 0.5 is multiplied coordinate by coordinate by forget gate 0.9, 0.2, 1 to retain 1.8, negative 0.2, 0.5. Candidate negative 1, 0.5, 0.8 is multiplied by input gate 0.1, 0.8, 0.5 to write negative 0.1, 0.4, 0.4. Adding matching coordinates produces the new cell state 1.7, 0.2, 0.9. Along the direct cell-state path, a unit gradient is multiplied by the same forget gate and becomes 0.9, 0.2, 1.</desc>
			<rect class="viz-plot-bg" x="8" y="30" width="344" height="390" rx="5"></rect>
			<text class="viz-axis-label" x="184" y="20" text-anchor="middle">MATCHING STATE COORDINATES</text>
			<text class="viz-label" x="206" y="50" text-anchor="middle">A</text>
			<text class="viz-label" x="267" y="50" text-anchor="middle">B</text>
			<text class="viz-label" x="328" y="50" text-anchor="middle">C</text>
			<path class="viz-gridline" d="M176 55V347M237 55V347M298 55V347"></path>
			<text class="viz-axis-label" x="18" y="76">OLD CELL cₜ₋₁</text>
			<text class="viz-callout" x="206" y="76" text-anchor="middle">2.0</text>
			<text class="viz-callout" x="267" y="76" text-anchor="middle">−1.0</text>
			<text class="viz-callout" x="328" y="76" text-anchor="middle">0.5</text>
			<text class="viz-label" x="18" y="106">× forget gate fₜ</text>
			<text class="viz-callout" x="206" y="106" text-anchor="middle">0.9</text>
			<text class="viz-callout" x="267" y="106" text-anchor="middle">0.2</text>
			<text class="viz-callout" x="328" y="106" text-anchor="middle">1.0</text>
			<rect class="viz-node viz-node--input" x="174" y="121" width="174" height="38" rx="4"></rect>
			<text class="viz-axis-label" x="18" y="145">RETAINED</text>
			<text class="viz-callout" x="206" y="145" text-anchor="middle">1.8</text>
			<text class="viz-callout" x="267" y="145" text-anchor="middle">−0.2</text>
			<text class="viz-callout" x="328" y="145" text-anchor="middle">0.5</text>
			<text class="viz-axis-label" x="18" y="201">CANDIDATE c̃ₜ</text>
			<text class="viz-callout" x="206" y="201" text-anchor="middle">−1.0</text>
			<text class="viz-callout" x="267" y="201" text-anchor="middle">0.5</text>
			<text class="viz-callout" x="328" y="201" text-anchor="middle">0.8</text>
			<text class="viz-label" x="18" y="231">× input gate iₜ</text>
			<text class="viz-callout" x="206" y="231" text-anchor="middle">0.1</text>
			<text class="viz-callout" x="267" y="231" text-anchor="middle">0.8</text>
			<text class="viz-callout" x="328" y="231" text-anchor="middle">0.5</text>
			<rect class="viz-node viz-node--focus" x="174" y="246" width="174" height="38" rx="4"></rect>
			<text class="viz-axis-label" x="18" y="270">WRITTEN</text>
			<text class="viz-callout" x="206" y="270" text-anchor="middle">−0.1</text>
			<text class="viz-callout" x="267" y="270" text-anchor="middle">0.4</text>
			<text class="viz-callout" x="328" y="270" text-anchor="middle">0.4</text>
			<path class="viz-axis" d="M174 301H348"></path>
			<text class="viz-label" x="18" y="310">retained + written</text>
			<rect class="viz-node viz-node--output" x="174" y="316" width="174" height="38" rx="4"></rect>
			<text class="viz-axis-label" x="18" y="340">NEW CELL cₜ</text>
			<text class="viz-callout" x="206" y="340" text-anchor="middle">1.7</text>
			<text class="viz-callout" x="267" y="340" text-anchor="middle">0.2</text>
			<text class="viz-callout" x="328" y="340" text-anchor="middle">0.9</text>
			<text class="viz-axis-label" x="18" y="385">DIRECT BACKWARD PATH</text>
			<text class="viz-label" x="18" y="407">unit gradient × fₜ →</text>
			<text class="viz-callout" x="206" y="407" text-anchor="middle">0.9</text>
			<text class="viz-callout" x="267" y="407" text-anchor="middle">0.2</text>
			<text class="viz-callout" x="328" y="407" text-anchor="middle">1.0</text>
		</svg>
	</div>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> read down one column at a time. The forget gate scales only the matching old-state coordinate; the input gate scales only the matching candidate coordinate; then those two contributions add. On the direct cell-state path, backpropagation uses the same forget value: coordinate C copies state and gradient at 1.0, A mostly retains them at 0.9, and B deliberately forgets them at 0.2. Original worked example checked against <a href="https://www.bioinf.jku.at/publications/older/2604.pdf">Hochreiter and Schmidhuber's LSTM formulation</a> and the <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html">PyTorch LSTM equations</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="lstm-coordinate-gated-state-update"><strong>Check:</strong> Why does coordinate B become 0.2? Why is its direct-path gradient factor 0.2 not the entire recurrent derivative?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="lstm-coordinate-gated-state-update">For one coordinate, forget gate 0.8 retains 1.6 from old state 2; input gate 0.5 admits 1.5 from candidate 3. Their sum is 3.1. <strong>If the input gate closes, which contribution remains?</strong></span></figcaption>
</figure>

## Why Hadamard products fix gradients

The cell-state recurrence is $c_t = f_t \odot c_{t-1} + (\dots)$. Holding the gate values fixed, the direct cell-state path has Jacobian $\partial c_t / \partial c_{t-1} = \operatorname{diag}(f_t)$. If the forget gate stays near 1, gradients pass through nearly unchanged; if it goes near 0, that coordinate and its direct gradient are deliberately forgotten. This direct path cannot amplify a gradient because $0 < f_t < 1$, although repeated values below 1 can still make it decay.

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
