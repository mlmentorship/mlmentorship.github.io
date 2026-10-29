---
title: "Transformer architecture: trace one complete block"
description: "Compute both residual updates, test causal visibility, and distinguish token communication, feature projections, normalization, and architectural variants."
date: "2025-11-05"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

A transformer block lets positions exchange information, then transforms each position's features.
Residual additions and normalization connect those operations without changing the token-by-feature output shape.

**Task:** trace a two-token, two-feature pre-norm block with input $X=[[1,3],[3,1]]$.
Use per-row LayerNorm with unit gain, zero bias, and zero epsilon only for this nonconstant-row calculation.
Production normalization uses declared positive epsilon.
Set query/key projections to zero and value/output projections to identity, with one causal attention head.
Use a row-wise FFN with identity matrices, zero biases, and ReLU.
What reaches each residual output, and can the second input row change the first output?

## The minimum mental model

A transformer block is:

```
x = x + Attention(LayerNorm(x))   # mix across positions
x = x + FFN(LayerNorm(x))         # mix across features
```

The second line uses the first line's updated tensor, not the original input.
Write these states as $X_1=X+A(\operatorname{LN}(X))$ and $X_2=X_1+F(\operatorname{LN}(X_1))$.
Token embeddings and the task's output projection sit outside this block.

The retained figure shows unmasked encoder connectivity; our numerical trace changes visibility to a causal decoder mask.
Attention also contains feature projections: the picture distinguishes primary communication roles, not exclusive ownership of feature mixing.

<!-- visual:transformer-two-mixing-axes -->
<figure class="learning-figure plot-panel" aria-labelledby="transformer-mixing-title">
	<p class="visual-kicker">One tensor, two dependency patterns</p>
	<p class="visual-title" id="transformer-mixing-title">Attention mixes token positions; the feed-forward layer transforms each position independently</p>
	<svg viewBox="0 0 360 420" role="img" aria-labelledby="transformer-mixing-svg-title transformer-mixing-svg-desc">
		<title id="transformer-mixing-svg-title">Attention connects token positions while the feed-forward network transforms each token independently</title>
		<desc id="transformer-mixing-svg-desc">The input is a token-by-feature tensor with three token rows and four feature columns. In the unmasked encoder example in stage one, arrows from token rows one, two, and three converge on the updated representation of token two, showing that self-attention can gather information across positions allowed by its mask. In stage two, three separate horizontal paths apply the same feed-forward network to each token row, with no path between rows, showing independent feature transformation at every position. Both sublayers preserve the token-by-feature shape and add their updates through pre-norm residual connections.</desc>
		<defs>
			<marker id="transformer-mixing-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" style="fill:var(--viz-edge)"></path></marker>
			<marker id="transformer-mixing-focus-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" style="fill:var(--viz-focus-stroke)"></path></marker>
		</defs>
		<rect class="viz-plot-bg" x="8" y="30" width="344" height="176" rx="5"></rect>
		<text class="viz-axis-label" x="14" y="18">1 · ATTENTION COMMUNICATES ACROSS TOKEN POSITIONS</text>
		<text class="viz-label" x="22" y="49">unmasked encoder rows</text>
		<text class="viz-label" x="262" y="49">updated token 2</text>
		<g aria-label="Three input token rows, each with four feature cells">
			<rect class="viz-node viz-node--input" x="22" y="60" width="118" height="30" rx="3"></rect>
			<path d="M51.5 60V90M81 60V90M110.5 60V90" class="viz-gridline"></path>
			<rect class="viz-node viz-node--focus" x="22" y="106" width="118" height="30" rx="3"></rect>
			<path d="M51.5 106V136M81 106V136M110.5 106V136" class="viz-gridline"></path>
			<rect class="viz-node viz-node--input" x="22" y="152" width="118" height="30" rx="3"></rect>
			<path d="M51.5 152V182M81 152V182M110.5 152V182" class="viz-gridline"></path>
			<text class="viz-callout" x="10" y="79">t₁</text>
			<text class="viz-callout" x="10" y="125">t₂</text>
			<text class="viz-callout" x="10" y="171">t₃</text>
		</g>
		<path d="M140 75C190 75 199 108 247 118M140 121H247M140 167C190 167 199 134 247 124" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;marker-end:url(#transformer-mixing-focus-arrow)"></path>
		<rect class="viz-node viz-node--output" x="249" y="106" width="90" height="30" rx="3"></rect>
		<path d="M271.5 106V136M294 106V136M316.5 106V136" class="viz-gridline"></path>
		<text class="viz-callout" x="294" y="157" text-anchor="middle">mask permits all rows</text>
		<text class="viz-label" x="180" y="194" text-anchor="middle">shape stays T × d; attention returns an update for every token row</text>
		<rect class="viz-plot-bg" x="8" y="228" width="344" height="140" rx="5"></rect>
		<text class="viz-axis-label" x="14" y="218">2 · THE FFN TRANSFORMS FEATURES WITHIN EACH TOKEN</text>
		<text class="viz-label" x="20" y="247">one row at a time</text>
		<text class="viz-label" x="253" y="247">same T × d shape</text>
		<g aria-label="Three isolated token paths through the same feed-forward network">
			<rect class="viz-node viz-node--input" x="20" y="258" width="76" height="24" rx="3"></rect>
			<rect class="viz-node viz-node--input" x="20" y="295" width="76" height="24" rx="3"></rect>
			<rect class="viz-node viz-node--input" x="20" y="332" width="76" height="24" rx="3"></rect>
			<path d="M96 270H128M96 307H128M96 344H128" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;marker-end:url(#transformer-mixing-arrow)"></path>
			<rect class="viz-node viz-node--focus" x="130" y="256" width="92" height="28" rx="3"></rect>
			<rect class="viz-node viz-node--focus" x="130" y="293" width="92" height="28" rx="3"></rect>
			<rect class="viz-node viz-node--focus" x="130" y="330" width="92" height="28" rx="3"></rect>
			<text class="viz-callout" x="176" y="274" text-anchor="middle">same FFN</text>
			<text class="viz-callout" x="176" y="311" text-anchor="middle">same FFN</text>
			<text class="viz-callout" x="176" y="348" text-anchor="middle">same FFN</text>
			<path d="M222 270H250M222 307H250M222 344H250" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;marker-end:url(#transformer-mixing-arrow)"></path>
			<rect class="viz-node viz-node--output" x="252" y="258" width="76" height="24" rx="3"></rect>
			<rect class="viz-node viz-node--output" x="252" y="295" width="76" height="24" rx="3"></rect>
			<rect class="viz-node viz-node--output" x="252" y="332" width="76" height="24" rx="3"></rect>
			<text class="viz-callout" x="10" y="274">t₁</text>
			<text class="viz-callout" x="10" y="311">t₂</text>
			<text class="viz-callout" x="10" y="348">t₃</text>
		</g>
		<text class="viz-label" x="180" y="384" text-anchor="middle">no row-to-row path: each position is processed independently</text>
		<text class="viz-callout" x="180" y="405" text-anchor="middle">x′ = x + Attention(LN(x))  →  x″ = x′ + FFN(LN(x′))</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> hold the <code>T × d</code> tensor shape fixed. Here, unmasked encoder attention connects every token row; a causal decoder omits future links. The shared FFN transforms each row independently. Each sublayer contributes an update through its own residual addition. Attention's internal feature projections are not shown. Sources: <a href="https://arxiv.org/abs/1706.03762">Vaswani et al. (2017)</a> and the <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html">PyTorch encoder-layer documentation</a>. <span class="visual-reader-check" data-reader-check="transformer-two-mixing-axes"><strong>Check:</strong> Can token 1 directly affect token 2 inside the FFN sublayer? Where can that communication occur instead?</span></figcaption>
</figure>

### Compute the two residuals

Each input row has mean $2$ and variance $1$, so normalized rows are $[-1,1]$ and $[1,-1]$.
Zero queries and keys give equal logits on allowed positions.
The causal first row sees only itself; the second averages both value rows.

| State | Row 0 | Row 1 |
| --- | --- | --- |
| $X$ | $[1,3]$ | $[3,1]$ |
| $\operatorname{LN}(X)$ | $[-1,1]$ | $[1,-1]$ |
| Attention contribution | $[-1,1]$ | $[0,0]$ |
| First residual $X_1$ | $[0,4]$ | $[3,1]$ |
| $\operatorname{LN}(X_1)$ | $[-1,1]$ | $[1,-1]$ |
| Row-wise FFN contribution | $[0,1]$ | $[1,0]$ |
| Second residual $X_2$ | $[0,5]$ | $[4,1]$ |

The first residual's row zero has mean $2$ and variance $4$; renormalization still gives $[-1,1]$.
ReLU acts within each row, and the same FFN parameters serve both rows.
No FFN edge directly joins different positions, although its inputs already contain attention-derived context.

### Change normalization placement

**New task:** use post-norm, $h=\operatorname{LN}(X+A(X))$, then $Y=\operatorname{LN}(h+F(h))$.
**Picture and change:** move normalization outside each addition and recompute branch inputs too.
**Check:** isolate zero branches on row $[1,3]$.
Pre-norm returns $[1,3]$; post-norm returns $[-1,1]$ under the same toy normalization.
**Use when:** the architecture changes residual-normalization order; its direct identity path is no longer the same.

### Change position visibility

**New task:** encode a complete document bidirectionally instead of predicting a next token.
Remove the causal restriction where allowed, while retaining padding validity and position semantics.
With the original input, both attention rows now average to $[0,0]$, giving first final row $[1,4]$.

**Check:** replace the second input row with $[1,3]$.
Bidirectional attention now changes the first final row to $[0,5]$.
The causal block's first final row remains $[0,5]$ before and after that perturbation.
**Use when:** the entire input is available; the future-perturbation invariant is no longer required.

### Shapes and failure boundaries

For $T$ tokens and width $d$, each residual contribution must return shape $T\times d$.
A head computes $Q=ZW_Q$, $K=ZW_K$, $V=ZW_V$, and $\operatorname{softmax}(QK^\top/\sqrt{d_k}+M)V$.
The additive mask $M$ is zero on allowed keys and negative infinity on forbidden keys.
Require an allowed key per query; concatenate head outputs and project back to residual width.
The position-wise FFN expands $d$ to its chosen hidden width, applies a nonlinearity or gate, then returns to $d$.

A causal mask alone does not validate training if teacher-forced inputs expose the target token.
Check shifted inputs, target alignment, padding, and absolute positions as part of the prediction contract.

## What an interviewer expects you to articulate

Name the objective and available positions before naming a model family.
Encoder-only, decoder-only, and encoder-decoder systems differ in visibility, cross-attention, and prediction targets.
See [their task-specific routes](/concepts/encoder-decoder-architectures/) rather than assuming every transformer is a causal language model.

**Reconstruct:** does representing an identity with zero residual branches prove a deeper model always trains at least as well?

<details><summary>Answer</summary>

No. Representational availability does not make optimization find that identity.
Normalization, finite precision, regularization, and changed training dynamics also prevent an automatic performance ordering.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

## The why behind each piece

### Why attention?

Attention permits content-dependent communication among allowed keys.
Dense unmasked attention connects the supplied context in one layer; causal, sparse, and window masks restrict that graph.
Parallel training across supplied positions does not remove autoregressive generation dependencies or finite context limits.

### Why multi-head?

Heads form separate learned query/key/value subspaces and attention distributions.
No theorem assigns each head a particular linguistic role.
At fixed total projection widths, parameter counts can match a single-head design.
Logits and softmax storage still depend on head count.

### Why residual connections?

[Residual addition](/concepts/residual-connections/) supplies a direct route alongside a learned branch.
Its derivative includes an identity contribution, but branch derivatives can cancel or amplify it.
Neither vanishing-gradient prevention nor successful deeper training follows automatically.

### Why LayerNorm (not BatchNorm)?

Feature-axis LayerNorm computes statistics within each token, avoiding cross-sample statistical coupling.
BatchNorm is a different reduction and train/evaluation contract, not mathematically impossible for sequence inputs.
The configured normalized axes matter; see [the shared-statistics trace](/concepts/batchnorm-vs-layernorm/).

### Why pre-norm vs post-norm?

Their equations and identity paths differ, as the core's zero-branch check shows.
Depth, initialization, warmup, residual scaling, optimizer, and final normalization affect empirical stability and quality.
The [original transformer](https://arxiv.org/abs/1706.03762) used post-norm; that does not define every later architecture.

### Why FFN?

It supplies a shared row-wise nonlinear transformation alongside attention's communication.
Attention's projections already mix features; the FFN is not the sole feature-mixing operation.
Hidden width is architectural, not necessarily four times residual width.

### Why positional encoding?

Without positional information or asymmetric masks, row permutation permutes attention outputs: the operation is equivariant, not invariant.
Causal masks already introduce an order-dependent restriction.
Position schemes add distinct absolute or relative signals; defined beyond-training-length values do not guarantee useful extrapolation.

## The scale story

[Kaplan et al.](https://arxiv.org/abs/2001.08361) and [Hoffmann et al.](https://arxiv.org/abs/2203.15556) fit empirical loss/resource relationships under specified training regimes.
Compute-optimal parameter/token allocation depends on those fits and constraints.
They do not guarantee monotonic downstream improvements from every added parameter or dataset.

## What's changed since 2017

| Choice | Actual change |
| --- | --- |
| RMSNorm | Scales by a root mean square without subtracting the token mean. |
| SwiGLU | Adds a multiplicative gate and separate input projections. |
| RoPE | Rotates queries and keys by position-dependent angles. |
| ALiBi | Adds a relative-distance term to allowed attention scores. |
| Learned or sinusoidal embeddings | Add a specified positional vector to the token representation. |
| GQA / MQA | Share key/value heads among query heads, changing cache dimensions. |
| Conditional experts | Route tokens among FFNs with additional capacity and dispatch contracts. |
| Bias or dropout removal | Changes the specified training computation, not an automatic quality-preserving simplification. |

## Common confusions

Parameter sharing, masks, normalization, and position mechanisms impose structural priors.
The dominant compute or memory term depends on width, length, batch, implementation, and training versus generation.
Architecture names do not replace a shape, visibility, or resource ledger.

---

*Related: [BatchNorm vs LayerNorm](/concepts/batchnorm-vs-layernorm/), [FlashAttention](/concepts/flashattention/), [KV cache](/concepts/kv-cache/).*

</div>
