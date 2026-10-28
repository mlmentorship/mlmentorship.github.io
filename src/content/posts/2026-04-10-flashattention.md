---
title: "FlashAttention"
description: "Derive a mergeable softmax row, rescale its numerator and denominator across tiles, and distinguish mathematical equivalence, storage, traffic, and latency."
date: "2026-04-10"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Attention needs a normalized weighted sum, not a permanently stored score matrix.
FlashAttention tiles that computation to reduce global-memory storage and data movement while preserving the mathematical attention operator.
Floating-point reduction order can change bits; validate numerical tolerance rather than claiming bitwise identity.

**Task:** one query sees scores $[0,\log3]$ with scalar values $[2,6]$.
Process one score/value pair per tile without retaining prior scores.
Recover the same output as dense softmax, then change masking and the performance regime.

Use finite allowed scores and a float64 reference tolerance of $10^{-12}$ for these checks.
Production tolerances depend on dtype, shape, and kernel.

## The mechanism

A materialized baseline writes score and probability matrices of size $n\times n$ to high-bandwidth memory, HBM.
The tiled algorithm keeps a score tile and row summaries in on-chip working storage.
It still computes every allowed dense pair; memory traffic, auxiliary storage, arithmetic, and latency are different quantities.

For processed scores $s_j$ and value vectors $v_j$, retain

$$
m=\max_j s_j,\qquad
\ell=\sum_j e^{s_j-m},\qquad
u=\sum_j e^{s_j-m}v_j.
$$

Here $u$ is an **unnormalized numerator**; output is $O=u/\ell$.
For a new valid block, set

$$
m'=\max(m,\max s_{block}),\quad a=e^{m-m'},\qquad
\ell'=a\ell+\sum_{block}e^{s_j-m'},\quad
u'=au+\sum_{block}e^{s_j-m'}v_j.
$$

Changing reference from $m$ to $m'$ multiplies every earlier exponential by $a$.
Rescaling both totals preserves their ratio and admits the new block under a common reference.
Initialize $m=-\infty$, $\ell=0$, and $u=0$; the first valid block has old-state scale zero.
Skip entirely forbidden blocks without evaluating an undefined $-\infty-(-\infty)$.

| Processed tile | Maximum | Denominator | Numerator | Output |
| --- | --- | --- | --- | --- |
| Score $0$, value $2$ | $0$ | $1$ | $2$ | $2$ |
| Then score $\log3$, value $6$ | $\log3$ | $1/3+1=4/3$ | $2/3+6=20/3$ | $5$ |

The second maximum rescales old totals by $1/3$.
Dense evaluation agrees: $(1\cdot2+3\cdot6)/(1+3)=5$.
Forgetting rescaling gives $(2+6)/(1+1)=4$, a useful failing check.

<!-- visual:flashattention-standard-hbm-path -->
<figure class="learning-figure editorial-illustration" aria-labelledby="flashattention-standard-hbm-path-edition-title"><p class="visual-title" id="flashattention-standard-hbm-path-edition-title">The avoided HBM round trips belong to intermediates, not every tensor</p><svg viewBox="0 0 360 510" role="img" aria-labelledby="flashattention-standard-hbm-path-edition-svg-title flashattention-standard-hbm-path-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="flashattention-standard-hbm-path-edition-svg-title">The avoided HBM round trips belong to intermediates, not every tensor</title><desc id="flashattention-standard-hbm-path-edition-svg-desc">A declared two-byte-intermediate toy compares a materialized attention baseline with exact tiled attention. Scores and probabilities each incur one n-squared-element write and read in the baseline. Tiling avoids those global intermediates but still moves inputs, weights as applicable, and final output and retains dense quadratic arithmetic.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Toy 2-byte intermediates; count only</tspan><tspan x="180" dy="19">score/probability traffic</tspan></text><text x="90" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">MATERIALIZED</tspan></text><text x="270" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">TILED EXACT</tspan></text><path d="M15 91L345 91" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="111" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Scores: n by n</tspan></text><g data-editorial-node><rect x="15" y="125" width="150" height="92" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="166.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">write 2n^2 bytes</tspan><tspan x="90" dy="19">read 2n^2 bytes</tspan></text></g><g data-editorial-node><rect x="195" y="125" width="150" height="92" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="147.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">tile stays on</tspan><tspan x="270" dy="19">chip</tspan><tspan x="270" dy="19">no global score</tspan><tspan x="270" dy="19">matrix</tspan></text></g><path d="M15 232L345 232" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="252" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Probabilities: n by n</tspan></text><g data-editorial-node><rect x="15" y="266" width="150" height="92" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="307.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">write 2n^2 bytes</tspan><tspan x="90" dy="19">read 2n^2 bytes</tspan></text></g><g data-editorial-node><rect x="195" y="266" width="150" height="92" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="288.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">online softmax</tspan><tspan x="270" dy="19">state</tspan><tspan x="270" dy="19">no global P</tspan><tspan x="270" dy="19">matrix</tspan></text></g><path d="M15 373L345 373" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="393" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">What does not disappear</tspan></text><g data-editorial-node><rect x="15" y="407" width="150" height="73" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="429.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">Q, K, V, output</tspan><tspan x="90" dy="19">dense attention</tspan><tspan x="90" dy="19">work</tspan></text></g><g data-editorial-node><rect x="195" y="407" width="150" height="73" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="270" y="429.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">Q, K, V, output</tspan><tspan x="270" dy="19">dense attention</tspan><tspan x="270" dy="19">work</tspan></text></g></svg><figcaption><strong>Read it this way:</strong> For these declared two-byte intermediates, four transfers total 8n^2 bytes in the baseline. Avoiding them does not eliminate all HBM traffic or quadratic arithmetic. Actual intermediate precision can differ. <span class="visual-reader-check" data-reader-check="flashattention-standard-hbm-path"><strong>Check:</strong> Which quadratic materializations does tiled attention avoid without eliminating all HBM reads and writes?</span></figcaption></figure>

### Change masking

**New task:** the second score/value pair is future-only for this causal query.
Exclude it before local reductions and skip that empty tile.
**Check:** retained state stays $(m,\ell,u)=(0,1,2)$, giving output $2$, not $5$.
**Use when:** visibility, padding, or a window restricts the allowed pairs.
This reference rejects a fully masked row; match the chosen library's declared empty-row behavior rather than dividing by zero.

### Change the performance regime

**New task:** compare a short sequence on a different GPU.
Keep output tolerance checks and measure kernel time, launch overhead, utilization, and peak memory separately.
**Check:** illustrative baseline $7$ microseconds versus tiled $8$ gives speedup $7/8=0.875$, despite unchanged mathematical output.
Those timings are invented inputs to the comparison, not measurements.
**Use when:** shape or hardware changes the bottleneck; asymptotics cannot supply a universal crossover length.

## Backward pass: trade memory for compute

Backward can recompute score/probability tiles from Q/K and saved row normalization information.
It need not retain the entire probability matrix.
Saved outputs, statistics, and RNG requirements depend on the algorithm and dropout convention.
This is a kernel-level recompute-versus-store tradeoff, not cost-free memory removal.

Dense arithmetic remains $O(n^2d)$.
Auxiliary attention storage can be $O(nd)$, linear in sequence length at fixed width.
This does not imply linear HBM traffic or latency; tile sizes, on-chip capacity, and scheduling matter.

**Reconstruct:** does reducing quadratic storage make dense attention subquadratic in sequence length?

<details><summary>Answer</summary>

No. The algorithm still evaluates the allowed dense query/key pairs.
Sparse attention changes that pair set; linear-attention methods change the operator or its representation.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

<!-- visual:flashattention-sram-tile-stream -->
<figure class="learning-figure" aria-labelledby="flashattention-tile-title" aria-describedby="flashattention-tile-description">
<p class="visual-kicker">Learning objective</p>
<p class="visual-title" id="flashattention-tile-title">Follow one query block as key and value blocks stream through SRAM.</p>
<p id="flashattention-tile-description">The grid represents the full score matrix; only a query-key tile enters working storage here. Each tile updates row summaries before the next key/value block arrives.</p>
<div class="visual-scroll attention-tile-scroll">
<svg class="attention-tile-visual" viewBox="0 0 1000 500" role="img" aria-labelledby="flashattention-tile-svg-title flashattention-tile-svg-description" width="1000" height="500">
<title id="flashattention-tile-svg-title">FlashAttention streams one score tile through SRAM instead of materializing the full matrix</title>
<desc id="flashattention-tile-svg-description">The conceptual score grid highlights one Bq by Bk tile. HBM supplies query and key/value blocks to on-chip storage. The running maximum m, denominator ell, and unnormalized numerator u are updated. Output O equals u divided by ell. The final output block and row statistics are written, not the global score matrix.</desc>
<defs><marker id="flashattention-tile-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="var(--viz-focus-stroke)" /></marker></defs>
<rect x="24" y="24" width="420" height="370" rx="10" fill="var(--viz-canvas)" stroke="var(--c-rule)" />
<text x="48" y="58" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="20" font-weight="700">Global score matrix S</text>
<text x="48" y="82" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="13">n query rows x n key columns</text>
<text x="148" y="108" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="12">K blocks</text><text x="62" y="157" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="12" transform="rotate(-90 62 157)">Q blocks</text>
<text x="137" y="125" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="12">K1</text><text x="195" y="125" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="12">K2</text><text x="253" y="125" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="12">K3</text><text x="311" y="125" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="12">K4</text>
<text x="87" y="157" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="12">Q1</text><text x="87" y="215" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="12">Q2</text><text x="87" y="273" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="12">Q3</text><text x="87" y="331" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="12">Q4</text>
<rect x="120" y="132" width="50" height="50" rx="4" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><rect x="178" y="132" width="50" height="50" rx="4" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><rect x="236" y="132" width="50" height="50" rx="4" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><rect x="294" y="132" width="50" height="50" rx="4" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" />
<rect x="120" y="190" width="50" height="50" rx="4" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><rect x="178" y="190" width="50" height="50" rx="4" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><rect x="236" y="190" width="50" height="50" rx="4" fill="var(--viz-focus-bg)" stroke="var(--viz-focus-stroke)" stroke-width="3" /><rect x="294" y="190" width="50" height="50" rx="4" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" />
<rect x="120" y="248" width="50" height="50" rx="4" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><rect x="178" y="248" width="50" height="50" rx="4" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><rect x="236" y="248" width="50" height="50" rx="4" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><rect x="294" y="248" width="50" height="50" rx="4" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" />
<rect x="120" y="306" width="50" height="50" rx="4" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><rect x="178" y="306" width="50" height="50" rx="4" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><rect x="236" y="306" width="50" height="50" rx="4" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" /><rect x="294" y="306" width="50" height="50" rx="4" fill="var(--viz-neutral-bg)" stroke="var(--c-rule)" />
<text x="261" y="220" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="12" font-weight="700">Sij</text><text x="48" y="374" fill="var(--viz-focus-stroke)" font-family="Lato, sans-serif" font-size="13" font-weight="700">Only this tile is resident</text>
<line x1="444" y1="214" x2="476" y2="214" stroke="var(--viz-focus-stroke)" stroke-width="3" marker-end="url(#flashattention-tile-arrow)" />
<rect x="468" y="24" width="508" height="370" rx="10" fill="var(--viz-canvas)" stroke="var(--c-rule)" />
<text x="494" y="58" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="20" font-weight="700">One SRAM tile at a time</text><text x="494" y="82" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="13">stream blocks, update state, evict the tile</text>
<rect x="494" y="110" width="116" height="112" rx="7" fill="var(--viz-neutral-bg)" stroke="var(--viz-state-stroke)" /><text x="552" y="136" text-anchor="middle" fill="var(--viz-state-stroke)" font-family="Lato, sans-serif" font-size="13" font-weight="700">HBM</text><text x="552" y="162" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="15">Qi</text><text x="552" y="184" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="12">Kj, Vj</text><text x="552" y="207" text-anchor="middle" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="11">load blocks</text>
<line x1="610" y1="150" x2="646" y2="150" stroke="var(--viz-focus-stroke)" stroke-width="2" marker-end="url(#flashattention-tile-arrow)" /><line x1="610" y1="190" x2="646" y2="190" stroke="var(--viz-focus-stroke)" stroke-width="2" marker-end="url(#flashattention-tile-arrow)" />
<rect x="650" y="100" width="284" height="142" rx="8" fill="var(--viz-focus-bg)" stroke="var(--viz-focus-stroke)" stroke-width="2" /><text x="792" y="126" text-anchor="middle" fill="var(--viz-focus-stroke)" font-family="Lato, sans-serif" font-size="14" font-weight="700">GPU SRAM</text>
<rect x="672" y="145" width="78" height="50" rx="4" fill="var(--viz-canvas)" stroke="var(--c-rule)" /><rect x="761" y="145" width="78" height="50" rx="4" fill="var(--viz-canvas)" stroke="var(--c-rule)" /><rect x="850" y="145" width="62" height="50" rx="4" fill="var(--viz-canvas)" stroke="var(--c-rule)" />
<text x="711" y="174" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">Qi</text><text x="800" y="174" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">Kj, Vj</text><text x="881" y="174" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">Sij</text>
<text x="711" y="188" text-anchor="middle" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="10">Bq x d</text><text x="800" y="188" text-anchor="middle" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="10">Bk x d</text><text x="881" y="188" text-anchor="middle" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="10">Bq x Bk</text>
<line x1="792" y1="242" x2="792" y2="270" stroke="var(--viz-focus-stroke)" stroke-width="2" marker-end="url(#flashattention-tile-arrow)" /><rect x="650" y="274" width="284" height="68" rx="8" fill="var(--viz-neutral-bg)" stroke="var(--viz-output-stroke)" stroke-width="2" /><text x="792" y="299" text-anchor="middle" fill="var(--viz-output-stroke)" font-family="Lato, sans-serif" font-size="13" font-weight="700">streaming softmax update</text><text x="792" y="322" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="12">mi, elli, ui -&gt; next tile</text>
<text x="494" y="374" fill="var(--viz-output-stroke)" font-family="Lato, sans-serif" font-size="13" font-weight="700">Final write: Oi and row statistics; never S</text>
</svg>
</div>
<figcaption><strong>Read it this way:</strong> the left grid is conceptual, not a retained global matrix. A query block consumes key/value tiles, updating its maximum, denominator, and unnormalized numerator. After all allowed keys, divide numerator by denominator and write the output. Actual on-chip residency and saved statistics depend on the kernel version. <span class="visual-reader-check" data-reader-check="flashattention-sram-tile-stream"><strong>Check:</strong> Which state survives between tiles? Which large matrix is never retained globally?</span></figcaption>
</figure>

## What an interviewer expects you to say

Report mathematical equivalence, numerical tolerance, arithmetic, auxiliary memory, traffic, and observed latency as distinct claims.
Memory-boundedness depends on the actual shape, device, and implementation.

At $n=8192$, $d=128$, the term ratio $n^2/(nd)=n/d=64$ compares tensor sizes only.
It is not a measured traffic ratio or speedup.

The original normalized-output recurrence is an alternative state convention:

```
m_new = max(m_i, max(S_ij))
ell_new = exp(m_i - m_new) * ell_i + sum(exp(S_ij - m_new))
O_new   = (exp(m_i - m_new) * ell_i * O_i + exp(S_ij - m_new) @ V_j) / ell_new
```

Here $O_i$ is already normalized; multiplying by $\ell_i$ reconstructs its numerator before merging.
Do not substitute unnormalized $u_i$ into that recurrence unchanged.

## Common confusions

Mathematical equivalence does not imply identical floating-point bits or equal low-precision errors.
Validate forward and backward against a stable reference across masks, extreme logits, and dtypes.
Recomputation increases some arithmetic while removing stored intermediates.

## Why interviewers care

FlashAttention-2 changes work partitioning; FlashAttention-3 adds hardware-oriented asynchrony and low-precision techniques.
Check supported device, head dimensions, masks, and dtype in the version being deployed.
Low-precision variants need their own error characterization.

## Reading list

- *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* [(Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* [(Dao, 2023)](https://arxiv.org/abs/2307.08691)
- *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* [(Shah et al., 2024)](https://arxiv.org/abs/2407.08608)
- Tri Dao's implementation notes complement the papers; they are not a substitute for workload-specific measurements.

---

*Related: [speculative decoding](/concepts/speculative-decoding/), [LayerNorm versus BatchNorm](/concepts/batchnorm-vs-layernorm/), and [production LLM inference design](/questions/design-production-llm-inference-service/).*

</div>
