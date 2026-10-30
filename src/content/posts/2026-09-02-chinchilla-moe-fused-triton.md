---
title: "Chinchilla scaling, MoE, and fused Triton kernels"
description: "Derive Chinchilla limits for dense and MoE models, implement an MoE layer in PyTorch, and fuse projections in Triton when F exceeds D."
date: "2026-09-02"
draft: false
tags: ["concepts"]
category: "concepts"
roles: ["RE", "Systems MLE", "RS"]
rounds: ["Systems", "Performance", "Training"]
difficulty: "Advanced"
priority: "Specialist"
prerequisites: ["transformer-compute-memory-accounting", "gpu-memory-hierarchy"]
---

## Summary

Chinchilla scaling laws determine compute-optimal model size and token allocation for a given training budget. Mixture-of-experts (MoE) decoupling expands parameter capacity while keeping active FLOPs fixed. A fused Triton kernel removes the intermediate activation write-read cycle to high-bandwidth memory (HBM), speeding up MoE feed-forward layers when expert expansion $F$ exceeds hidden dimension $D$.

## Chinchilla scaling derivation for dense and MoE models

A fixed compute budget $C$ in floating-point operations (FLOPs) can be spent on a larger model parameter count $N$ or a larger dataset token count $D$. Loss decreases with both parameters and tokens according to an empirical power law:

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}.
$$

For a dense transformer, each parameter participates in roughly 2 FLOPs per token during the forward pass and 4 FLOPs per token during backpropagation. Total training compute is:

$$
C \approx 6 N D.
$$

Minimizing loss under the compute constraint $6 N D = C$ yields the optimal parameter and token allocation:

$$
\min_{N,D}\ \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} \quad \text{subject to}\quad 6 N D = C.
$$

Setting the partial derivatives equal via a Lagrangian multiplier produces the scaling relationship:

$$
N^* \propto C^{a}, \quad D^* \propto C^{b}, \quad a + b = 1.
$$

When $\alpha \approx \beta \approx 0.34$, the exponents yield equal scaling rates $a \approx b \approx 0.5$. This gives the compute-optimal training ratio of approximately 20 tokens per parameter:

$$
\frac{D^*}{N^*} \approx 20.
$$

### The Mixture-of-Experts scaling relationship

An MoE layer with $E$ experts and top-$k$ routing separates total stored parameters $N_t$ from active parameters $N_a$:

$$
N_a = \frac{k}{E} N_t + N_{\text{shared}}.
$$

Training compute per token depends on active parameters ($C \approx 6 N_a D$), while total model capacity scales with $N_t$. Effective parameters can be modeled as:

$$
N_{\text{eff}}(N_t, N_a) \approx N_a^{1-\gamma} N_t^{\gamma}, \quad \gamma \in (0, 1).
$$

Under this formulation, total parameter count $N_t$ can grow significantly faster than active compute $N_a$. MoE models scale knowledge capacity without increasing per-token FLOPs, constrained by GPU memory capacity and network communication rather than compute budget alone.

## PyTorch MoE implementation

An MoE feed-forward layer routes tokens to top-$k$ experts, permutes tokens to group them contiguously per expert, executes grouped matrix multiplications, and recombines expert outputs.

```python
import torch
import torch.nn.functional as F

def moe_ffn(x: torch.Tensor, Wup: torch.Tensor, Wdown: torch.Tensor, gate_w: torch.Tensor, k: int = 2) -> torch.Tensor:
    # x: [T, D], Wup: [E, D, F], Wdown: [E, F, D], gate_w: [D, E]
    T, D = x.shape
    E = Wup.shape[0]

    logits = x @ gate_w
    w, idx = torch.topk(logits, k, dim=-1)
    w = w.softmax(dim=-1)

    flat_idx = idx.reshape(-1)
    perm = torch.argsort(flat_idx, stable=True)
    xg = x.repeat_interleave(k, dim=0)[perm]
    sizes = torch.bincount(flat_idx, minlength=E)

    h = torch._grouped_mm(xg, Wup, sizes)
    h = F.silu(h) * h
    y = torch._grouped_mm(h, Wdown, sizes)

    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=x.device)
    y = y[inv].reshape(T, k, D)
    return (w.unsqueeze(-1) * y).sum(dim=1)
```

In standard execution, the intermediate tensor $h \in \mathbb{R}^{(T \cdot k) \times F}$ is written to HBM by the first grouped matrix multiplication and read back from HBM by the second.

## Fused Triton kernel mechanism

When an MoE layer is memory-bandwidth bound, reading and writing the intermediate activation tensor dominates execution time.

For $t$ tokens routed to an expert with hidden dimension $D$ and expanded dimension $F$:

- Unfused memory traffic (reads and writes): $4tD + 4tF + 4DF$ bytes.
- Fused memory traffic (intermediate kept in SRAM): $4tD + 4DF$ bytes.

The relative speedup from keeping $h$ in GPU SRAM registers is:

$$
\text{Speedup} \approx 1 + \frac{t F}{t D + D F}.
$$

When expansion ratio $F > D$ and per-expert batch size $t$ is moderate, the intermediate activation $tF$ dominates memory traffic. Fusing the up-projection, activation, and down-projection into one tile program removes the $4tF$ HBM traffic term.

<!-- visual:fused-moe-triton-sram-roofline -->
<figure class="learning-figure editorial-illustration" aria-labelledby="fused-moe-triton-sram-roofline-edition-title"><p class="visual-title" id="fused-moe-triton-sram-roofline-edition-title">Keep the intermediate on chip to avoid its write-and-read trip</p><div class="visual-argument" data-visual-argument="fused-moe-triton-sram-roofline" data-argument-kind="ledger"><p class="visual-argument-premise">Keep the intermediate on chip instead of sending it out and back</p><svg viewBox="0 0 360 240" role="img" aria-label="Keep the intermediate on chip instead of sending it out and back. The unfused route writes and rereads the intermediate in HBM. A fitting fused tile avoids that round trip, not the input and output traffic."><title>Keep the intermediate on chip instead of sending it out and back</title><desc>The unfused route writes and rereads the intermediate in HBM. A fitting fused tile avoids that round trip, not the input and output traffic.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Unfused intermediate route</tspan></text><rect data-proof-token="SRAM" x="20" y="58" width="76" height="50" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="58" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="58" dy="0">SRAM</tspan></text><rect data-proof-token="HBM write" x="100" y="58" width="76" height="50" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="138" y="78" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="138" dy="0">HBM</tspan><tspan x="138" dy="18">write</tspan></text><rect data-proof-token="HBM read" x="180" y="58" width="76" height="50" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="218" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="218" dy="0">HBM read</tspan></text><rect data-proof-token="SRAM" x="260" y="58" width="76" height="50" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="298" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="298" dy="0">SRAM</tspan></text><text x="20" y="142" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Fused fitting intermediate route</tspan></text><rect data-proof-token="SRAM" x="20" y="172" width="102.66666666666667" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="71.33333333333334" y="193" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="71.33333333333334" dy="0">SRAM</tspan></text><rect data-proof-token="reuse" x="126.66666666666667" y="172" width="102.66666666666667" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="178" y="193" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">reuse</tspan></text><rect data-proof-token="SRAM" x="233.33333333333334" y="172" width="102.66666666666667" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="284.6666666666667" y="193" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="284.6666666666667" dy="0">SRAM</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 542" role="img" aria-labelledby="fused-moe-triton-sram-roofline-edition-svg-title fused-moe-triton-sram-roofline-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="fused-moe-triton-sram-roofline-edition-svg-title">Fusion removes the intermediate round trip, not every memory transfer</title><desc id="fused-moe-triton-sram-roofline-edition-svg-desc">Aligned unfused and fused paths keep the same intermediate h with shape t by F. For two-byte elements, the unfused path writes 2tF bytes then reads 2tF bytes. A fitting fused tile avoids that 4tF-byte intermediate round trip. Input, weight, and output traffic remain.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Two-byte elements; same intermediate h [t,</tspan><tspan x="180" dy="19">F]</tspan></text><text x="90" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">TWO KERNELS</tspan></text><text x="270" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">FUSED TILE</tspan></text><path d="M15 91L345 91" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="111" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Where h waits between GEMMs</tspan></text><g data-editorial-node><rect x="15" y="125" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="163.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">HBM</tspan></text></g><g data-editorial-node><rect x="195" y="125" width="150" height="67" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="163.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">SRAM / registers</tspan></text></g><path d="M15 207L345 207" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="227" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Intermediate round trip</tspan></text><g data-editorial-node><rect x="15" y="241" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="270" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">write 2tF</tspan><tspan x="90" dy="19">read 2tF bytes</tspan></text></g><g data-editorial-node><rect x="195" y="241" width="150" height="67" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="270" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">no HBM round</tspan><tspan x="270" dy="19">trip</tspan></text></g><path d="M15 323L345 323" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="343" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">What still moves</tspan></text><g data-editorial-node><rect x="15" y="357" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="386" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">inputs, weights,</tspan><tspan x="90" dy="19">output</tspan></text></g><g data-editorial-node><rect x="195" y="357" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="270" y="386" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">inputs, weights,</tspan><tspan x="270" dy="19">output</tspan></text></g><text x="180" y="466" style="font-size:14px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="180" dy="0">4tF intermediate bytes avoided</tspan></text><text x="180" y="496" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Tile fit and other traffic still limit</tspan><tspan x="180" dy="19">performance.</tspan></text></svg><div class="visual-derivation-caption">The avoided transfer is the same intermediate written and read once: 4tF bytes for two-byte elements. Other traffic and tile-fit constraints remain.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="fused-moe-triton-sram-roofline"><strong>Check:</strong> Which exact memory round trip disappears? Which other memory traffic still remains?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="fused-moe-triton-sram-roofline">The unfused route writes and rereads the intermediate in HBM. A fitting fused tile avoids that round trip, not the input and output traffic.</span></figcaption></figure>

### Triton fused kernel implementation

```python
import triton
import triton.language as tl

@triton.jit
def fused_up_act_down_kernel(
    x_ptr, wup_ptr, wdown_ptr, y_ptr,
    stride_xt, stride_xd,
    stride_wup_d, stride_wup_f,
    stride_wd_f, stride_wd_d,
    stride_yt, stride_yd,
    T, D, F,
    BT: tl.constexpr, BD: tl.constexpr, BF: tl.constexpr,
):
    pid_t = tl.program_id(0)
    offs_t = pid_t * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, BD)

    x_tile = tl.load(
        x_ptr + offs_t[:, None] * stride_xt + offs_d[None, :] * stride_xd,
        mask=offs_t[:, None] < T,
    )
    acc = tl.zeros((BT, BD), dtype=tl.float32)

    for f0 in range(0, F, BF):
        offs_f = f0 + tl.arange(0, BF)
        Wup_tile = tl.load(
            wup_ptr + offs_d[:, None] * stride_wup_d + offs_f[None, :] * stride_wup_f
        )
        h = tl.dot(x_tile, Wup_tile)
        h = h * tl.sigmoid(h)

        Wd_tile = tl.load(
            wdown_ptr + offs_f[:, None] * stride_wd_f + offs_d[None, :] * stride_wd_d
        )
        acc += tl.dot(h.to(Wd_tile.dtype), Wd_tile)

    tl.store(
        y_ptr + offs_t[:, None] * stride_yt + offs_d[None, :] * stride_yd,
        acc.to(tl.bfloat16),
        mask=offs_t[:, None] < T,
    )
```

## Verification and roofline profiling

To verify memory-bandwidth savings, profile execution using Nsight Compute (`ncu`) or `torch.profiler` and inspect DRAM transfer counters (`dram__bytes_read` and `dram__bytes_write`).

1. Measure total HBM bytes moved by both implementations. The fused kernel moves approximately $4tF$ fewer bytes per layer invocation.
2. Sweep the expansion ratio $F/D$ at fixed token batch size $t$. The measured speedup increases monotonically with $F/D$ while the kernel remains memory-bound.
3. Sweep per-expert token count $t$. As $t$ increases, arithmetic intensity grows ($tFD / (tD + FD)$), transitioning the workload from memory-bandwidth bound to compute bound.
4. On a roofline plot, the unfused implementation sits on the HBM memory bandwidth line, while the fused implementation shifts upward toward peak Tensor Core compute throughput.

Related: [Transformer compute and memory accounting](/concepts/transformer-compute-memory-accounting/), [GPU memory hierarchy](/concepts/gpu-memory-hierarchy/), [Neural scaling laws and compute-optimal training](/concepts/neural-scaling-laws/).
