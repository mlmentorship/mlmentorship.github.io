---
title: "Belief propagation (message passing)"
description: "How to compute marginals in a graphical model without summing over an exponential number of configurations. The sum-product algorithm passes local messages along the graph; exact on trees, approximate (loopy BP) on general graphs."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

Belief propagation is a **message-passing** algorithm for computing marginal distributions in a graphical model. Nodes exchange local "messages" summarizing the evidence from their part of the graph; the **sum-product** variant computes marginals, the **max-product** variant computes the most probable configuration. It is **exact on trees** and an approximation (**loopy BP**) on graphs with cycles.

## Why it matters

Inference (computing $p(x_i \mid \text{evidence})$) is the central operation in probabilistic models, and the naive sum over all configurations is exponential. Belief propagation is *the* algorithm that exploits the graph's factorization to make it tractable. It's the engine behind:

- **HMM forward-backward** and **CRF** training/decoding (these are BP on a chain).
- Decoding **LDPC / turbo codes** (loopy BP, the reason your phone's error correction works).
- General **factor-graph** inference in vision, sensor fusion, and probabilistic programming.

Interviewers use it to test whether you understand that **inference cost is governed by graph structure**, not just by the number of variables.

## The intuition

A graphical model factorizes a joint distribution into local factors:

$$
p(\mathbf{x}) = \frac{1}{Z}\prod_{a} \psi_a(\mathbf{x}_a).
$$

To get a marginal $p(x_i)$ you must sum out every other variable, which is exponential in general. BP avoids the blow-up by noticing that on a **tree**, the sum **distributes**: you can push summations inside products and reuse partial sums. Each "message" is exactly one of those reusable partial sums, flowing along an edge.

## The sum-product algorithm (factor graphs)

Two message types alternate until they reach the root / converge:

**Variable → factor** (multiply incoming messages from other factors):

$$
\mu_{x \to a}(x) = \prod_{b \in N(x)\setminus a} \mu_{b \to x}(x).
$$

**Factor → variable** (multiply the factor by incoming messages, then sum out the other variables):

$$
\mu_{a \to x}(x) = \sum_{\mathbf{x}_a \setminus x} \psi_a(\mathbf{x}_a) \prod_{y \in N(a)\setminus x} \mu_{y \to a}(y).
$$

The marginal at a variable is the product of all incoming messages (normalized):

$$
p(x_i) \propto \prod_{a \in N(x_i)} \mu_{a \to x_i}(x_i).
$$

Replace the inner $\sum$ with a $\max$ and you get **max-product** (a.k.a. max-sum in log space), which finds the MAP configuration, the general version of **Viterbi**.

## Exact vs approximate

| Graph structure | BP behavior | Cost |
| --- | --- | --- |
| **Tree / chain** | Exact marginals in two passes (leaves→root→leaves) | $O(\text{edges} \cdot \lvert V \rvert^2)$ |
| **General graph** | **Loopy BP**: iterate messages until (hopefully) convergence; approximate | per-iteration linear in edges |
| **Treewidth-$k$ graph** | Exact via the **junction-tree** algorithm | exponential in treewidth $k$ |

The deep fact: **exact inference is exponential in the graph's treewidth**, not its size. A chain has treewidth 1 (cheap); a fully connected grid has high treewidth (hard). That's why we fall back to loopy BP, variational inference, or sampling on dense graphs.

## Loopy BP

Run the same message updates on a graph with cycles, iterating until messages stop changing. There's **no guarantee of convergence or correctness**, yet it works remarkably well in practice: it's how modern error-correcting codes are decoded and is closely connected to variational (Bethe free energy) approximations.

## What an interviewer expects you to say

1. State that BP computes **marginals by passing local messages** and exploits the **factorization** of the joint to avoid the exponential sum.
2. Distinguish **sum-product (marginals)** from **max-product (MAP / Viterbi)**.
3. Know it is **exact on trees**, and that **HMM forward-backward and CRF inference are special cases** of sum-product on a chain.
4. State the key complexity insight: exact inference is **exponential in treewidth**; use **junction tree** for exactness or **loopy BP / variational / sampling** otherwise.
5. Bonus: loopy BP has no convergence guarantee but powers LDPC/turbo decoding.

## Common confusions

- **"BP always gives the right answer."** Only on trees (and junction trees). On loopy graphs it's a heuristic approximation.
- **"Sum-product and max-product are different algorithms."** Same message structure; one sums to marginalize, the other maxes to find the mode.
- **"Inference is hard because there are many variables."** It's hard because of **treewidth / connectivity**. A million-variable chain is easy; a 50-variable dense graph can be infeasible.
- **"Forward-backward is unrelated to BP."** It *is* sum-product BP on a chain. Viterbi is max-product BP on a chain.
- **"Messages are probabilities."** They're unnormalized factors (functions of a variable); you normalize only at the end to read off a marginal.

---

*Related: [Graphical models](/concepts/graphical-models/), [Forward-backward and Viterbi](/concepts/forward-backward-and-viterbi/), [Conditional random fields](/concepts/conditional-random-fields/), [Hidden Markov models](/concepts/hidden-markov-models/), [Expectation-maximization](/concepts/expectation-maximization/).*
