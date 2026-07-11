---
title: "Knowledge-graph embeddings"
description: "How to learn vector representations of entities and relations so that link prediction becomes a geometric operation. Covers TransE, DistMult, ComplEx, and RotatE, plus why the scoring function determines which relation patterns a model can express."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

Knowledge-graph embeddings map **entities** and **relations** of a graph of (head, relation, tail) triples into a continuous vector space, with a **scoring function** $f(h, r, t)$ that ranks true triples above false ones, turning **link prediction** into a geometric / algebraic operation.

## Why it matters

Knowledge graphs (entities like titles, people, genres, products linked by typed relations) power recommendation, search, and question answering. Embedding them lets you **predict missing links** ("which genre is this new title?"), compute entity similarity, and inject structured side-information into recsys and RAG. The interview angle is sharp: **the choice of scoring function determines which relation patterns (symmetry, antisymmetry, inversion, composition) the model can represent**, a clean test of representational reasoning.

## The task: knowledge-graph completion

A KG is a set of triples $(h, r, t)$, e.g. (`Inception`, `directed_by`, `Nolan`). Graphs are radically incomplete, so the goal is **link prediction**: score candidate triples and rank the true tail (or head) highly. Trained with a **margin / ranking loss** against **negative samples** (corrupt $h$ or $t$), evaluated with **Mean Reciprocal Rank (MRR)** and **Hits@k**.

## The four models to know

### TransE: translation

Model the relation as a translation in embedding space:

$$
f(h,r,t) = -\lVert \mathbf{h} + \mathbf{r} - \mathbf{t}\rVert,
$$

so a true triple satisfies $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$. Simple, scalable, intuitive. **Limitation**: it cannot model **symmetric** relations (would force $\mathbf{r} = 0$) or **one-to-many / many-to-one** relations (many valid tails collapse to one point).

### DistMult: bilinear diagonal

$$
f(h,r,t) = \mathbf{h}^\top \operatorname{diag}(\mathbf{r})\, \mathbf{t} = \sum_i h_i r_i t_i.
$$

Efficient, captures pairwise feature interactions. **Limitation**: the score is **symmetric in $h$ and $t$**, so it cannot distinguish $(h,r,t)$ from $(t,r,h)$, which is useless for antisymmetric relations like `parent_of`.

### ComplEx: complex bilinear

Move embeddings into $\mathbb{C}^k$ and use the Hermitian product:

$$
f(h,r,t) = \mathrm{Re}\big(\mathbf{h}^\top \operatorname{diag}(\mathbf{r})\, \bar{\mathbf{t}}\big).
$$

The complex conjugate $\bar{\mathbf{t}}$ breaks the symmetry, so ComplEx handles **symmetric *and* antisymmetric** relations, a strict generalization of DistMult.

### RotatE: rotation in complex space

$$
f(h,r,t) = -\lVert \mathbf{h} \circ \mathbf{r} - \mathbf{t}\rVert, \quad |r_i| = 1,
$$

each relation is an element-wise **rotation** (unit-modulus complex multiply). Rotations compose and invert, so RotatE can express **symmetry, antisymmetry, inversion, and composition**, the most expressive of the four on relation patterns.

## Which patterns each model expresses

| Model | Space | Symmetry | Antisymmetry | Inversion | Composition |
| --- | --- | --- | --- | --- | --- |
| **TransE** | $\mathbb{R}^k$ | ✗ | ✓ | ✓ | ✓ |
| **DistMult** | $\mathbb{R}^k$ | ✓ | ✗ | ✗ | ✗ |
| **ComplEx** | $\mathbb{C}^k$ | ✓ | ✓ | ✓ | ✗ |
| **RotatE** | $\mathbb{C}^k$ | ✓ | ✓ | ✓ | ✓ |

This table *is* the interview answer: pick the model by which relation patterns your graph contains.

## Where this fits in recsys / RAG

- **Recsys side-information**: embed a catalog KG (titles, actors, genres) and concatenate entity embeddings with user/item collaborative-filtering vectors to fight cold-start and add semantics.
- **Beyond shallow embeddings**: **R-GCN** and other relational GNNs generalize these scoring functions with message passing; **node2vec / metapath2vec** learn embeddings from random walks.
- **KG + LLM**: structured triples ground LLM answers and constrain RAG retrieval.

## What an interviewer expects you to say

1. Frame the task as **link prediction over (h, r, t) triples**, trained with **negative sampling** and a ranking loss, evaluated with **MRR / Hits@k**.
2. Give **TransE** ($\mathbf{h}+\mathbf{r}\approx\mathbf{t}$) and immediately name its failure on **symmetric and 1-to-many** relations.
3. Explain that **DistMult is symmetric** (can't do antisymmetry), **ComplEx fixes it via complex conjugation**, and **RotatE models relations as rotations** to also capture composition.
4. Tie model choice to **relation patterns** in the data.
5. Bonus: connect to **GNNs (R-GCN)** and to recsys cold-start / RAG grounding.

## Common confusions

- **"More dimensions is the main lever."** The **scoring function's inductive bias** matters more than dimensionality: DistMult literally cannot represent antisymmetry at any width.
- **"TransE handles any relation."** It breaks on symmetric and many-to-one relations by construction.
- **"These are just word embeddings."** They jointly embed **entities and typed relations** with a relation-specific operator, not a single similarity space.
- **"Link prediction is classification."** It's a **ranking** problem over corrupted negatives; accuracy is the wrong metric, MRR/Hits@k are standard.
- **"KG embeddings replaced GNNs."** They're the shallow end; relational GNNs add message passing and usually win when neighborhood structure is rich.

---

*Related: [Graph neural networks](/concepts/graph-neural-networks/), [Word embeddings](/concepts/word-embeddings/), [Negative sampling strategies](/questions/negative-sampling-strategies/), [Matrix factorization for recsys](/concepts/matrix-factorization-recsys/), [Content-based filtering](/concepts/content-based-filtering/).*
