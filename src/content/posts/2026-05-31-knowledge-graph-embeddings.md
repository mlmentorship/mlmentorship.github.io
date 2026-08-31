---
title: "Knowledge-graph embeddings"
description: "Knowledge-graph embeddings turn link prediction into vector scoring. Compare TransE, DistMult, ComplEx, and RotatE by the relation patterns they can represent."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Knowledge-graph embeddings map **entities** and **relations** of a graph of (head, relation, tail) triples into a continuous vector space, with a **scoring function** $f(h, r, t)$ that ranks true triples above false ones, turning **link prediction** into a geometric / algebraic operation.

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

<!-- visual:kge-antisymmetry-score-swap -->
<figure class="learning-figure" aria-labelledby="kge-antisymmetry-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="kge-antisymmetry-title">Why can no amount of width make DistMult represent an antisymmetric relation?</p>
	<div class="visual-grid--two">
		<section class="visual-panel plot-panel" aria-labelledby="distmult-swap-title">
			<h4 id="distmult-swap-title">DistMult: swapping is invisible</h4>
			<p>Real multiplication commutes in every dimension.</p>
			<svg viewBox="0 0 300 250" role="img" aria-labelledby="distmult-swap-svg-title distmult-swap-svg-desc">
				<title id="distmult-swap-svg-title">DistMult gives a parent relation and its reversal equal scores</title>
				<desc id="distmult-swap-svg-desc">A solid arrow marks the true triple Ada parent of Ben, and a dashed arrow marks the false reversed triple Ben parent of Ada. DistMult multiplies the same three real values after swapping the entities, so both sums are equal and the model cannot rank the true direction above the false one.</desc>
				<rect class="viz-plot-bg" x="5" y="5" width="290" height="240" rx="3"></rect>
				<circle class="viz-node viz-node--input" cx="65" cy="52" r="25"></circle>
				<text class="viz-node-label" x="65" y="57">Ada</text>
				<circle class="viz-node viz-node--output" cx="235" cy="52" r="25"></circle>
				<text class="viz-node-label" x="235" y="57">Ben</text>
				<path class="viz-roc-curve" d="M92 45 H199"></path>
				<path class="viz-arrow-forward" d="M199 39 L211 45 L199 51 Z"></path>
				<text class="viz-edge-label" x="150" y="36" style="font-size:11px">parent_of · true</text>
				<path class="viz-baseline" d="M208 69 H101"></path>
				<path class="viz-arrow-forward" d="M101 63 L89 69 L101 75 Z"></path>
				<text class="viz-edge-label" x="150" y="87" style="font-size:11px">parent_of · false</text>
				<text class="viz-callout" x="18" y="122">score(A, r, B) = Σ Aᵢ rᵢ Bᵢ</text>
				<text class="viz-callout" x="18" y="147">score(B, r, A) = Σ Bᵢ rᵢ Aᵢ</text>
				<path class="viz-operating-guide" d="M30 164 H270"></path>
				<text class="viz-node-label" x="150" y="194">forced equal</text>
				<text class="viz-node-value" x="150" y="216">true cannot outrank reverse</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel" aria-labelledby="complex-swap-title">
			<h4 id="complex-swap-title">ComplEx: conjugation preserves direction</h4>
			<p>Swapping entities moves the conjugate to the other vector.</p>
			<svg viewBox="0 0 300 250" role="img" aria-labelledby="complex-swap-svg-title complex-swap-svg-desc">
				<title id="complex-swap-svg-title">ComplEx can give a parent relation and its reversal different scores</title>
				<desc id="complex-swap-svg-desc">The same solid true arrow and dashed false reverse arrow connect Ada and Ben. In ComplEx, swapping the entities changes which entity vector is complex-conjugated. The two expressions are therefore not forced to match, allowing the model to score Ada parent of Ben above Ben parent of Ada.</desc>
				<rect class="viz-plot-bg" x="5" y="5" width="290" height="240" rx="3"></rect>
				<circle class="viz-node viz-node--input" cx="65" cy="52" r="25"></circle>
				<text class="viz-node-label" x="65" y="57">Ada</text>
				<circle class="viz-node viz-node--output" cx="235" cy="52" r="25"></circle>
				<text class="viz-node-label" x="235" y="57">Ben</text>
				<path class="viz-roc-curve" d="M92 45 H199"></path>
				<path class="viz-arrow-forward" d="M199 39 L211 45 L199 51 Z"></path>
				<text class="viz-edge-label" x="150" y="36" style="font-size:11px">parent_of · true</text>
				<path class="viz-baseline" d="M208 69 H101"></path>
				<path class="viz-arrow-forward" d="M101 63 L89 69 L101 75 Z"></path>
				<text class="viz-edge-label" x="150" y="87" style="font-size:11px">parent_of · false</text>
				<text class="viz-callout" x="18" y="122">score(A, r, B) = Re Σ Aᵢ rᵢ B̄ᵢ</text>
				<text class="viz-callout" x="18" y="147">score(B, r, A) = Re Σ Bᵢ rᵢ Āᵢ</text>
				<path class="viz-operating-guide" d="M30 164 H270"></path>
				<text class="viz-node-label" x="150" y="194">free to differ</text>
				<text class="viz-node-value" x="150" y="216">true can outrank reverse</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> compare the two score pairs after swapping Ada and Ben. DistMult only reorders real scalar factors, so equality is unavoidable at every width. ComplEx moves the conjugate to the other entity; its score need not stay equal, so the directed fact can outrank its reversal. The broad pattern table below applies this same operator-first test to all four models. Original schematic checked against the <a href="https://proceedings.mlr.press/v48/trouillon16.html">ComplEx paper</a> and the <a href="https://arxiv.org/abs/1902.10197">RotatE comparison of relation patterns</a>.</figcaption>
</figure>

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
