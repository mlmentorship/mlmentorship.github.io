---
title: "Knowledge-graph embeddings"
description: "Knowledge-graph embeddings turn link prediction into vector scoring. Compare TransE, DistMult, ComplEx, and RotatE by the relation patterns they can represent."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

Knowledge-graph embeddings score typed triples, but the scoring operator determines which relation distinctions are even possible.

## Summary

Ada is Ben's parent; the reverse relation is false in this example.
Can swapping the entity embeddings change the proposed score?
If not, increasing width cannot teach this score the missing direction.

## The task: knowledge-graph completion

Represent a fact as head, relation and tail, $(h,r,t)$, with corresponding learned vectors.
Score candidate heads or tails under a stated higher-is-better convention.
An unobserved triple is not necessarily false; corrupted-entity negatives need a declared validity policy.
Architecture, training loss and ranking evaluation are separate choices.

## The four models to know

### TransE: translation

Use $f(h,r,t)=-\lVert h+r-t\rVert_2$.
Under exact zero residual, two true tails require $t_1=h+r=t_2$.
Likewise, two reversed zero-residual facts require $2r=0$ and collapse their entity embeddings.
These are exact-constraint limits; approximate ranking need not satisfy those equalities.

### DistMult: bilinear diagonal

$$
f(h,r,t)=\sum_j h_jr_jt_j=\sum_j t_jr_jh_j=f(t,r,h).
$$

For real scalars $h=1,r=2,t=3$, both scores are six.
Commutativity holds in every coordinate, so adding dimensions preserves the equality.

### ComplEx: complex bilinear

Use complex embeddings and $f(h,r,t)=\operatorname{Re}\sum_j h_jr_j\overline{t_j}$.
Choose $h=1,r=i,t=i$, with $i^2=-1$.
Forward score is $\operatorname{Re}(1\cdot i\cdot(-i))=1$.
Reverse score is $\operatorname{Re}(i\cdot i\cdot1)=-1$.

Conjugation changes which factors contribute imaginary signs.
It permits this directed distinction; it does not guarantee learning the correct facts.

<!-- visual:kge-antisymmetry-score-swap -->
<figure class="learning-figure" aria-labelledby="kge-antisymmetry-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="kge-antisymmetry-title">Complex conjugation lets the reversed relation receive a different score</p><div class="visual-argument" data-visual-argument="kge-antisymmetry-score-swap" data-argument-kind="ledger" data-argument-composition="arithmetic"><svg viewBox="0 0 360 260" role="img" aria-label="Complex conjugation makes relation direction matter. With h=i, r=i, and t=1, the forward ComplEx score is -1. Swapping head and tail gives +1 for the same relation."><title>Complex conjugation makes relation direction matter</title><desc>With h=i, r=i, and t=1, the forward ComplEx score is -1. Swapping head and tail gives +1 for the same relation.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Forward: real(h r conjugate(t))</tspan></text><g data-proof-token="i" data-proof-row="0" data-proof-position="0" data-proof-box="20,60,58,48"><text x="49" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="49" dy="0">i</tspan></text></g><g data-proof-token="x i" data-proof-row="0" data-proof-position="1" data-proof-box="84,60,58,48"><text x="113" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="113" dy="0">x i</tspan></text></g><g data-proof-token="x 1" data-proof-row="0" data-proof-position="2" data-proof-box="148,60,58,48"><text x="177" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="177" dy="0">x 1</tspan></text></g><g data-proof-token="=" data-proof-row="0" data-proof-position="3" data-proof-box="212,60,58,48"><text x="241" y="87" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="241" dy="0">=</tspan></text></g><g data-proof-token="-1" data-proof-row="0" data-proof-position="4" data-proof-box="276,60,58,48"><text x="305" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="305" dy="0">-1</tspan></text></g><text x="20" y="144" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Reverse: real(t r conjugate(h))</tspan></text><g data-proof-token="1" data-proof-row="1" data-proof-position="0" data-proof-box="20,176,58,48"><text x="49" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="49" dy="0">1</tspan></text></g><g data-proof-token="x i" data-proof-row="1" data-proof-position="1" data-proof-box="84,176,58,48"><text x="113" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="113" dy="0">x i</tspan></text></g><g data-proof-token="x -i" data-proof-row="1" data-proof-position="2" data-proof-box="148,176,58,48"><text x="177" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="177" dy="0">x -i</tspan></text></g><g data-proof-token="=" data-proof-row="1" data-proof-position="3" data-proof-box="212,176,58,48"><text x="241" y="203" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="241" dy="0">=</tspan></text></g><g data-proof-token="+1" data-proof-row="1" data-proof-position="4" data-proof-box="276,176,58,48"><text x="305" y="203" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="305" dy="0">+1</tspan></text><path d="M280 221L330 221" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption">swap Ada and Ben. DistMult only reorders real factors, forcing equal scores. ComplEx changes which entity is conjugated, permitting unequal scores. The worked scalar example gives 1 versus -1. Original schematic checked against <a href="https://proceedings.mlr.press/v48/trouillon16.html">ComplEx</a> and <a href="https://arxiv.org/abs/1902.10197">RotatE</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="kge-antisymmetry-score-swap"><strong>Check:</strong> Which algebraic operation allows the complex model's reversed edge to receive a different score?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="kge-antisymmetry-score-swap">With h=i, r=i, and t=1, the forward ComplEx score is -1. Swapping head and tail gives +1 for the same relation.</span></figcaption>
</figure>

### RotatE: rotation in complex space

Use $f(h,r,t)=-\lVert h\odot r-t\rVert_2$, with unit-modulus relation coordinates.
Multiplying $1$ by $i$ gives $i$ with zero residual; inverse rotation $-i$ maps it back to one.
The inverse is coordinatewise conjugation; composing exact rotations multiplies their coordinates.

## Which patterns each model expresses

| Operator | Discriminating test | Scope |
| --- | --- | --- |
| TransE | distinct tails force conflicting translations | exact zero residual |
| DistMult | head/tail swap leaves score unchanged | every shared-entity real score |
| ComplEx | conjugated swap can change real score | capacity, not learned accuracy |
| RotatE | conjugate inverse and componentwise composition | unit-modulus, exact rotations |

Pattern claims depend on their formal definitions and residual assumptions, not a universal expressivity ordering.

## Change the required information

### Require direction

**New task:** replace undirected similarity with `parent_of`.

**Picture and change:** reject the forced-symmetric score or use a directional operator with explicitly distinct entity roles.

**Check:** duplicating the DistMult coordinate gives twelve in both directions, still tied.
The one-coordinate ComplEx example gives one and minus one.

**Use when:** reversal must change the ranking; test held-out directed facts.

### Add an unseen entity

**New task:** score a new title with metadata but no trained ID vector.

**Picture and change:** supply an explicit feature/neighbor encoder or fallback, then evaluate unseen-entity generalization.

**Check:** a toy feature map $E([u,v])=u+iv$ maps $[1,0]$ to one.
It can enter the example's forward score, which becomes one.
That arithmetic supplies an embedding, not evidence that the asserted fact is true.

**Use when:** deployment must cover entities absent from the transductive training IDs.

## Reconstruct the limitation

Does algebraic capacity prove that a missing link is correct?

<details>
<summary>Reasoned answer</summary>

No. Observation coverage, negative validity, split leakage, training evidence and optimization remain separate requirements.
Changing width cannot remove DistMult's forced equality; changing its operator can.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Where this fits in recsys / RAG

Catalog graphs can supply typed side-information alongside collaborative features.
Relational GNNs add neighborhood computation; random-walk embeddings learn another neighborhood objective.
Neither is a universal replacement or winner.
Structured RAG inputs can also contain stale or false triples; graph grounding does not establish truth.

## What an interviewer expects you to say

Define the fact population, candidate set, negative construction and temporal/train/test split.
Evaluate filtered MRR or Hits@K under a declared known-positive filtering protocol and tie policy.
State whether encoder inputs can expose future or test information.

## Common confusions

Classification or ranking losses can train a triple score; the evaluation task determines the appropriate metrics.
Unobserved does not mean negative.
ID-only embeddings do not provide an unseen entity's representation.
Relation-pattern theorems use specific definitions; approximate learned scores need separate empirical evaluation.

---

*Related: [Graph neural networks](/concepts/graph-neural-networks/), [Word embeddings](/concepts/word-embeddings/), [Negative sampling strategies](/questions/negative-sampling-strategies/), [Matrix factorization for recsys](/concepts/matrix-factorization-recsys/), [Content-based filtering](/concepts/content-based-filtering/).*

</div>
