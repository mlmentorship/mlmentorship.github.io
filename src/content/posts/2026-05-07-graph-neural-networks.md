---
title: "Graph neural networks: message passing as A·X·W"
description: "Neighbors carry signal. A graph neural network averages each node's neighborhood and projects with a learned matrix. The same matmul as a CNN, on irregular structure."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **graph neural network** (GNN) updates each node's features by aggregating its neighbors' features and applying a learned transformation. The simplest variant is one matmul: $H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})$, where $\hat{A}$ is a normalized adjacency matrix and $H^{(l)}$ stacks node features.

A CNN exploits regular grid structure with shared local filters. A GNN applies shared parameters and local aggregation to neighborhoods defined by graph edges. This supports models for social networks, molecules, knowledge graphs, code ASTs, and recommender bipartite graphs.

GNNs power drug-discovery pipelines (AlphaFold's Evoformer, DeepMind's GNoME), large-scale recommenders (Pinterest's PinSAGE, Uber's GraphSAGE), and protein structure prediction. Modern transformers are arguably a special case (complete graph with attention as edge weighting).

## The mechanism: GCN

The graph convolutional network ([Kipf & Welling, 2017](https://arxiv.org/abs/1609.02907)) is the canonical GNN:

$$
H^{(l+1)} = \sigma\!\left(\hat{A} H^{(l)} W^{(l)}\right),
$$

where $\hat{A} = \tilde{D}^{-1/2} (A + I) \tilde{D}^{-1/2}$ is the symmetrically normalized adjacency with self-loops.

Decompose:

1. **Aggregate**: $\hat{A} H$ forms a degree-normalized weighted sum of each node's neighbors and itself (via the self-loop).
2. **Transform**: multiply by $W$ to mix features.
3. **Activate**: ReLU or similar.

After $L$ layers, each node has aggregated information from its $L$-hop neighborhood. The network is a sequence of matmuls, the same operation as any other deep learning model. The graph structure shows up only in $\hat{A}$.

<!-- visual:gcn-neighborhood-to-matrix-row -->
<figure class="learning-figure plot-panel" aria-labelledby="gcn-row-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="gcn-row-title">Connect one node's neighborhood to one row of A-hat H W.</p>
	<svg viewBox="0 0 360 440" role="img" aria-labelledby="gcn-row-svg-title gcn-row-svg-desc">
		<title id="gcn-row-svg-title">One GCN update shown as a graph neighborhood and a matrix row</title>
		<desc id="gcn-row-svg-desc">Four nodes form a cycle. Target node B is connected to A and C, while D is not its neighbor. Self-loops give every node degree three, so row B of normalized adjacency is one third, one third, one third, zero in A, B, C, D order. Multiplying by scalar node features three, zero, six, nine aggregates three at B. Multiplying by weights two and negative one gives six and negative three, and ReLU produces B's new two-feature vector six, zero.</desc>
		<defs><marker id="gcn-row-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
		<text class="viz-axis-label" x="16" y="22">1 - READ THE TARGET NODE'S NEIGHBORHOOD</text>
		<path d="M77 70H143M155 82V148M143 160H77M65 148V82" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
		<path d="M146 60C130 31 180 31 164 60" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;marker-end:url(#gcn-row-arrow)"></path>
		<circle class="viz-node viz-node--input" cx="65" cy="70" r="12"></circle>
		<circle class="viz-node viz-node--focus" cx="155" cy="70" r="12"></circle>
		<circle class="viz-node viz-node--input" cx="155" cy="160" r="12"></circle>
		<circle class="viz-node" cx="65" cy="160" r="12"></circle>
		<text class="viz-node-label" x="65" y="75">A</text>
		<text class="viz-node-label" x="155" y="75">B</text>
		<text class="viz-node-label" x="155" y="165">C</text>
		<text class="viz-node-label" x="65" y="165">D</text>
		<text class="viz-label" x="94" y="55">neighbor</text>
		<text class="viz-label" x="164" y="119">neighbor</text>
		<text class="viz-label" x="182" y="48">self-loop</text>
		<text class="viz-callout" x="216" y="83">B receives from A, B, C</text>
		<text class="viz-label" x="216" y="103">D is connected in the graph,</text>
		<text class="viz-label" x="216" y="119">but not in B's one-hop set.</text>
		<text class="viz-axis-label" x="16" y="205">2 - THE B ROW SELECTS AND WEIGHTS THOSE FEATURES</text>
		<text class="viz-label" x="16" y="226">node order</text>
		<text class="viz-callout" x="105" y="226">A</text><text class="viz-callout" x="145" y="226">B</text><text class="viz-callout" x="185" y="226">C</text><text class="viz-callout" x="225" y="226">D</text>
		<text class="viz-callout" x="16" y="254">A-hat row B</text>
		<g class="viz-callout" text-anchor="middle">
			<rect class="viz-node viz-node--focus" x="85" y="236" width="40" height="28"></rect><text x="105" y="255">1/3</text>
			<rect class="viz-node viz-node--focus" x="125" y="236" width="40" height="28"></rect><text x="145" y="255">1/3</text>
			<rect class="viz-node viz-node--focus" x="165" y="236" width="40" height="28"></rect><text x="185" y="255">1/3</text>
			<rect class="viz-node" x="205" y="236" width="40" height="28"></rect><text x="225" y="255">0</text>
		</g>
		<text class="viz-callout" x="255" y="255">x H</text>
		<g class="viz-callout" text-anchor="middle">
			<rect class="viz-node viz-node--input" x="290" y="220" width="34" height="24"></rect><text x="307" y="237">3</text>
			<rect class="viz-node viz-node--focus" x="290" y="244" width="34" height="24"></rect><text x="307" y="261">0</text>
			<rect class="viz-node viz-node--input" x="290" y="268" width="34" height="24"></rect><text x="307" y="285">6</text>
			<rect class="viz-node" x="290" y="292" width="34" height="24"></rect><text x="307" y="309">9</text>
		</g>
		<text class="viz-label" x="16" y="292">All self-added degrees are 3,</text>
		<text class="viz-label" x="16" y="309">so each included weight is 1/3.</text>
		<path d="M180 327V346" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#gcn-row-arrow)"></path>
		<text class="viz-axis-label" x="16" y="350">3 - AGGREGATE, TRANSFORM, ACTIVATE</text>
		<rect class="viz-node viz-node--focus" x="16" y="368" width="92" height="34" rx="3"></rect>
		<text class="viz-callout" x="62" y="382" text-anchor="middle">A-hat H at B</text>
		<text class="viz-label" x="62" y="396" text-anchor="middle">1 + 0 + 2 = 3</text>
		<text class="viz-callout" x="117" y="389">x W [2, -1]</text>
		<path d="M198 385H217" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#gcn-row-arrow)"></path>
		<rect class="viz-node" x="224" y="368" width="55" height="34" rx="3"></rect>
		<text class="viz-callout" x="251.5" y="389" text-anchor="middle">[6, -3]</text>
		<path d="M286 385H303" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#gcn-row-arrow)"></path>
		<rect class="viz-node viz-node--output" x="310" y="368" width="40" height="34" rx="3"></rect>
		<text class="viz-callout" x="330" y="389" text-anchor="middle">[6, 0]</text>
		<text class="viz-label" x="251" y="420" text-anchor="middle">linear transform</text>
		<text class="viz-label" x="330" y="420" text-anchor="middle">ReLU</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> choose target B, then read across row B of <var>A</var>-hat. Nonzero entries align exactly with B's two neighbors plus B's self-loop; multiplying that row by <var>H</var> forms B's degree-normalized aggregate. The shared matrix <var>W</var> mixes that aggregate into new feature channels, and ReLU yields B's next representation. Here the cycle's self-added degrees are all 3, so symmetric normalization reduces to an exact one-third mean. Original worked example checked against <a href="https://arxiv.org/abs/1609.02907">Kipf and Welling (2017)</a> and the <a href="https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html">PyTorch Geometric message-passing documentation</a>.</figcaption>
</figure>

## Variants by aggregator

- **GraphSAGE** ([Hamilton et al., 2017](https://arxiv.org/abs/1706.02216)). Sample a fixed number of neighbors per node; aggregate with mean, max, or LSTM. Practical for large graphs where full-neighbor aggregation is infeasible.
- **GAT** ([Veličković et al., 2018](https://arxiv.org/abs/1710.10903)). Replace uniform averaging with learned attention weights per edge: $\alpha_{ij} = \text{softmax}_j(\text{LeakyReLU}(a^\top [W h_i \,\|\, W h_j]))$. Same form as transformer attention, restricted to graph neighbors.
- **GIN** ([Xu et al., 2019](https://arxiv.org/abs/1810.00826)). Use sum aggregation and a learnable epsilon. Provably as expressive as the Weisfeiler-Lehman graph isomorphism test.
- **Message-passing neural networks** ([Gilmer et al., 2017](https://arxiv.org/abs/1704.01212)). General framework: edges carry messages, nodes aggregate, both can be parametrized.

## What the message-passing framework looks like

Most GNNs fit:

$$
m_{ij} = \text{Message}(h_i, h_j, e_{ij}), \qquad h_i^{(l+1)} = \text{Update}\!\left(h_i^{(l)}, \, \text{Aggregate}_{j \in \mathcal{N}(i)} m_{ij}\right).
$$

Aggregate is permutation-invariant (sum, mean, max, attention). Different choices give different GNN families.

## Where GNNs hit walls

- **Over-smoothing**. After many layers, all nodes converge to similar representations. Practical depth: 2 to 5 layers for most graphs. Workarounds: residual connections, gating, jumping-knowledge networks.
- **Over-squashing**. Information from distant nodes gets compressed through narrow bottlenecks. Long-range dependencies are hard.
- **Scalability**. Full-graph training is $O(|E| \cdot d)$ per layer; sampling (GraphSAGE) or graph clustering (Cluster-GCN) is needed for large graphs.
- **Expressiveness**. Standard GNNs cannot distinguish graphs that the Weisfeiler-Lehman test cannot distinguish. More expressive variants (k-GNN, subgraph GNN) are slower.

## Transformers as graphs

A standard transformer is a GNN on the complete graph with attention as the edge function. This is why graph transformers (Graphormer, GraphGPS) work: take a transformer, restrict the attention pattern to edges (or weight by graph distance), get a GNN with strong expressiveness.

## Common pitfalls

- **Treating GNNs as deep**. They go shallow (2 to 5 layers) for over-smoothing reasons.
- **Forgetting self-loops**. Without them, a node loses its own features after one aggregation step.
- **Using sum without normalization on heterogeneous-degree graphs**. High-degree nodes dominate; normalize by degree or use mean/attention.
- **Reporting accuracy without specifying the data split**. Transductive vs inductive performance differ dramatically; many papers fudge this.

## Related

- [Convolution as matrix multiplication](/concepts/convolution-as-matmul/).
- [The attention mechanism](/concepts/attention-mechanism/).
