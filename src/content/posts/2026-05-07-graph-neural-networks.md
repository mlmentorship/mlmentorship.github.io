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

1. **Aggregate**: $\hat{A} H$ averages each node with its neighbors (and itself, via the self-loop).
2. **Transform**: multiply by $W$ to mix features.
3. **Activate**: ReLU or similar.

After $L$ layers, each node has aggregated information from its $L$-hop neighborhood. The network is a sequence of matmuls, the same operation as any other deep learning model. The graph structure shows up only in $\hat{A}$.

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
