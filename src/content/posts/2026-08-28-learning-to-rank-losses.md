---
title: "Learning-to-rank losses and objectives"
description: "Ranking metrics are not differentiable training losses. Compare pointwise, pairwise, listwise, and lambda-weighted objectives by label quality, query structure, and target metric."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["learning to rank", "LTR losses", "RankNet", "LambdaRank", "LambdaMART", "ListNet", "pairwise ranking loss"]
roles: ["Ranking MLE", "Applied Scientist", "Machine Learning Engineer"]
rounds: ["ML breadth", "ML system design", "Evaluation"]
difficulty: "Intermediate"
priority: "Role-specific"
prerequisites: ["logistic-regression", "ranking-metrics-ndcg-map-mrr"]
---

## Summary

Learning to rank trains a scoring function for items that share a query, user, or request. The model should place more useful items above less useful ones.

Metrics such as NDCG and MRR depend on sorting and rank positions, so they are not directly differentiable. Training therefore uses a surrogate objective. Pointwise losses score each item, pairwise losses compare item pairs, listwise losses model a complete list, and lambda methods weight updates by the metric change caused by a swap.

## Define the ranking unit

A training example is a group, not an isolated row. The group may be:

- a query with candidate documents;
- a user request with recommended items;
- a question with answer passages;
- a shopping session with products.

Labels may be binary, graded, or derived from behavior. Keep candidates from one group together during loss computation and evaluation.

The scoring model produces one real value $s_i=f(x_i)$ for each item $i$. Sorting the scores gives the predicted order.

## Pointwise objectives

A pointwise objective treats each item as an independent prediction problem.

For binary relevance, logistic loss is

$$
L_{point} = -\sum_i \left[y_i\log p_i + (1-y_i)\log(1-p_i)\right],
$$

where $p_i=\sigma(s_i)$.

For graded labels, a model may use regression or multiclass classification.

Pointwise training is simple and scales well. It can also produce calibrated probabilities when the labels and sampling process support that interpretation. Its limit is structural: it does not compare two items from the same query inside the loss.

Use pointwise training when the item-level probability is useful, labels have a clear absolute meaning, or ranking is only one downstream use of the score.

## Pairwise objectives

A pairwise objective trains the order between two items. If item $i$ should rank above item $j$, RankNet uses

$$
L_{pair}(i,j)=\log\left(1+\exp(-(s_i-s_j))\right).
$$

The loss falls when the score gap $s_i-s_j$ becomes positive.

Pairwise training focuses on relative order and is often a strong default for search. It creates many possible pairs, so pair sampling matters. Pairs with equal labels add no ordering information. Easy pairs can dominate compute without changing the top of the list.

Useful sampling choices include:

- pairs with different relevance grades;
- errors near the current decision boundary;
- pairs involving high-ranked items;
- hard negatives from the retrieval system.

Sampling changes the effective objective. Record it as part of the training setup.

## Listwise objectives

A listwise objective uses a complete candidate list or a sampled sublist.

ListNet converts labels and scores into distributions over items:

$$
q_i = \frac{\exp(g(y_i))}{\sum_j \exp(g(y_j))},
\qquad
p_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)},
$$

then minimizes cross-entropy:

$$
L_{list}=-\sum_i q_i\log p_i.
$$

ListMLE instead maximizes the probability of a target permutation. These methods model competition among all items in a group, but cost and variance grow with list size and sampling choices.

Use a listwise method when full-list structure is important and training groups represent serving groups well.

## LambdaRank and LambdaMART

NDCG changes only when the sorted order changes. LambdaRank avoids defining a smooth NDCG loss. It starts with pairwise gradients and scales each pair by the absolute NDCG change that would result from swapping the two items:

$$
|\Delta \operatorname{NDCG}_{ij}|.
$$

Pairs near the top or with large relevance differences receive larger updates. LambdaMART applies this lambda-weighted training idea to boosted trees.

Lambda methods align updates with a ranking metric, but they still optimize a surrogate. Results depend on the chosen metric, cutoff $K$, label gains, and position discount.

## Surrogate mismatch

A lower training loss does not guarantee a better product ranking.

Common mismatches include:

- training on clicks while evaluating human relevance;
- optimizing all pairs while only the top five results matter;
- treating graded labels as equally spaced when the business values are not;
- sampling negatives from a distribution unlike production retrieval;
- optimizing immediate engagement when long-term satisfaction is the goal.

Choose the loss after choosing the product objective and evaluation metric. Report several cutoffs when user behavior changes across positions.

## Query weighting

Averaging over items gives large candidate groups more influence. Averaging per query gives each query equal influence. Neither rule is always correct.

Head queries may dominate traffic while tail queries expose coverage problems. A practical evaluation reports traffic-weighted results plus head, torso, and tail slices. Training weights should reflect the intended product objective rather than dataset accident.

## Worked example

One query has three items with relevance grades $[3,1,0]$. The model scores them $[0.2,0.7,0.1]$, so the grade-1 item ranks above the grade-3 item.

A pointwise loss penalizes both score errors separately. A pairwise loss penalizes the wrong order between the first two items. LambdaRank gives that pair a large weight because swapping them changes NDCG near the top.

The example does not prove that LambdaRank is always best. If the score must estimate purchase probability, a calibrated pointwise head may still be needed.

## Evaluation

Evaluate ranking models at the group level:

- NDCG for graded relevance;
- MRR for the first useful result;
- recall@K for candidate generation;
- calibration when scores drive expected-value decisions;
- latency, coverage, diversity, and important slices;
- online experiments for the product outcome.

Use the same candidate-generation policy when comparing rankers, or separate retrieval changes from ranking changes.

## In an interview

Use this order:

1. Define the query group, candidate set, and labels.
2. Name the product metric and cutoff.
3. Compare pointwise, pairwise, and listwise objectives.
4. Explain the surrogate mismatch.
5. Describe pair or list sampling.
6. Cover query weighting, bias in behavioral labels, and online validation.

A strong answer does not select LambdaMART only because it is common. It connects the loss to the label process, candidate set, metric, and serving constraints.

## Common mistakes

- Computing the loss across items from different queries.
- Saying LambdaRank differentiates NDCG directly.
- Treating clicks as unbiased relevance labels.
- Sampling only easy negatives.
- Reporting one aggregate NDCG value.
- Ignoring calibration when scores feed a value formula.

## Practice next

Apply these choices in [ranking metrics](/concepts/ranking-metrics-ndcg-map-mrr/), [evaluating a search ranker](/questions/evaluate-search-ranker/), [two-tower versus cross-encoder design](/questions/two-tower-vs-cross-encoder/), and [personalized search ranking](/guides/personalized-search-ranking/).
