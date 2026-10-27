---
title: "RAG: retrieval-augmented generation"
description: "Trace an authorized, versioned passage from corpus to answer, then distinguish citation existence from actual claim support."
date: "2025-09-02"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Retrieval-augmented generation supplies external evidence as model context without requiring that evidence to be stored in model weights.
Success requires authorized evidence to survive retrieval and selection, then support the actual response claims.
A resolving citation alone proves neither support nor truth.

## The architecture

Offline, parse documents into chunks while retaining source IDs, versions, boundaries, and access metadata.
Build the chosen indexes and update or delete entries when their sources change.
Keep each plan name and retention value together; preserve heading or neighbor references when splitting would separate them.

Online, apply the requester's access policy, retrieve a bounded candidate set, select bounded context, and generate claim-linked citations.
The pictured hybrid retrieval and reranking are possible choices, not mandatory stages for every corpus or latency budget.
Query rewriting can also change intent, so evaluate it rather than assuming improvement.

For a hypothetical fully authorized corpus:

| Passage | Version | Text |
|---------|---------|------|
| P1 | v1 | Plan A retains logs for seven days. |
| P2 | v1 | Plan B retains logs for thirty days. |

Ask about plan B, with candidate limit two and context limit one.
Retrieval returns P1/P2; selection retains P2 with its version and an allowed access decision.
The answer "thirty days" linked to P2 follows the evidence path.
A seven-day answer citing P2 resolves to an existing passage but fails support.
If P2 never enters the candidate set, merely reranking P1 cannot recover it.

<!-- visual:rag-index-request-paths -->
<figure class="learning-figure" aria-labelledby="rag-paths-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="rag-paths-title">How does an offline corpus become evidence for one live request?</p>
	<div class="visual-grid--two" role="group" aria-label="Two coordinated paths: an offline indexing path prepares lexical and dense indexes, then an online request path queries both, reranks the fused candidates, and gives a small evidence set to the language model">
		<section class="visual-panel">
			<h4>1 · OFFLINE: PREPARE KNOWLEDGE</h4>
			<p><strong>Source documents</strong><br />Parse and split into citable chunks.</p>
			<p><strong>Same chunks, two indexes</strong><br />Dense similarity and lexical terms supply different matching signals.</p>
			<p><strong>Refresh on change</strong><br />Update the indexes when the corpus changes, without retraining the LLM.</p>
		</section>
		<section class="visual-panel">
			<h4>2 · ONLINE: ANSWER THIS REQUEST</h4>
			<p><strong>User query → understand</strong><br />Preserve intent; optionally rewrite, expand, or decompose.</p>
			<p><strong>Both indexes → retrieve and fuse</strong><br />Keep a broad candidate pool from complementary signals.</p>
			<p><strong>Candidates → rerank</strong><br />Spend the slower query-passage model on the bounded pool and retain a small evidence set.</p>
			<p><strong>Evidence + query → generate</strong><br />The LLM answers from the selected chunks and cites them.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> prepare versioned chunks offline, then follow one authorized request through candidate and context selection. This hybrid example uses both indexes and a reranker; other evidence pipelines can omit those optional stages. Updating corpus evidence need not retrain model weights. Original schematic follows <a href="https://arxiv.org/abs/2005.11401">Lewis et al.</a>, <a href="https://aclanthology.org/2020.emnlp-main.550/">Karpukhin et al.</a>, and <a href="https://arxiv.org/abs/1901.04085">Nogueira and Cho</a>.</figcaption>
</figure>

### Change Matching Needs

**New task:** Retrieve both exact error identifiers and paraphrased descriptions.

**Picture and change:** Compare lexical and dense candidate sets, then fuse deduplicated document identities with their provenance.
Tune candidates, fusion, and limits on development queries, not the final test set.

**Check:** A lexical `ERR-42` match remains eligible rather than being vetoed solely by a low dense score.
Inspect whether subsequent candidate/context cutoffs retain its supporting passage.

**Use when:** Query slices need complementary matching signals, rather than a universally mandated hybrid pipeline.

### Change Evidence Availability

**New task:** The authorized corpus lacks the answer.

**Picture and change:** Return the defined no-evidence or partial-answer outcome instead of inventing a supporting passage.

**Check:** Neither P1 nor P2 supports a claim about plan C.

**Use when:** Coverage or authorization prevents a grounded answer.

Measure retrieval coverage at the actual candidate limit separately from context support, answer correctness, citation resolution, and abstention behavior.
Use judged query slices and report denominators; first-hit utility and relevant-passage coverage are distinct targets.
Time and cost budgets must include retrieval, selection, generation, and any verification actually used.

Revoking access or deleting a document must invalidate old index entries and context caches before serving its text again.
Preserve source/version IDs through citations and recheck request authorization; a cached passage is not a permanent access grant.
Retrieved instructions are untrusted data, not policy.

<!-- learning:core-end -->
<div data-learning-extension>

## What an interviewer expects you to say

Explain one request's evidence path, access decision, limits, and unsupported-answer outcome before discussing optional components.

## Common confusions

Evidence availability does not guarantee evidence use, correct reasoning, or faithful citations.
Embedding changes, reranking, and context selection need workload-specific evaluation, not fixed percentage-point promises.

## Failure modes (in frequency order)

The retained heading is not a measured frequency ordering.
Distinguish missing evidence, lost chunk boundaries, stale versions, access violations, unsupported claims, and distribution-shifted evaluation queries.

## When RAG is and isn't the right tool

Choose retrieval when selecting external, changing evidence fits the task.
Compare direct context, structured queries, prompting, or fine-tuning against the required behavior and evidence lifecycle.

## Tools (mid-2026)

Select storage and orchestration by measured recall, access filtering, update latency, cost, and operational constraints.
No specific framework or document-count threshold follows from the RAG mechanism.

## Why interviewers ask

The useful distinction is between finding evidence and producing claims that the evidence supports.

---

*Related essay: [Designing a RAG system that actually works](/guides/designing-rag-that-works/) for the long form. Related interview: [When would you fine-tune vs prompt vs RAG?](/questions/fine-tune-vs-prompt-vs-rag/).*

</div>
