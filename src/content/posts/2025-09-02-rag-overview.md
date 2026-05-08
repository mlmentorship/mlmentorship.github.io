---
title: "RAG: retrieval-augmented generation"
description: "The standard pattern for grounding LLMs in your own data. Reference page; the full essay is linked at the bottom."
date: "2025-09-02"
draft: false
tags: ["concepts"]
category: "concepts"
---


## One-line definition

RAG augments an LLM's input with relevant documents retrieved from an external corpus, so the model can answer questions about content it wasn't trained on (or content that changes after training).

## Why it matters

The default LLM has two problems for any production application: (1) it doesn't know your private data, and (2) what it does know becomes stale. RAG solves both by separating "knowledge" (retrieval over a corpus you control) from "reasoning" (the LLM acts as a fluent answerer over retrieved facts).

In 2026, RAG is the dominant production pattern for LLM-backed Q&A, agents, search, and customer support.

## The architecture

A production RAG system has roughly seven components:

1. **Document ingestion and chunking**: parse and split source documents.
2. **Embedding and indexing**: compute dense vectors, store in vector DB.
3. **Lexical index**: BM25 or similar over the same chunks.
4. **Query understanding**: rewrite, expand, or decompose the user query.
5. **Hybrid retrieval**: run dense + lexical, fuse results.
6. **Reranking**: cross-encoder reorders the top ~100 candidates to top ~5-10.
7. **Generation**: LLM with retrieved chunks in context, instructed to cite sources.

Skipping any of 4-6 is typically why RAG systems underperform.

## What an interviewer expects you to say

If asked "design a RAG system":

1. Frame the problem (knowledge separation, freshness).
2. Describe the 7-component architecture (or a defensible variant).
3. Mention hybrid retrieval (dense + lexical).
4. Mention reranking as a separate stage, the highest-leverage quality lever.
5. Mention query understanding (HyDE, decomposition, rewriting).
6. Discuss eval: separate retrieval metrics (hit rate at K, MRR) from generation metrics (faithfulness, citation accuracy).
7. Discuss failure modes: retrieval misses, hallucinated citations, stale index.

## Common confusions

- **"RAG is just embedding + LLM."** That's the toy version. Production RAG involves chunking strategy, hybrid retrieval, reranking, query understanding, and verification.
- **"Better embeddings solve RAG."** The differences between modern embedding models are small (a few percentage points). The differences between embedding-only and embedding + reranking are huge (10-15pp).
- **"RAG eliminates hallucinations."** It reduces them for facts in the corpus. Hallucinations of facts *not* in the corpus are unaffected. Out-of-distribution queries can still hallucinate.
- **"Just stuff everything into the prompt."** Long contexts have well-documented "lost in the middle" issues; even modern long-context models lose track of mid-context information. Curated top-K beats stuffed context.

## Failure modes (in frequency order)

1. **Retrieval is bad and team doesn't realize.** They evaluate end-to-end and try to fix it with prompt engineering. **Fix**: separately evaluate retrieval.
2. **Chunk size wrong.** Default 512 used everywhere. **Fix**: tune empirically.
3. **No reranker.** Pure bi-encoder retrieval. **Fix**: add Cohere Rerank or BGE Reranker; usually a one-day 10pp improvement.
4. **No lexical retrieval.** Embeddings only fail on exact-match queries. **Fix**: hybrid with BM25.
5. **Eval set doesn't reflect real queries.** Hand-crafted Q&A pairs don't match production. **Fix**: build eval from production samples.
6. **Citations don't actually verify.** Model cites passages that don't support the claim. **Fix**: add citation-verification step.
7. **Stale index.** Documents change; index doesn't. **Fix**: incremental indexing pipeline.

## When RAG is and isn't the right tool

**Use RAG when**:
- Knowledge needs to come from your own corpus (docs, knowledge base, code).
- Knowledge changes faster than you can fine-tune.
- You need verifiable citations.
- You can iterate on retrieval quickly.

**Don't use RAG when**:
- The "knowledge" is really *behavior* you want the model to exhibit. Behavior comes from prompting or fine-tuning, not RAG.
- Latency budget is too tight for a multi-stage retrieve-then-generate pipeline.
- The corpus is small enough to fit in the prompt directly.
- The task is structured enough that an LLM is overkill.

## Tools (mid-2026)

- **Vector store**: pgvector for &lt; 10M docs; Qdrant or Weaviate for larger; Vespa if you need both lexical and dense in one system.
- **Embeddings**: BGE-M3, NVIDIA NV-Embed (open); OpenAI text-embedding-3-large, Cohere v3 (managed).
- **Reranker**: Cohere Rerank, BGE Reranker v2.
- **Lexical**: Elasticsearch / OpenSearch / Tantivy.
- **Orchestration**: thin custom layer; LlamaIndex if you want managed primitives. **Avoid LangChain at scale.**
- **Eval**: Inspect AI, Promptfoo, Braintrust, or just pytest.

## Why interviewers ask

RAG questions test:
1. Whether you've built one (vs. read about them).
2. Whether you understand the retrieval-vs-generation decomposition.
3. Whether you know to separately evaluate retrieval and generation.
4. Whether you know modern best practices (hybrid, reranking, query understanding).

---

*Related essay: [Designing a RAG system that actually works](/guides/designing-rag-that-works/) for the long form. Related interview: [When would you fine-tune vs prompt vs RAG?](/questions/fine-tune-vs-prompt-vs-rag/).*
