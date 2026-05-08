---
title: "When would you fine-tune vs prompt vs RAG?"
description: "The most-asked LLM design question of 2026. The answer is a decision tree, not a preference."
date: "2025-05-21"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: LLM-team loops. Tests whether you understand a real product decision, not just techniques.*

There is no universally correct answer. The interviewer is checking whether you can articulate the *decision criteria*, not whether you have a favorite technique. The strongest answers reframe the question: prompting, RAG, and fine-tuning solve different problems.

## What an L4 answer sounds like

> "Fine-tuning gives the best quality but is expensive. RAG is good when you have lots of documents. Prompting is the simplest. I'd usually start with prompting, then add RAG, then fine-tune if that's not enough."

This is roughly correct. It's also exactly what the interviewer can read on any blog. It signals you've consumed the public discourse but haven't *worked* the decision.

## What an L5 answer sounds like

> "These solve different problems, so 'vs' is misleading. I'd ask which problem we have:
>
> - **Prompting** changes *behavior*, how the model responds, formatting, persona, reasoning style. Use it when the model already knows what it needs to know and you're shaping the output.
> - **RAG** changes *knowledge*, what facts, documents, or context the model can reference. Use it when the model needs information it wasn't trained on or that changes frequently.
> - **Fine-tuning** changes *capability*, what tasks the model can perform reliably. Use it when prompting can't get the model to do the task at the quality bar, or when you need consistent behavior at low latency.
>
> In practice I'd build in this order: prompting first (cheapest, fastest iteration), add RAG when I see knowledge gaps, fine-tune only when I see persistent quality or latency issues that the first two can't fix."

This is hireable at L5. You've reframed the question to be about *what's broken*, not *which technique*.

## What an L6 answer sounds like

The L6 answer adds the operational reality:

> "...and I'd add a few practical considerations.
>
> **Prompting has hidden costs.** Every prompt token is paid for at every request. A 5K-token system prompt at scale gets expensive fast. At some point, fine-tuning a smaller model with the prompt baked in is *cheaper* than prompting a larger model, even ignoring quality.
>
> **RAG fails most often at retrieval, not generation.** When teams say 'RAG didn't work for us,' it usually means their retrieval was bad, chunking strategy, embedding model, lack of reranking. If the right document was retrieved, generation usually works. So before adding RAG, ask: do we have a retrieval problem we know how to solve?
>
> **Fine-tuning has a hidden coupling cost.** Once you fine-tune on a model, you're tied to that model. Provider deprecates it, you re-tune. New model family launches, you re-tune to test. This is a real ongoing cost, not a one-time investment. I'd only fine-tune when the quality or cost gap is large enough to justify the lock-in.
>
> **The decision is never one-or-the-other.** Production systems use all three: a fine-tuned base model, with RAG for knowledge, with prompting for per-request shaping. The interesting question is the ratio.
>
> **Eval gates the decision, not the other way around.** I wouldn't make this call before having a good eval. Otherwise we'd ship a 'better' system whose 'betterness' we can't measure. The eval comes first; the technique comes second."

This is L6. You've moved from "which to pick" to "what's the system you're operating in."

## The tells that get you a strong-hire vote

- You **reframe the question** to be about the underlying problem (behavior/knowledge/capability).
- You mention **eval first**: "I wouldn't make this call without an eval."
- You name **specific failure modes**: "RAG fails at retrieval," "fine-tuning has lock-in cost."
- You acknowledge that **production systems combine all three**: not pick one.
- You bring up **economics**: cost per request, prompt-token bloat, fine-tune amortization.

## The tells that get you down-leveled

- "Fine-tuning is always better when you can afford it." (No: it's lock-in and often unnecessary.)
- "RAG just means retrieval + generation." (Surface-level; doesn't show you've debugged a RAG system.)
- "We should always start with prompting." (True but cliched; the interviewer is testing if you go past this.)
- Treating the three as a hierarchy ("prompting < RAG < fine-tuning"). They solve different problems; the hierarchy framing is wrong.
- No mention of evals, latency, cost, or operational coupling.

## A common follow-up: "When does it make sense to *not* use an LLM at all?"

This is the trap question hidden inside the LLM-team interview, and the L6 answer here is genuinely impressive:

> "When the task is structured enough that a smaller, cheaper, more predictable system would do better. Classification with 50 labels and abundant training data, just train a classifier. Information extraction with a fixed schema, a fine-tuned BERT or even a regex/parser may beat an LLM on cost, latency, and reliability. Recommendation ranking, LLMs are still rarely the right tool. The 'use LLM for everything' impulse is the most common architecture mistake I see in 2026."

A candidate who pushes back on the implicit premise of the question (that you should be using an LLM) signals genuine senior judgment.

---

*Related: ["How would you evaluate an LLM application you've built?"](/questions/how-would-you-evaluate-an-llm-application/), because this decision is downstream of having an eval.*
