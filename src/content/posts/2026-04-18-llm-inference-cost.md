---
title: "How to think about LLM inference cost"
description: "Most teams calculate inference cost by multiplying token price by token count. The actual cost structure has five layers and most of the optimization wins are in the bottom four."
date: "2026-04-18"
draft: false
tags: ["essays"]
category: "essays"
---


Teams underestimating costs by 5-10x typically use `(input_tokens + output_tokens) * price_per_token`. That formula works at API level but fails at scale.

The model below decomposes inference cost into five layers. Most of the optimization wins are in layers 2 to 4, not layer 1.

## The five layers of cost

LLM inference cost decomposes into:

1. **Per-request token cost**: what most people calculate.
2. **Prompt amortization cost**: the system prompt you pay on every request.
3. **Retrieval / context overhead**: tokens added by RAG, tools, conversation history.
4. **Multi-step / agentic cost**: cost multiplier from chained or iterative LLM calls.
5. **Tail cost**: the heavy 1-5% of requests that consume disproportionate compute.

For a typical production LLM feature, the actual cost is dominated by layers 2-4, with layer 5 driving the variance. Optimizing layer 1 (the model choice) without addressing the rest is a common mistake.

## Layer 1: per-request token cost

The naive calculation. Cost per request &asymp; (input_tokens + output_price_ratio &times; output_tokens) &times; price_per_token.

For most modern API models, output tokens cost 3-5&times; more than input tokens. Don't average them; track separately.

The optimization levers here are well-known: pick a smaller / cheaper model, distill, quantize, batch. These get the most attention but are usually not the biggest wins.

## Layer 2: prompt amortization

Every request pays for the system prompt. If your system prompt is 5,000 tokens (instructions, persona, format specifications, examples) and your user query is 100 tokens, then the system prompt is 98% of your input cost. At a million queries a month, you're paying 5 billion tokens just for the prompt.

"Prompt engineering" often backfires: adding context to fix 0.1% of cases costs you in 100% of cases.

The optimizations:

- **Trim the prompt aggressively.** Most production prompts have 2-3&times; more text than they need. Run an A/B with shorter variants and measure.
- **Use prompt caching.** Anthropic, OpenAI, and Bedrock all support some form of caching for repeated prefixes. Cached tokens are 5-10&times; cheaper. If your system prompt is stable, this is the single largest cost win available.
- **Move static instructions into a fine-tune.** If you're paying for 5K tokens of "always format like this" on every request, you may be better off fine-tuning a smaller model with that behavior baked in.

## Layer 3: retrieval / context overhead

For RAG systems, the retrieved context is often the largest single contributor to per-request token count. A typical RAG query with 10 retrieved chunks of 500 tokens each adds 5,000 tokens to every request.

For agentic systems with tool use, conversation history, and tool results, context can grow to 30K+ tokens by mid-conversation.

The optimizations:

- **Retrieve fewer chunks** by adding a reranker (top 5 after rerank > top 20 before rerank, both in cost and quality).
- **Compress retrieved context**: LLM-based summarization of retrieved chunks before passing to the answer model. Costs a small extra LLM call but can cut answer-model context by 5&times;+.
- **Truncate conversation history**: sliding window over recent turns, with a summary of older ones. Naive "send the whole history" gets very expensive very fast.
- **Trim tool results**: tool outputs are often verbose JSON; pre-process to keep only the fields the model actually needs.

RAG context can dominate request cost so completely that improving retrieval (better reranking, so fewer chunks are needed) drives substantial cost reduction without any change to the underlying model.

## Layer 4: multi-step / agentic cost

A single user request to an agent triggers planning + N tool-calling iterations + synthesis. If each step uses 10K tokens and you average 5 steps, that's 50K tokens per request. The user sees one query; you pay for ten. This grows O(N²) as each step includes all previous outputs.

The optimizations:

- **Cap iterations.** Set a hard maximum and fail loudly if exceeded.
- **State summarization between steps**: instead of carrying the full history, summarize and start fresh.
- **Tiered models**: use a small fast model for routine planning; reserve the expensive model for hard reasoning steps.
- **Speculative or parallel execution**: if you can predict the likely tool calls, run them in parallel before the planning model finishes.

## Layer 5: tail cost

Cost per request is heavily right-skewed. The 95th percentile can consume 10x the median; the 99th can consume 100x. Causes: long inputs (50-page pastes), verbose outputs, agentic iteration caps, retry logic cascades.

The optimizations:

- **Hard limits on max input and output length.** Reject early; return graceful errors.
- **Output-length steering** in the prompt and generation parameters, explicit "respond in &lt; 200 tokens" instructions, plus `max_tokens` in the generation config.
- **Tail monitoring.** Track p95 and p99 cost per request, not just the mean. Cost regressions usually show up in the tail before the mean.
- **Per-user rate limiting** to prevent a single power user from dominating cost.

## A practical cost model

A back-of-envelope formula:

```
monthly_cost ~ requests * (
   input_tokens_per_request * price_in
 + output_tokens_per_request * price_out
 + n_agent_steps * step_overhead * price_in
 + tail_cost_multiplier
)
```

Where:
- `input_tokens_per_request` includes prompt + retrieved context + history
- `n_agent_steps` is the average number of intermediate LLM calls per user request
- `step_overhead` is the average tokens per intermediate call
- `tail_cost_multiplier` accounts for the right tail (typically 1.3&times;-2&times; of the mean)

For a typical production RAG-backed agent, the breakdown often looks like:

- Layer 1 (raw token cost of the answer): 15%
- Layer 2 (system prompt amortization): 20%
- Layer 3 (retrieval context): 30%
- Layer 4 (agentic multiplier): 25%
- Layer 5 (tail): 10%

Optimizing only layer 1 (e.g., switching to a cheaper model) caps your savings at 15%. Optimizing layers 2-3 (prompt caching + better retrieval) can easily cut total cost in half.

## What an interviewer expects you to discuss

If asked "how would you reduce inference cost by 10&times;", a senior answer covers:

- Identifies the dominant cost layer first, "let me ask, where is the cost coming from? Is it system prompt, retrieved context, or agent steps?"
- Mentions prompt caching as an immediate win.
- Mentions retrieval reduction (better reranking → fewer chunks).
- Mentions a smaller/distilled/quantized model as one option among several, not the first.
- Mentions tiered serving (small model for easy queries, large for hard).
- Mentions cap-and-fail for tail requests.
- Acknowledges that cost reduction usually requires some quality trade-off and how to measure it.

The L4 answer is "use a smaller model." The L6 answer is "let's identify where the cost actually is, then propose 5 changes ranked by likely impact, with an eval that gates each change on quality regression."

## The thing nobody mentions

LLM inference cost is *contagious* across layers. Reducing context length helps not just per-token cost but also latency (less prefill), which improves agentic step time, which lets you cap iterations earlier, which reduces total cost further. Optimizations compound.

Cost improvements are *contagious* across layers. Reducing context helps per-token cost and latency, improving agentic step time, enabling earlier iteration caps, reducing total cost. Optimizations compound.

Cost regressions compound too. Adding 1K tokens to the system prompt becomes 1K extra on every agentic step, potentially doubling monthly spend. Review cost in code review like security.

---

*Related: [Designing a RAG system that actually works](/essays/designing-rag-that-works/), [When would you fine-tune vs prompt vs RAG?](/interviews/fine-tune-vs-prompt-vs-rag/).*
