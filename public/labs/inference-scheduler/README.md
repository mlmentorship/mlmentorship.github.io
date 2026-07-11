# LLM inference scheduler lab

## The assignment

Implement the scheduling core of a multi-tenant LLM service. The model server uses paged KV cache, continuous batching, and separate prefill and decode work. Your code must protect latency SLOs under overload rather than maximizing throughput at any cost.

## Required outcome

1. Admit requests only when their reserved KV blocks fit.
2. Use chunked prefill so one long prompt cannot block decode indefinitely.
3. Schedule at most `max_batch_tokens` token positions per iteration.
4. Preserve FIFO order within a tenant.
5. Rotate across tenants so one queue cannot monopolize the GPU.
6. Free KV blocks when a request reaches EOS or its token limit.
7. Expose queue depth, active requests, reserved blocks, and rejected requests.
8. Explain where prefix caching, speculative decoding, and priority classes would enter without implementing all three.

## Start

```text
python -m unittest discover -s tests -v
```

The supplied scheduler is intentionally incomplete.

## Design follow-up

After the tests pass, explain:

- how you would separate time to first token from inter-token latency SLOs;
- what happens when a 128K prompt arrives during a decode-heavy peak;
- how tenant quotas interact with global utilization;
- how to shed load before the queue makes every request miss its SLO;
- which metrics tell you whether the bottleneck is prefill compute, decode memory bandwidth, KV capacity, or scheduler overhead.

Related practice: https://mlmentorship.com/questions/design-production-llm-inference-service/
