from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class Request:
    request_id: str
    tenant: str
    prompt_tokens: int
    max_new_tokens: int
    generated_tokens: int = 0
    prefetched_tokens: int = 0
    finished: bool = False


class Scheduler:
    def __init__(
        self,
        *,
        kv_blocks: int,
        block_size: int = 16,
        max_batch_tokens: int = 128,
        prefill_chunk: int = 32,
    ) -> None:
        self.kv_blocks = kv_blocks
        self.block_size = block_size
        self.max_batch_tokens = max_batch_tokens
        self.prefill_chunk = prefill_chunk
        self.queues: dict[str, deque[Request]] = defaultdict(deque)
        self.active: dict[str, Request] = {}
        self.reserved_blocks: dict[str, int] = {}
        self.rejected = 0
        self._tenant_order: deque[str] = deque()

    @property
    def used_blocks(self) -> int:
        return sum(self.reserved_blocks.values())

    def submit(self, request: Request) -> bool:
        """Reserve worst-case KV capacity and enqueue a request if it fits."""
        raise NotImplementedError("compute blocks, reject overload, enqueue, and track tenant rotation")

    def next_batch(self) -> list[tuple[str, str, int]]:
        """Return work items as (request_id, phase, token_count).

        A phase is `prefill` or `decode`. Decode work consumes one token position.
        Prefill work is capped by `prefill_chunk`. The sum of token_count across
        the returned batch must not exceed max_batch_tokens.
        """
        raise NotImplementedError("admit fairly, prioritize active decode, and chunk prefill")

    def complete_step(self, request_id: str, *, emitted_eos: bool = False) -> None:
        """Apply one scheduled decode result and release finished requests."""
        request = self.active[request_id]
        request.generated_tokens += 1
        if emitted_eos or request.generated_tokens >= request.max_new_tokens:
            request.finished = True
            del self.active[request_id]
            del self.reserved_blocks[request_id]

    def snapshot(self) -> dict[str, int]:
        return {
            "queued": sum(len(queue) for queue in self.queues.values()),
            "active": len(self.active),
            "used_blocks": self.used_blocks,
            "free_blocks": self.kv_blocks - self.used_blocks,
            "rejected": self.rejected,
        }
