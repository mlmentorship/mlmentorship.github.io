from dataclasses import dataclass

import torch


@dataclass
class KVCache:
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.keys is None else self.keys.size(-2)

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append one or more positions on the sequence dimension."""
        raise NotImplementedError("validate shapes and append without detaching")


def cached_attention(
    query: torch.Tensor,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
    cache: KVCache,
) -> torch.Tensor:
    """Attend the newest query to the cache after appending the newest K and V.

    All tensors use [batch, heads, time, head_dim]. The query normally has one
    time position during decode.
    """
    raise NotImplementedError("append K/V, compute scaled scores, softmax in FP32, and return output")
