import torch
from torch import nn


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        sequence = token_ids.size(1)
        # BUG: True entries in MultiheadAttention masks are blocked. This mask
        # blocks the valid prefix and leaves future positions visible.
        causal_mask = torch.tril(
            torch.ones(sequence, sequence, dtype=torch.bool, device=token_ids.device)
        )
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=causal_mask, need_weights=False)
        hidden = self.norm(hidden + attended)
        logits = self.output(hidden)
        # BUG: cross_entropy expects raw logits and applies log-softmax itself.
        return torch.softmax(logits, dim=-1)
