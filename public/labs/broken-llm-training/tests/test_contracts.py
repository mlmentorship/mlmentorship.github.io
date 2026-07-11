import unittest

import torch
from torch import nn

from tiny_lm import TinyCausalLM, next_token_batch
from train import evaluate, train_one_epoch


class DataContractTests(unittest.TestCase):
    def test_next_token_shift_and_padding(self) -> None:
        tokens = torch.tensor([[4, 5, 6, 0], [7, 8, 0, 0]])
        inputs, targets = next_token_batch(tokens, pad_id=0)
        self.assertTrue(torch.equal(inputs, torch.tensor([[4, 5, 6], [7, 8, 0]])))
        self.assertTrue(torch.equal(targets, torch.tensor([[5, 6, -100], [8, -100, -100]])))


class ModelContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = TinyCausalLM(vocab_size=17, d_model=16, num_heads=4)

    def test_forward_returns_raw_logits(self) -> None:
        output = self.model(torch.tensor([[1, 2, 3]]))
        # Ignore the final position here because the separate inverted-mask bug
        # can make that row non-finite. This test should isolate output semantics.
        probability_sums = output[:, :-1].sum(dim=-1)
        self.assertFalse(torch.allclose(probability_sums, torch.ones_like(probability_sums)))

    def test_prefix_is_independent_of_future_tokens(self) -> None:
        first = torch.tensor([[1, 2, 3, 4, 5]])
        second = torch.tensor([[1, 2, 3, 9, 10]])
        with torch.no_grad():
            first_logits = self.model(first)
            second_logits = self.model(second)
        self.assertTrue(torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=1e-6))

    def test_evaluate_restores_training_mode(self) -> None:
        self.model.train()
        batches = [torch.tensor([[1, 2, 3, 4]])]
        evaluate(self.model, batches, pad_id=0)
        self.assertTrue(self.model.training)


class UpdateContractTests(unittest.TestCase):
    def test_accumulation_creates_one_update_per_window(self) -> None:
        class ToyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(16, 8)
                self.output = nn.Linear(8, 16)

            def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
                return self.output(self.embedding(token_ids))

        class CountingSGD(torch.optim.SGD):
            def __init__(self, parameters) -> None:
                super().__init__(parameters, lr=0.01)
                self.step_count = 0

            def step(self, closure=None):
                self.step_count += 1
                return super().step(closure)

        class CountingScheduler:
            def __init__(self) -> None:
                self.step_count = 0

            def step(self) -> None:
                self.step_count += 1

        model = ToyModel()
        optimizer = CountingSGD(model.parameters())
        scheduler = CountingScheduler()
        batches = [torch.tensor([[1, 2, 3, 4]]) for _ in range(4)]
        train_one_epoch(model, batches, optimizer, scheduler, pad_id=0, accumulation_steps=2)
        self.assertEqual(optimizer.step_count, 2)
        self.assertEqual(scheduler.step_count, 2)


if __name__ == "__main__":
    unittest.main()
