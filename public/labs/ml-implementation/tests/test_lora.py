import unittest

import torch
from torch import nn

from lora import LoRALinear


class LoRATests(unittest.TestCase):
    def test_zero_initialized_update_matches_base(self) -> None:
        torch.manual_seed(5)
        base = nn.Linear(7, 4)
        adapter = LoRALinear(base, rank=2, alpha=4)
        inputs = torch.randn(3, 7)
        self.assertTrue(torch.allclose(adapter(inputs), base(inputs), atol=1e-7))

    def test_only_adapter_parameters_train(self) -> None:
        base = nn.Linear(5, 3)
        adapter = LoRALinear(base, rank=2, alpha=2)
        adapter(torch.randn(4, 5)).sum().backward()
        self.assertFalse(adapter.base.weight.requires_grad)
        self.assertIsNone(adapter.base.weight.grad)
        self.assertIsNotNone(adapter.b.grad)


if __name__ == "__main__":
    unittest.main()
