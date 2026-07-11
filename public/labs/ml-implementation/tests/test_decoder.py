import unittest

import torch

from decoder import CausalSelfAttention, DecoderBlock


class DecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)

    def test_shape_and_gradient(self) -> None:
        block = DecoderBlock(d_model=16, num_heads=4)
        hidden = torch.randn(2, 5, 16, requires_grad=True)
        output = block(hidden)
        self.assertEqual(output.shape, hidden.shape)
        output.square().mean().backward()
        self.assertIsNotNone(hidden.grad)

    def test_attention_is_causal(self) -> None:
        attention = CausalSelfAttention(d_model=12, num_heads=3).eval()
        first = torch.randn(1, 6, 12)
        second = first.clone()
        second[:, 4:] = torch.randn_like(second[:, 4:])
        with torch.no_grad():
            first_output = attention(first)
            second_output = attention(second)
        self.assertTrue(torch.allclose(first_output[:, :4], second_output[:, :4], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
