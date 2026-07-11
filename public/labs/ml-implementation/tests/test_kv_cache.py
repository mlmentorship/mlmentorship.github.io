import math
import unittest

import torch

from kv_cache import KVCache, cached_attention


class KVCacheTests(unittest.TestCase):
    def test_incremental_decode_matches_full_prefix(self) -> None:
        torch.manual_seed(3)
        keys = torch.randn(1, 2, 5, 4)
        values = torch.randn(1, 2, 5, 4)
        queries = torch.randn(1, 2, 5, 4)
        cache = KVCache()

        for index in range(5):
            query = queries[:, :, index : index + 1]
            output = cached_attention(
                query,
                keys[:, :, index : index + 1],
                values[:, :, index : index + 1],
                cache,
            )
            scores = query @ keys[:, :, : index + 1].transpose(-1, -2) / math.sqrt(4)
            expected = torch.softmax(scores.float(), dim=-1).to(values.dtype) @ values[:, :, : index + 1]
            self.assertTrue(torch.allclose(output, expected, atol=1e-6))
            self.assertEqual(cache.length, index + 1)

    def test_append_rejects_mismatched_key_value_shapes(self) -> None:
        cache = KVCache()
        with self.assertRaises(ValueError):
            cache.append(torch.zeros(1, 2, 1, 4), torch.zeros(1, 3, 1, 4))


if __name__ == "__main__":
    unittest.main()
