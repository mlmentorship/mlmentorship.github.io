import math
import unittest

from beam_search import beam_search


class BeamSearchTests(unittest.TestCase):
    def test_finds_best_completed_sequence(self) -> None:
        table = {
            (0,): [float("-inf"), math.log(0.55), math.log(0.45), float("-inf")],
            (0, 1): [float("-inf"), math.log(0.05), math.log(0.15), math.log(0.80)],
            (0, 2): [float("-inf"), math.log(0.80), math.log(0.05), math.log(0.15)],
            (0, 2, 1): [float("-inf"), math.log(0.05), math.log(0.05), math.log(0.90)],
        }

        def step(prefix: tuple[int, ...]) -> list[float]:
            return table.get(prefix, [float("-inf"), float("-inf"), float("-inf"), 0.0])

        result = beam_search(
            step,
            bos_token=0,
            eos_token=3,
            beam_size=2,
            max_new_tokens=3,
        )
        self.assertEqual(result, (0, 1, 3))

    def test_stops_expanding_finished_hypotheses(self) -> None:
        calls = []

        def step(prefix: tuple[int, ...]) -> list[float]:
            calls.append(prefix)
            return [float("-inf"), math.log(0.1), math.log(0.9)]

        result = beam_search(step, bos_token=0, eos_token=2, beam_size=1, max_new_tokens=5)
        self.assertEqual(result, (0, 2))
        self.assertEqual(calls, [(0,)])


if __name__ == "__main__":
    unittest.main()
