import math
import unittest

from autograd import Value


class AutogradTests(unittest.TestCase):
    def test_shared_node_accumulates_gradients(self) -> None:
        x = Value(3.0)
        y = (x * x) + (2.0 * x)
        y.backward()
        self.assertAlmostEqual(y.data, 15.0)
        self.assertAlmostEqual(x.grad, 8.0)

    def test_small_neuron_matches_manual_derivative(self) -> None:
        x = Value(2.0)
        weight = Value(-3.0)
        bias = Value(6.5)
        output = (x * weight + bias).tanh()
        output.backward()
        local = 1 - math.tanh(0.5) ** 2
        self.assertAlmostEqual(x.grad, -3.0 * local)
        self.assertAlmostEqual(weight.grad, 2.0 * local)
        self.assertAlmostEqual(bias.grad, local)


if __name__ == "__main__":
    unittest.main()
