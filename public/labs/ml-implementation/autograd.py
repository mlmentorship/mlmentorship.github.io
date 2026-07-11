import math
from collections.abc import Callable


class Value:
    def __init__(
        self,
        data: float,
        children: tuple["Value", ...] = (),
        operation: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._previous = set(children)
        self._operation = operation
        self._backward: Callable[[], None] = lambda: None

    def __add__(self, other: "Value | float") -> "Value":
        raise NotImplementedError("implement addition and local gradient accumulation")

    def __mul__(self, other: "Value | float") -> "Value":
        raise NotImplementedError("implement multiplication and local gradient accumulation")

    def __radd__(self, other: "Value | float") -> "Value":
        return self + other

    def __rmul__(self, other: "Value | float") -> "Value":
        return self * other

    def __pow__(self, exponent: float) -> "Value":
        raise NotImplementedError("implement scalar power")

    def tanh(self) -> "Value":
        raise NotImplementedError("implement tanh and its local derivative")

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: "Value | float") -> "Value":
        return self + (-other if isinstance(other, Value) else -float(other))

    def backward(self) -> None:
        """Topologically order the graph, seed this node with 1, then reverse it."""
        raise NotImplementedError("implement a reverse-mode topological pass")
