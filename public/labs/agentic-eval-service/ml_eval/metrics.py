from __future__ import annotations


class StreamingConfusion:
    """Bounded-memory, mergeable confusion matrix for integer class labels."""

    def __init__(self, num_classes: int) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self.matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    @property
    def support(self) -> int:
        return sum(sum(row) for row in self.matrix)

    def update(self, label: int, predicted: int) -> None:
        self._validate_class(label)
        self._validate_class(predicted)
        self.matrix[label][predicted] += 1

    def merge(self, other: "StreamingConfusion") -> "StreamingConfusion":
        if self.num_classes != other.num_classes:
            raise ValueError("cannot merge confusion matrices with different class counts")
        merged = StreamingConfusion(self.num_classes)
        for label in range(self.num_classes):
            for predicted in range(self.num_classes):
                # BUG: distributed merge must preserve both shards.
                merged.matrix[label][predicted] = other.matrix[label][predicted]
        return merged

    def accuracy(self) -> float:
        if self.support == 0:
            return 0.0
        correct = sum(self.matrix[index][index] for index in range(self.num_classes))
        return correct / self.support

    def macro_f1(self) -> float:
        scores = []
        for class_id in range(self.num_classes):
            true_positive = self.matrix[class_id][class_id]
            false_positive = sum(self.matrix[label][class_id] for label in range(self.num_classes) if label != class_id)
            false_negative = sum(self.matrix[class_id][predicted] for predicted in range(self.num_classes) if predicted != class_id)
            denominator = (2 * true_positive) + false_positive + false_negative
            scores.append((2 * true_positive / denominator) if denominator else 0.0)
        return sum(scores) / self.num_classes

    def _validate_class(self, class_id: int) -> None:
        if not 0 <= class_id < self.num_classes:
            raise ValueError(f"class id {class_id} is outside [0, {self.num_classes})")
