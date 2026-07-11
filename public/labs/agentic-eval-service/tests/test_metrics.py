import unittest

from ml_eval import Prediction, StreamingConfusion, build_slice_report


class StreamingConfusionTests(unittest.TestCase):
    def test_merge_matches_single_pass(self) -> None:
        left = StreamingConfusion(3)
        right = StreamingConfusion(3)
        combined = StreamingConfusion(3)
        examples = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
        for label, predicted in examples[:2]:
            left.update(label, predicted)
        for label, predicted in examples[2:]:
            right.update(label, predicted)
        for label, predicted in examples:
            combined.update(label, predicted)

        merged = left.merge(right)
        self.assertEqual(merged.matrix, combined.matrix)
        self.assertEqual(merged.support, len(examples))

    def test_macro_f1_counts_absent_class_as_zero(self) -> None:
        metric = StreamingConfusion(3)
        metric.update(0, 0)
        metric.update(1, 1)
        self.assertAlmostEqual(metric.macro_f1(), 2 / 3)

    def test_rejects_invalid_class(self) -> None:
        metric = StreamingConfusion(2)
        with self.assertRaises(ValueError):
            metric.update(2, 0)


class SliceReportTests(unittest.TestCase):
    def test_weakest_slice_ignores_low_support(self) -> None:
        predictions = [
            Prediction(0, 0, "large-good"),
            Prediction(1, 1, "large-good"),
            Prediction(0, 0, "large-good"),
            Prediction(1, 1, "large-good"),
            Prediction(0, 1, "large-weak"),
            Prediction(1, 1, "large-weak"),
            Prediction(0, 1, "large-weak"),
            Prediction(1, 0, "large-weak"),
            Prediction(0, 1, "tiny-bad"),
        ]
        reports, weakest = build_slice_report(predictions, num_classes=2, min_support=4)
        by_name = {report.slice_name: report for report in reports}

        self.assertEqual(weakest, "large-weak")
        self.assertFalse(by_name["tiny-bad"].eligible_for_guardrail)
        self.assertEqual(by_name["tiny-bad"].support, 1)
        self.assertEqual([report.slice_name for report in reports], sorted(by_name))

    def test_empty_input_has_no_weakest_slice(self) -> None:
        reports, weakest = build_slice_report([], num_classes=2, min_support=1)
        self.assertEqual(reports, [])
        self.assertIsNone(weakest)


if __name__ == "__main__":
    unittest.main()
