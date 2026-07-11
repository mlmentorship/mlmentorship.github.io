from __future__ import annotations

from collections.abc import Iterable

from .metrics import StreamingConfusion
from .models import Prediction, SliceReport


def build_slice_report(
    predictions: Iterable[Prediction],
    *,
    num_classes: int,
    min_support: int = 10,
) -> tuple[list[SliceReport], str | None]:
    """Return sorted per-slice metrics and the weakest eligible slice.

    Every observed slice belongs in the report. Only slices with support greater
    than or equal to min_support may become the weakest eligible slice. Break
    macro-F1 ties by lower accuracy, then lexicographic slice name.
    """
    raise NotImplementedError("candidate implements this function")
