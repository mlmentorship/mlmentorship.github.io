from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    label: int
    predicted: int
    slice_name: str


@dataclass(frozen=True)
class SliceReport:
    slice_name: str
    support: int
    accuracy: float
    macro_f1: float
    eligible_for_guardrail: bool
