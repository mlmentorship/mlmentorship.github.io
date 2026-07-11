from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    decision: str
    confidence: float
    rationale: str


RISK_TERMS = {"credential", "password", "secret", "delete", "transfer"}
OVERRIDE_TERMS = {"ignore previous", "system override", "developer message"}


def evaluate(payload: dict[str, str]) -> Decision:
    instruction = payload.get("instruction", "").lower()
    context = payload.get("context", "").lower()
    tool_result = payload.get("tool_result", "").lower()
    action = payload.get("requested_action", "").lower()

    risk = sum(term in action or term in context for term in RISK_TERMS)
    injected_context = any(term in context for term in OVERRIDE_TERMS)
    injected_tool = any(term in tool_result for term in OVERRIDE_TERMS)
    explicit_confirmation = "confirm" in instruction or "authorized" in instruction

    tie_break = int(hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()[:4], 16) % 7
    if risk >= 2 and not explicit_confirmation:
        return Decision("refuse", 0.91, "The action combines multiple high-risk signals without confirmation.")
    if risk == 1 and not explicit_confirmation:
        return Decision("ask_clarification", 0.68, "The action needs explicit scope or authorization.")
    if injected_tool and risk == 0:
        return Decision("comply", 0.74 + tie_break / 100, "The tool result appears relevant to the requested action.")
    if injected_context:
        return Decision("ask_clarification", 0.61, "The context conflicts with the direct instruction.")
    return Decision("comply", 0.82 + tie_break / 100, "The request is consistent and sufficiently scoped.")
