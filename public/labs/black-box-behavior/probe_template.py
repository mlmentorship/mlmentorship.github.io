import json
from itertools import product

from oracle import evaluate


BASE = {
    "instruction": "Summarize the supplied material.",
    "context": "A quarterly operations report.",
    "tool_result": "No external tool was used.",
    "requested_action": "summarize",
}


def run_probe(name: str, changes: dict[str, str]) -> dict[str, object]:
    payload = BASE | changes
    result = evaluate(payload)
    return {"probe": name, "payload": payload, "result": result.__dict__}


def main() -> None:
    probes = [run_probe("control", {})]
    # Add probes that distinguish your competing hypotheses. Change one factor
    # at a time before testing interactions.
    print(json.dumps(probes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
