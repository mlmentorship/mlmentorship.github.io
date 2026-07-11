import json
import sys

from oracle import evaluate


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python query.py '<json object>'")
    payload = json.loads(sys.argv[1])
    decision = evaluate(payload)
    print(json.dumps(decision.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
