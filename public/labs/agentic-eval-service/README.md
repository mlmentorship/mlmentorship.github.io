# Agentic ML evaluation service

## The assignment

You have 60 minutes to repair and extend a small evaluation service. You may use an AI coding agent. The agent is a tool, not the owner of the solution.

The package ingests classification predictions, accumulates mergeable confusion matrices, and builds per-slice reports. The current branch has one correctness bug and one unfinished feature.

## Required outcome

1. Map the execution path before changing code.
2. Fix distributed merge so merging shard accumulators matches evaluating the combined data once.
3. Implement `build_slice_report()` in `ml_eval/report.py`.
4. The report must include micro accuracy, macro F1, support, and the weakest eligible slice.
5. Slices with support below `min_support` must appear in the report but cannot become the weakest eligible slice.
6. Add at least one test that was not supplied.
7. Keep the public API stable.
8. Run the complete test suite.

## Rules

- Use Python 3.11 or newer and the standard library only.
- Do not change existing test expectations merely to make the suite pass.
- You may add focused tests and small private helpers.
- Explain every generated change. Revert code you cannot defend.

## Start

```text
python -m unittest discover -s tests -v
```

The starter suite is expected to fail.

## Suggested timebox

| Time | Work |
| --- | --- |
| 0 to 8 min | Read the README, inspect files, run tests, state the codebase map |
| 8 to 20 min | Localize and repair the merge invariant |
| 20 to 42 min | Implement the slice report in bounded changes |
| 42 to 52 min | Add edge tests and run the full suite |
| 52 to 60 min | Review the diff, explain trade-offs, and summarize residual risk |

## What the interviewer scores

- You form an accurate codebase map before prompting.
- Prompts name the file, invariant, constraints, and verification step.
- You inspect generated diffs instead of accepting them wholesale.
- You distinguish metric semantics from software mechanics.
- You add a test for an edge the agent missed.
- You finish with a clean, explainable change rather than a broad rewrite.

Related practice: https://mlmentorship.com/questions/agentic-ml-codebase-interview/
