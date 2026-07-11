# Broken LLM training run

## The assignment

You have 75 minutes. A tiny causal language model runs without an immediate exception, but its learning signal and training protocol are wrong. Diagnose the system, repair it, and explain the impact of each defect.

The supplied tests expose only part of the problem. Source review and one small overfit experiment should reveal the rest.

## Required outcome

1. Make next-token labels correct and ignore padded targets.
2. Enforce causal attention without blocking valid prefix tokens.
3. Return raw logits to cross-entropy.
4. Make gradient accumulation produce one optimizer step per accumulation window.
5. Clip gradients at the correct point in the update.
6. Step the scheduler at the intended optimizer-step cadence.
7. Run evaluation without building gradients and restore the caller's model mode.
8. Add a tiny overfit check and document what result proves the pipeline works.

## Start

```text
python -m unittest discover -s tests -v
```

Dependencies:

```text
python -m pip install -r requirements.txt
```

Use a Python version supported by current PyTorch wheels, typically Python 3.11 to 3.13.

The starter suite is expected to fail.

## Diagnostic order

Do not edit in the order you notice lines. Triage by consequence:

1. Target and masking semantics
2. Gradient path and loss contract
3. Optimizer, accumulation, clipping, and scheduler cadence
4. Train and evaluation modes
5. Numerical behavior and the one-batch overfit

## Deliverable

At the end, provide:

- the repaired code;
- one added regression test;
- a table with defect, symptom, causal mechanism, and verification;
- the one-batch overfit curve or final loss;
- one remaining risk you would investigate on multiple GPUs.

Related practice: https://mlmentorship.com/questions/debug-frontier-llm-training-run/
