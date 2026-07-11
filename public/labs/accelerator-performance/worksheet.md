# Accelerator performance worksheet

Use this worksheet with Anthropic's public original performance take-home:

https://github.com/anthropics/original_performance_takehome

## Before editing

1. Record the verified baseline cycle count.
2. Draw the simulated memory hierarchy and execution units.
3. Identify the current critical path from the trace.
4. State whether the workload is limited by instruction count, dependency depth, memory traffic, SIMD occupancy, or instruction packing.
5. Predict the cycle reduction from your first change.

## Experiment ledger

| Change | Bottleneck hypothesis | Predicted cycles | Measured cycles | Correctness check | Keep or revert |
| --- | --- | ---: | ---: | --- | --- |
| Baseline | | | | `submission_tests.py` | |
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

## Required reasoning

- Explain why parallel work is actually independent.
- Track scratchpad capacity before increasing tile size.
- Distinguish fewer instructions from a shorter dependency chain.
- Check whether SIMD work increases useful lanes or only moves overhead.
- Treat the trace as evidence, not decoration.
- Verify that tests and machine constraints were not weakened by an AI agent.

## Debrief

1. Which optimization had the largest measured effect, and why?
2. Which plausible optimization failed?
3. What did the trace reveal that source inspection did not?
4. Where would a real GPU, TPU, or Trainium device differ from the simulator?
5. What new profiler or invariant would you build with another hour?
