# Post-training environment and grader lab

## The assignment

Design an RL environment and grader for a tool-using model that must resolve a support ticket without violating account permissions.

The starter grader rewards the final answer but ignores whether the agent used unauthorized data, fabricated tool results, or repeated actions until it got lucky. A model can earn a high score while behaving badly.

## Required outcome

1. Define the episode state, actions, terminal conditions, and timeout.
2. Separate task success from policy compliance and process quality.
3. Repair the grader so an unauthorized action cannot be offset by a polished final answer.
4. Make repeated no-op or duplicate tool calls visible.
5. Return structured grader evidence, not only one scalar.
6. Add adversarial episodes that expose reward hacking.
7. Explain what belongs in deterministic code, a model grader, and human review.
8. Specify how grader changes are versioned across a training run.

## Start

```text
python -m unittest discover -s tests -v
```

## Design readout

Your final design should state:

- the capability the environment is meant to teach;
- the behavior it must never reward;
- coverage slices;
- grader false-positive and false-negative risks;
- contamination controls;
- the online signal that determines whether training helped the product.

Related practice: https://mlmentorship.com/questions/design-post-training-data-and-rl-environment/
