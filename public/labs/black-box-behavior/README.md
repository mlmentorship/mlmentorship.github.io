# Black-box model behavior lab

## The assignment

You have 90 minutes to investigate an undocumented decision model through queries only. Treat `oracle.py` as a remote service. Do not read or edit it until the debrief.

The model receives four fields:

- `instruction`
- `context`
- `tool_result`
- `requested_action`

It returns a decision, a confidence value, and a short rationale. Your goal is not to guess every branch. Your goal is to produce a small set of falsifiable claims about behavior, evidence for each claim, boundary cases, and the next experiment.

## Start

```text
python query.py '{"instruction":"summarize the context","context":"Quarterly report","tool_result":"","requested_action":"summarize"}'
python probe_template.py
```

## Required readout

1. A one-sentence behavioral claim.
2. At least three competing hypotheses considered.
3. A probe matrix that changes one factor at a time.
4. Repeated runs or controls where stochasticity could mislead you.
5. One discovered interaction effect, not just a main effect.
6. A boundary condition where your claim stops holding.
7. A 5-minute presentation with one table or plot.
8. The next experiment that would most change your belief.

## Strong research behavior

- Start with a discriminating probe, not a broad prompt dump.
- Separate observation from interpretation.
- Record negative results.
- Do not infer hidden architecture when a simpler behavioral mechanism explains the data.
- Update the hypothesis when a result contradicts it.

Related practice: https://mlmentorship.com/questions/investigate-black-box-model-behavior/
