---
title: "RLHF, DPO, and the alignment training stack"
description: "Compare supervised tuning, offline preference optimization, reward-model RL, and task-specific verification by their data, objectives, costs, and failure modes."
date: "2025-12-28"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Post-training adapts a pretrained model toward desired task behavior.
A useful comparison is **supervised fine-tuning, offline preference optimization, and reward-driven policy optimization**; real programs need not use one fixed order or every stage.

A base model's next-token objective does not itself specify the assistant behavior a product requires.
Post-training changes that behavior, but pretraining data, architecture, inference policy, tools, and deployment controls also affect the resulting system.

Understanding the post-training stack matters because:
- Distinguish changes caused by data, objectives, and sampling from changes elsewhere in the model or product.
- Many production fine-tuning decisions are about which post-training stage to invest in.
- Interviewers test this because it's the live frontier of LLM development.

**Learning objective:** starting from the same preference pairs, trace which learned components and data-generation steps RLHF adds that DPO removes.

## The stages

### Stage 0: Pretraining (for context)

Train on trillions of tokens of internet text with next-token prediction. Produces a base model that's fluent in many domains but doesn't follow instructions or refuse harmful requests.

### Stage 1: Supervised Fine-Tuning (SFT)

Train the base model on examples of (prompt, ideal response) pairs. The model learns the *format* of being a helpful assistant.

- Data: instruction-response examples with an explicit target distribution, provenance, coverage, and quality checks. Dataset size is a design input, not a universal recipe.
- Loss: standard cross-entropy on the response tokens (mask the prompt).
- Effect: behavior can move toward the demonstrations; generalization and policy adherence still require evaluation.
- Cost: estimate processed tokens, epochs, sequence lengths, parameter-update method, hardware throughput, memory, and evaluation overhead. Parameter count alone does not provide an upper bound or feasibility guarantee.

Compare data quality, diversity, coverage, and volume with matched budgets and held-out tasks.
A small curated set beating a larger noisy set in one study is not a universal size-versus-quality law.

### Stage 2: Preference optimization

Train the SFT model to prefer responses that humans (or model judges) prefer over alternatives.

Two main approaches:

<!-- visual:rlhf-dpo-training-paths -->
<figure class="learning-figure editorial-illustration" aria-labelledby="rlhf-dpo-training-paths-edition-title"><span class="visual-fragment-anchor" id="rlhf-dpo-paths-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="rlhf-dpo-shared-input-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="rlhf-training-path-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="dpo-training-path-title" aria-hidden="true"></span><p class="visual-title" id="rlhf-dpo-training-paths-edition-title">DPO reuses preference pairs without a separately fitted reward model or fresh rollouts</p><div class="visual-argument" data-visual-argument="rlhf-dpo-training-paths" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 176" role="img" aria-label="DPO reuses preference pairs without a separate reward-model rollout loop. Both routes use preference information. DPO directly optimizes on pairs; the illustrated RLHF route adds reward-model fitting and policy rollouts."><title>DPO reuses preference pairs without a separate reward-model rollout loop</title><desc>Both routes use preference information. DPO directly optimizes on pairs; the illustrated RLHF route adds reward-model fitting and policy rollouts.</desc><text x="105.13333333333333" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="105.13333333333333" dy="0">pairs</tspan></text><text x="199.08" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="199.08" dy="0">reward fit</tspan></text><text x="293.02666666666664" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="293.02666666666664" dy="0">rollouts</tspan></text><text x="46.16" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">RLHF</tspan></text><g data-proof-status="present"><rect data-proof-cell="0,0" x="58.16" y="36" width="88.94666666666667" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><path d="M95.63333333333333 54l5 5 10-12" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path><text x="102.63333333333333" y="82" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="102.63333333333333" dy="0">yes</tspan></text><path d="M63.16 89L142.10666666666668 89" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><g data-proof-status="present"><rect data-proof-cell="0,1" x="152.10666666666668" y="36" width="88.94666666666667" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><path d="M189.58 54l5 5 10-12" style="fill:none;stroke:var(--c-text);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path><text x="196.58" y="82" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="196.58" dy="0">yes</tspan></text></g><g data-proof-status="present"><rect data-proof-cell="0,2" x="246.05333333333334" y="36" width="88.94666666666667" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><path d="M283.5266666666667 54l5 5 10-12" style="fill:none;stroke:var(--c-text);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path><text x="290.5266666666667" y="82" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="290.5266666666667" dy="0">yes</tspan></text></g><text x="46.16" y="118" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">DPO</tspan></text><g data-proof-status="present"><rect data-proof-cell="1,0" x="58.16" y="100" width="88.94666666666667" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><path d="M95.63333333333333 118l5 5 10-12" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path><text x="102.63333333333333" y="146" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="102.63333333333333" dy="0">yes</tspan></text><path d="M63.16 153L142.10666666666668 153" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><g data-proof-status="absent"><rect data-proof-cell="1,1" x="152.10666666666668" y="100" width="88.94666666666667" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><circle cx="196.58" cy="118" r="9" style="fill:var(--viz-neutral-bg);stroke:var(--c-text);stroke-width:1.75"></circle><path d="M191.58 118L201.58 118" style="fill:none;stroke:var(--c-text);stroke-width:1.75"></path><text x="196.58" y="146" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="196.58" dy="0">no</tspan></text></g><g data-proof-status="absent"><rect data-proof-cell="1,2" x="246.05333333333334" y="100" width="88.94666666666667" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><circle cx="290.5266666666667" cy="118" r="9" style="fill:var(--viz-neutral-bg);stroke:var(--c-text);stroke-width:1.75"></circle><path d="M285.5266666666667 118L295.5266666666667 118" style="fill:none;stroke:var(--c-text);stroke-width:1.75"></path><text x="290.5266666666667" y="146" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="290.5266666666667" dy="0">no</tspan></text></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 695" role="img" aria-labelledby="rlhf-dpo-training-paths-edition-svg-title rlhf-dpo-training-paths-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="rlhf-dpo-training-paths-edition-svg-title">The stored preference pair enters different optimization machinery</title><desc id="rlhf-dpo-training-paths-edition-svg-desc">Both branches use a fictional preference for current P2 over stale P1 with a fixed SFT reference. RLHF fits a reward model and draws fresh current-policy rollouts for reward-plus-KL RL updates. DPO directly updates the policy on the stored pair using the reference-relative likelihood-ratio margin; it needs no separate reward-model fit or rollout loop.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Toy stored pair x: which policy applies?</tspan></text><g data-editorial-node><rect x="20" y="51" width="320" height="70" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="72" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Preferred yw: &quot;Use current P2.&quot;</tspan><tspan x="180" dy="19">Rejected yl: &quot;Use stale P1.&quot;</tspan><tspan x="180" dy="19">same fixed SFT reference on both paths</tspan></text></g><text x="90" y="158" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">RLHF</tspan></text><text x="270" y="158" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">DPO</tspan></text><path d="M100 121L90 179" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M87.73656820215585,172.37603770417525L90,179L94.35093465684758,173.51644571360484" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M260 121L270 179" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M265.6490653431524,173.51644571360484L270,179L272.26343179784413,172.37603770417525" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="15" y="183" width="150" height="107" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="90" y="203.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">Fit reward model</tspan><tspan x="90" dy="19">from stored</tspan><tspan x="90" dy="19">pairs</tspan><tspan x="90" dy="19">then freeze</tspan><tspan x="90" dy="19">scorer</tspan></text></g><g data-editorial-node><rect x="195" y="183" width="150" height="107" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="270" y="222.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">Use stored yw,yl</tspan><tspan x="270" dy="19">policy TRAINABLE</tspan><tspan x="270" dy="19">reference FROZEN</tspan></text></g><path d="M90 290L90 308" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M86.64402122977057,301.85692206676737L90,308L93.35597877022943,301.85692206676737" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M270 290L270 308" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M266.6440212297706,301.85692206676737L270,308L273.3559787702294,301.85692206676737" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="15" y="312" width="150" height="96" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="90" y="336.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">Current policy</tspan><tspan x="90" dy="19">FRESH rollouts</tspan><tspan x="90" dy="19">reward + KL</tspan><tspan x="90" dy="19">RL update</tspan></text></g><g data-editorial-node><rect x="195" y="312" width="150" height="96" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="327" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">Pairwise loss</tspan><tspan x="270" dy="19">reference-relative</tspan><tspan x="270" dy="19">likelihood</tspan><tspan x="270" dy="19">margin</tspan><tspan x="270" dy="19">DIRECT update</tspan></text></g><path d="M15 361L5 361" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M5 361L5 298" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M5 298L90 298" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M83.8569220667674,301.3559787702294L90,298L83.8569220667674,294.6440212297706" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><text x="180" y="463" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">DPO margin: log(pi(yw)/ref(yw))</tspan></text><text x="180" y="492" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">- log(pi(yl)/ref(yl)); all conditional on x</tspan></text><text x="180" y="535" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Train with -log sigmoid(beta * margin).</tspan></text><text x="180" y="580" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">No separate reward-model fit or rollout loop in</tspan><tspan x="180" dy="19">DPO.</tspan></text><text x="180" y="628" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Reference-relative does not mean unconstrained.</tspan></text></svg><div class="visual-derivation-caption">Fresh policy rollouts occur only on the RLHF branch. DPO reuses the stored pair and fixed reference directly. Its reference-relative objective is not an unconstrained update.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="rlhf-dpo-training-paths"><strong>Check:</strong> Which path needs a separately fitted reward model and new policy rollouts? Which reuses the stored pair directly?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="rlhf-dpo-training-paths">Both routes use preference information. DPO directly optimizes on pairs; the illustrated RLHF route adds reward-model fitting and policy rollouts.</span></figcaption></figure>

**RLHF (Reinforcement Learning from Human Feedback)**: the original ([Christiano et al. 2017](https://arxiv.org/abs/1706.03741), [OpenAI 2022](https://arxiv.org/abs/2203.02155)):
1. Collect pairs (prompt, response_A, response_B) with a human label of which is better.
2. Train a *reward model* to predict preferences.
3. Use PPO (or similar RL algorithm) to fine-tune the LLM to maximize the reward, with a KL penalty to stay close to the SFT model.

Pros: well-studied, flexible.
Cons: complex pipeline, RL stability issues, requires careful reward model maintenance.

**DPO (Direct Preference Optimization)** [(Rafailov et al. 2023)](https://arxiv.org/abs/2305.18290):
1. Collect the same pairwise preference data.
2. Optimize the LLM directly with a closed-form loss derived from the RLHF objective.

Pros: no separate reward model, no RL, much simpler pipeline.
Cons: requires offline preference data; can be less stable than RLHF in some regimes.

**Choose by the learning problem:** offline preference methods remove a rollout loop, whereas RL methods can collect new policy samples against a reward signal.
Compare objectives, data support, reward reliability, compute, and held-out behavior rather than infer a universal team default or quality ordering.

### Stage 3 (optional): On-policy RL with verifiable rewards

When a task has a sufficiently reliable checker, its output can supply a reward signal:
- Sample multiple responses from the model.
- Verify each (run the code, check the math, look up the fact).
- Use the verification signal as a reward (PPO, GRPO, or similar).

Passing visible tests, matching one answer key, or finding a citation verifies only the checker's covered properties.
Measure false accepts, exploitation, task coverage, and held-out behavior before treating the reward as correctness.
Named reasoning-model results are evidence about their reported recipes, not a guarantee for every checker or task.

### Stage 4 (optional): Safety and constitutional training

Additional fine-tuning specifically on safety, refusal behavior, and adherence to a "constitution" (a set of principles the model should follow). Anthropic uses this prominently; most other labs have analogues.

## What an interviewer expects you to say

If asked about the LLM post-training stack:

1. Distinguish pretraining from post-training.
2. Describe SFT (instruction tuning) as the first step.
3. Describe preference optimization (RLHF or DPO) as the second step.
4. Explain which components offline DPO removes and which data/reference assumptions remain.
5. State when a task-specific verifier supplies useful reward and which failures its checks miss.

Bonus: discuss data coverage, reward hacking, and KL regularization.
A KL penalty changes the optimization trade-off; it does not establish a hard bound on every policy change.

## DPO vs RLHF in detail

The DPO loss is:
```
L_DPO = -log sigmoid(beta * [log(pi_theta(y_w|x) / pi_ref(y_w|x)) - log(pi_theta(y_l|x) / pi_ref(y_l|x))])
```
where `y_w` is the winning response, `y_l` is the losing response, `pi_theta` is the policy, `pi_ref` is the SFT model, `beta` is a temperature.

Intuition: increase the winning-versus-losing response log-likelihood margin relative to the reference.
This does not require the winning response's absolute probability to increase at every update.
DPO is derived from a KL-regularized preference setup under its modeling assumptions; the empirical pairwise objective is not a hard measured-KL constraint.

vs RLHF: separate reward model, sampling from the policy, gradient updates via PPO with the reward model providing rewards. More moving parts; more flexible.

Offline DPO can simplify infrastructure, but comparative quality depends on the data, objective, model, and evaluation.
Choose after testing the relevant behavior and cost trade-offs, not from an assumed universal preference for one method.

## Common confusions

- **"RLHF is required for alignment."** Just SFT goes a long way. Most of the helpful-assistant behavior comes from SFT. RLHF/DPO adds polish.
- **"DPO is just RLHF without RL."** True high-level, but DPO has different optimization dynamics, different data requirements (offline pairs vs on-policy samples), and different failure modes.
- **"Alignment fixes hallucinations."** It changes refusal patterns and helpfulness but doesn't eliminate hallucinations. Hallucinations require separate techniques (RAG, verification).
- **"You can't fine-tune RLHF'd models."** You can; the standard pattern is SFT + DPO can be redone or extended. But it's tricky, further fine-tuning can degrade alignment.

## Open problems in 2026

- **Reward hacking**: model finds ways to get high reward that don't correspond to actual quality (e.g., sycophancy, length bias, format gaming).
- **Calibration**: post-training changes the model and score distribution. Measure the declared confidence target on independent data rather than assume a universal direction of change.
- **Long-horizon RL**: hard to do RL on tasks requiring many tokens of reasoning before the reward is observable.
- **Generalization of safety**: a model trained to refuse one category of harmful request may or may not generalize to related categories.

## Why interviewers ask

This question tests:
1. Whether you've kept up with the field (the stack changed substantially in 2023-2024 with DPO and verifiable rewards).
2. Whether you understand the *why* behind each stage (raw model needs SFT for format; SFT model needs preference optimization for nuance).
3. Whether you can discuss reward hacking and other operational concerns.

Senior LLM-team interviews often probe deeply on this, because it's where the interesting work is happening in 2026.

---

*Related: [Transformer architecture](/concepts/transformer-architecture/), [How would you evaluate an LLM application?](/questions/how-would-you-evaluate-an-llm-application/), [Calibration](/concepts/calibration/).*
