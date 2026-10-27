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
<figure class="learning-figure" aria-labelledby="rlhf-dpo-paths-title">
	<p class="visual-kicker">Same judgments, different optimization paths</p>
	<p class="visual-title" id="rlhf-dpo-paths-title">What machinery does RLHF add between preference pairs and the updated policy?</p>
	<section class="visual-panel" aria-labelledby="rlhf-dpo-shared-input-title">
		<h4 id="rlhf-dpo-shared-input-title">SHARED START · OFFLINE PREFERENCE DATA</h4>
		<p><strong>Prompt <var>x</var> + preferred response <var>y<sub>w</sub></var> + rejected response <var>y<sub>l</sub></var></strong><br />Both paths can begin with the same pairwise judgments and a fixed SFT reference policy.</p>
	</section>
	<div class="visual-grid--two" role="group" aria-label="Two training paths from shared preference pairs: RLHF learns a reward model and performs on-policy reinforcement learning, while DPO updates the policy directly from offline pairs">
		<section class="visual-panel" aria-labelledby="rlhf-training-path-title">
			<h4 id="rlhf-training-path-title">RLHF · EXPLICIT REWARD, ON-POLICY LOOP</h4>
			<p><strong>1 · Fit a reward model</strong><br />Learn a scalar score that ranks <var>y<sub>w</sub></var> above <var>y<sub>l</sub></var>.</p>
			<p><strong>2 · Generate fresh responses</strong><br />Sample rollouts from the current policy; the training distribution moves as the policy changes.</p>
			<p><strong>3 · Optimize with RL</strong><br />PPO or another RL algorithm raises predicted reward while an explicit KL penalty limits drift from the reference.</p>
			<p><strong>Maintain</strong><br />Policy + separate reward model + rollout and RL infrastructure.</p>
		</section>
		<section class="visual-panel" aria-labelledby="dpo-training-path-title">
			<h4 id="dpo-training-path-title">DPO · IMPLICIT REWARD, OFFLINE UPDATE</h4>
			<p><strong>1 · Score a likelihood-ratio margin</strong><br />Compare how much the trainable policy favors <var>y<sub>w</sub></var> over <var>y<sub>l</sub></var>, relative to the fixed reference.</p>
			<p><strong>2 · Optimize the policy directly</strong><br />Apply binary cross-entropy to the stored pairs; no reward-model fit is required.</p>
			<p><strong>3 · Reuse offline comparisons</strong><br />No on-policy rollout or PPO loop is part of the DPO objective.</p>
			<p><strong>Maintain</strong><br />Policy + fixed reference + preference dataset.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> begin at the shared preference triple, then read each column downward. RLHF converts the judgments into a separate reward model and repeatedly generates new policy samples for RL; DPO turns the same ranking signal into a direct policy loss against a fixed reference. “Direct” removes the reward-model and on-policy RL stages, not the need for preference data or reference regularization. Original synthesis based on <a href="https://arxiv.org/abs/2203.02155">Ouyang et al. (2022)</a> and <a href="https://arxiv.org/abs/2305.18290">Rafailov et al. (2023)</a>.</figcaption>
</figure>

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
