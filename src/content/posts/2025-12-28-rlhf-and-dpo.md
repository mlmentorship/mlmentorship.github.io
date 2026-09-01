---
title: "RLHF, DPO, and the alignment training stack"
description: "How LLMs get from 'next-token predictor' to 'helpful assistant.' The post-training pipeline in 2026."
date: "2025-12-28"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

The post-training pipeline turning a base LLM (fluent, raw next-token predictor) into a useful assistant. Standard 2026 stack: **supervised fine-tuning (SFT) -> preference optimization (DPO or RLHF)** plus optional safety / RL.

A base LLM trained on internet text is fluent but not useful. It will continue your prompt as if it were the next bit of internet text, not as a helpful answerer. Post-training is what turns the base model into Claude, ChatGPT, or any other production assistant.

Understanding the post-training stack matters because:
- Most of the differences between LLMs in 2026 come from post-training, not from architecture or pretraining.
- Many production fine-tuning decisions are about which post-training stage to invest in.
- Interviewers test this because it's the live frontier of LLM development.

**Learning objective:** starting from the same preference pairs, trace which learned components and data-generation steps RLHF adds that DPO removes.

## The stages

### Stage 0: Pretraining (for context)

Train on trillions of tokens of internet text with next-token prediction. Produces a base model that's fluent in many domains but doesn't follow instructions or refuse harmful requests.

### Stage 1: Supervised Fine-Tuning (SFT)

Train the base model on examples of (prompt, ideal response) pairs. The model learns the *format* of being a helpful assistant.

- Data: 10K-1M instruction-response pairs, often a mix of human-written and model-generated.
- Loss: standard cross-entropy on the response tokens (mask the prompt).
- Effect: model now follows instructions, uses formatting, refuses obvious bad requests.
- Cost: a few thousand GPU-hours at most for a 70B model; trivially feasible.

The quality of SFT data is the dominant factor. A small (~10K) high-quality SFT dataset typically beats a large (~1M) noisy one.

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

**The 2026 picture**: DPO and its variants (IPO, KTO, ORPO) have become the default for most teams because of simplicity. Big labs (Anthropic, OpenAI) still use RL-based methods for the most polished models, but the gap is narrowing.

### Stage 3 (optional): On-policy RL with verifiable rewards

For tasks where correctness is checkable (math, code, factual Q&A), use RL with the verifier as the reward signal:
- Sample multiple responses from the model.
- Verify each (run the code, check the math, look up the fact).
- Use the verification signal as a reward (PPO, GRPO, or similar).

This is what produced the recent jumps in math/reasoning capability (DeepSeek-R1 style training, OpenAI's o-series, etc.). Not yet universal but clearly the direction the frontier is going.

### Stage 4 (optional): Safety and constitutional training

Additional fine-tuning specifically on safety, refusal behavior, and adherence to a "constitution" (a set of principles the model should follow). Anthropic uses this prominently; most other labs have analogues.

## What an interviewer expects you to say

If asked about the LLM post-training stack:

1. Distinguish pretraining from post-training.
2. Describe SFT (instruction tuning) as the first step.
3. Describe preference optimization (RLHF or DPO) as the second step.
4. Mention DPO has largely displaced RLHF in 2026 due to simplicity.
5. Mention that on-policy RL with verifiable rewards is the new frontier for reasoning capabilities.

Bonus: discuss data quality (SFT data quality &gt;&gt; SFT data quantity), reward hacking (models gaming the reward model), and KL penalties (preventing the policy from deviating too far from the SFT model).

## DPO vs RLHF in detail

The DPO loss is:
```
L_DPO = -log sigmoid(beta * [log(pi_theta(y_w|x) / pi_ref(y_w|x)) - log(pi_theta(y_l|x) / pi_ref(y_l|x))])
```
where `y_w` is the winning response, `y_l` is the losing response, `pi_theta` is the policy, `pi_ref` is the SFT model, `beta` is a temperature.

Intuition: increase the model's likelihood of `y_w` relative to `y_l`, normalized by the SFT model's likelihoods (the implicit reward). The KL constraint to the SFT model is built into the loss form.

vs RLHF: separate reward model, sampling from the policy, gradient updates via PPO with the reward model providing rewards. More moving parts; more flexible.

The empirical picture: comparable quality, much simpler implementation. For most teams, DPO is the right choice unless you have specific reasons to need RL.

## Common confusions

- **"RLHF is required for alignment."** Just SFT goes a long way. Most of the helpful-assistant behavior comes from SFT. RLHF/DPO adds polish.
- **"DPO is just RLHF without RL."** True high-level, but DPO has different optimization dynamics, different data requirements (offline pairs vs on-policy samples), and different failure modes.
- **"Alignment fixes hallucinations."** It changes refusal patterns and helpfulness but doesn't eliminate hallucinations. Hallucinations require separate techniques (RAG, verification).
- **"You can't fine-tune RLHF'd models."** You can; the standard pattern is SFT + DPO can be redone or extended. But it's tricky, further fine-tuning can degrade alignment.

## Open problems in 2026

- **Reward hacking**: model finds ways to get high reward that don't correspond to actual quality (e.g., sycophancy, length bias, format gaming).
- **Calibration**: post-trained models are typically *less* calibrated than base models. Hard to recover.
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
