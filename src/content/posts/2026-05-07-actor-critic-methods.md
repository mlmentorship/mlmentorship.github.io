---
title: "Actor-critic methods"
description: "Policy gradient with a learned value baseline. The actor picks actions; the critic estimates how good they were. The architecture under PPO, A3C, SAC, and most modern RL."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

An **actor-critic** algorithm trains two networks jointly: an **actor** $\pi_\theta(a \mid s)$ that chooses actions and a **critic** $V_\phi(s)$ (or $Q_\phi(s, a)$) that estimates expected return. The actor is updated with a policy gradient using the critic's estimate as a baseline; the critic is updated to fit observed returns.

Pure policy gradient (REINFORCE) is unbiased but high-variance. Pure value-based methods (Q-learning, DQN) are sample-efficient but only support discrete actions and struggle with stochastic optimal policies. Actor-critic combines both: low-variance gradient estimates from the critic, direct policy parameterization from the actor.

Almost every modern continuous-control RL algorithm is actor-critic: PPO, A2C/A3C, SAC, TD3, DDPG. RLHF for LLMs is actor-critic.

## The two updates

### Actor (policy gradient)

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot \hat{A}(s, a) \right],
$$

where $\hat{A}$ is an advantage estimate from the critic. See [GAE](/concepts/advantage-estimation-and-gae/) for the standard recipe.

### Critic (value regression)

$$
\mathcal{L}_\phi = \mathbb{E}\left[ (V_\phi(s_t) - \hat{R}_t)^2 \right],
$$

where $\hat{R}_t$ is the return target (TD($\lambda$) target, or the GAE-derived $\hat{A}_t + V_\phi(s_t)$).

Both gradients flow on the same data. Many implementations share most of the backbone between actor and critic and split off two heads at the end.

<!-- visual:actor-critic-two-objective-flow -->
<figure class="learning-figure editorial-illustration" aria-labelledby="actor-critic-two-objective-flow-edition-title"><span class="visual-fragment-anchor" id="actor-critic-visual-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="actor-critic-svg-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="actor-critic-svg-desc" aria-hidden="true"></span><span class="visual-fragment-anchor" id="arrow-forward" aria-hidden="true"></span><span class="visual-fragment-anchor" id="arrow-backward" aria-hidden="true"></span><p class="visual-title" id="actor-critic-two-objective-flow-edition-title">Stop the advantage gradient; keep both losses updating shared features</p><svg viewBox="0 0 360 456" role="img" aria-labelledby="actor-critic-two-objective-flow-edition-svg-title actor-critic-two-objective-flow-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="actor-critic-two-objective-flow-edition-svg-title">The advantage stops gradients; both heads still train shared features</title><desc id="actor-critic-two-objective-flow-edition-svg-desc">One state enters shared features and then actor and critic heads. The policy loss uses log action probability and advantage y minus V. A stop mark on the V-to-advantage path blocks policy-loss gradients into the critic. The value loss is an illustrative squared error against fixed target y. Separate dashed paths return each loss through its own head into the shared features. Solid lines construct the losses; dashed lines indicate parameter-gradient ownership.</desc><g data-editorial-node><rect x="125" y="12" width="110" height="40" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="37" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">state s</tspan></text></g><path d="M180 52L180 84" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,77.8569220667674L180,84L183.35597877022943,77.8569220667674" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="90" y="88" width="180" height="50" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="180" y="118" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">shared features</tspan></text></g><path d="M180 138L180 158" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M90 158L270 158" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M90 158L90 172" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M86.64402122977057,165.8569220667674L90,172L93.35597877022943,165.8569220667674" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M270 158L270 172" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M266.6440212297706,165.8569220667674L270,172L273.3559787702294,165.8569220667674" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="15" y="172" width="150" height="58" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="196.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">actor</tspan><tspan x="90" dy="19">log pi(a|s)</tspan></text></g><g data-editorial-node><rect x="195" y="172" width="150" height="58" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="270" y="196.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">critic</tspan><tspan x="270" dy="19">V(s)</tspan></text></g><path d="M90 230L90 340" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M86.64402122977057,333.85692206676737L90,340L93.35597877022943,333.85692206676737" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M270 230L270 340" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M266.6440212297706,333.85692206676737L270,340L273.3559787702294,333.85692206676737" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="253" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">fixed target y</tspan></text><g data-editorial-node><rect x="120" y="263" width="120" height="42" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="180" y="289" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Ahat = y - V</tspan></text></g><path d="M270 280L240 280" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M246.1430779332326,276.6440212297706L240,280L246.1430779332326,283.3559787702294" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M120 305L102 340" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M101.8250934989347,333.0021855043246L102,340L107.79395546934207,336.07188594624836" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="15" y="340" width="150" height="62" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="366.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">policy loss</tspan><tspan x="90" dy="19">-log pi x Ahat</tspan></text></g><g data-editorial-node><rect x="195" y="340" width="150" height="62" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="270" y="366.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">value loss</tspan><tspan x="270" dy="19">(V - y)^2</tspan></text></g><path d="M165 370L180 370" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M180 370L180 320" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M180 320L180 305" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M183.35597877022943,311.14307793323263L180,305L176.64402122977057,311.14307793323263" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M240 293L258 293" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M258 286L258 300" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="240" y="319" style="font-size:12px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="240" dy="0">STOP</tspan></text><path d="M15 370L8 370" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M8 370L8 201" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M8 201L15 201" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M8.85692206676739,204.35597877022943L15,201L8.85692206676739,197.64402122977057" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M345 370L352 370" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M352 370L352 201" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M352 201L345 201" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M351.14307793323263,197.64402122977057L345,201L351.14307793323263,204.35597877022943" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M75 172L75 150" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M75 150L140 150" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M140 150L140 138" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M143.35597877022943,144.1430779332326L140,138L136.64402122977057,144.1430779332326" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M285 172L285 150" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M285 150L220 150" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M220 150L220 138" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M223.35597877022943,144.1430779332326L220,138L216.64402122977057,144.1430779332326" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="180" y="435" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">solid: values | dashed: loss gradients</tspan></text></svg><figcaption><strong>Read it this way:</strong> The advantage value reaches the policy loss, but its stop mark blocks the return gradient into V. Each loss still updates its own head and the shared features. The squared value loss and fixed target y are illustrative. <span class="visual-reader-check" data-reader-check="actor-critic-two-objective-flow"><strong>Check:</strong> Can the policy loss update the critic through the advantage? Which losses update shared features?</span></figcaption></figure>

## The taxonomy

### A2C / A3C

Synchronous (A2C) or asynchronous (A3C) advantage actor-critic. Multiple environments collect rollouts in parallel; updates use $n$-step returns. Simple, robust, the historical baseline ([Mnih et al., 2016](https://arxiv.org/abs/1602.01783)).

### PPO

Adds a **clipped surrogate objective** to bound how much the new policy can move from the old one. Allows multiple epochs of update on the same rollout. The default for most practical RL today, including RLHF.

### DDPG

Off-policy deterministic actor-critic for continuous control ([Lillicrap et al., 2016](https://arxiv.org/abs/1509.02971)). The actor outputs a deterministic action; the critic is $Q_\phi(s, a)$; gradient flows from $Q$ back through the actor via the chain rule. Notoriously brittle.

### TD3

DDPG plus three fixes ([Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477)): twin Q-networks (take the min to reduce overestimation), delayed policy updates, target policy smoothing. Much more stable than DDPG.

### SAC

Soft Actor-Critic ([Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)). Adds an entropy bonus to the reward, learning a maximum-entropy policy. Sample-efficient and robust; the standard for off-policy continuous control.

## On-policy vs off-policy

- **On-policy** (A2C, PPO): the data must come from the current policy. Rollouts are discarded after a few updates.
- **Off-policy** (DDPG, TD3, SAC): the data can come from any policy, stored in a replay buffer. Importance sampling or deterministic gradients handle the off-policyness.

Off-policy is more sample efficient (data can be reused) but harder to stabilize. On-policy is simpler and more reliable, at the cost of needing fresh rollouts.

## How RLHF fits

RLHF is just PPO (an actor-critic algorithm) with:

- The actor initialized from a pretrained LLM.
- The critic estimating value from the same model.
- The reward coming from a learned reward model trained on human preferences.

Modern alternatives (DPO, IPO, KTO) bypass the actor-critic step entirely; they reformulate the optimization as a supervised loss on preference pairs. Faster, simpler, and the new dominant approach in 2024 to 2026.

## Common pitfalls

- **Forgetting to detach the value estimate when computing the policy gradient.** The advantage flows into the actor; the actor gradient should not flow through the critic.
- **Sharing too much of the backbone**. With one body and two heads, the value loss can dominate the gradient and hurt the policy. Tune the loss weight or use separate networks for hard tasks.
- **Off-policy updates without correction**. PPO assumes near-on-policy data; running it with a deep replay buffer breaks the clipping bound.
- **Using PPO defaults blindly**. PPO has a dozen hyperparameters (clip range, GAE lambda, value-loss coef, entropy coef, learning rate, batch size, epochs per rollout). The interactions matter.

## Related

- [Policy gradient methods](/concepts/policy-gradient/).
- [Proximal Policy Optimization](/concepts/ppo/).
- [Advantage estimation and GAE](/concepts/advantage-estimation-and-gae/).
