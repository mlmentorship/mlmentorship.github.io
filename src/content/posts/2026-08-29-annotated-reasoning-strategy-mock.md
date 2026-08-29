---
title: "Annotated upper-IC mock: reasoning under a fixed budget"
description: "A synthetic ten-turn interview showing how fixed-capacity arithmetic, verification, serving, recovery, portfolio choices, and delegated leadership affect upper-IC calibration."
date: "2026-08-29"
draft: false
tags: ["guides", "mock-interview", "reasoning-models", "system-design", "principal", "senior-principal"]
category: "guides"
aliases: ["reasoning model mock interview", "fixed compute upper IC mock", "principal reasoning strategy transcript", "reasoning system design mock"]
roles: ["Applied Scientist", "Machine Learning Engineer", "Research Scientist", "Research Engineer"]
rounds: ["ML system design", "LLM systems", "Technical strategy", "Research system design"]
difficulty: "Advanced"
priority: "Role-specific"
prerequisites: ["design-reasoning-model-fixed-budget", "test-time-compute-search-verifiers", "l5-vs-l6-faang-ml"]
---

Upper-IC performance appears in the decisions a candidate preserves when compute, evidence, latency, and organizational conditions change. Longer answers and larger models do not establish senior scope.

> **Synthetic teaching example:** Every interviewer and candidate statement below is invented. This is not a company transcript, candidate report, or hiring claim. Use it for transfer practice rather than memorizing its wording.

The candidate will protect online capacity, estimate training cost, route extra inference effort selectively, bound verifier authority, and define recovery. Later turns add principal portfolio choices and company-dependent senior-principal direction through delegated owners.

## How to use this mock

Read only the interviewer challenge for a turn. Pause and answer aloud. Then compare your decision, arithmetic, and evidence with the synthetic candidate.

For every turn, ask six questions:

1. Did the candidate answer the changed condition directly?
2. Did they make a choice rather than list options?
3. Did the arithmetic reconcile with the fixed cluster?
4. Did they preserve a technical or safety invariant?
5. Did they name evidence that could reverse the choice?
6. Did the claimed scope match the authority in the answer?

The annotations explain observable signals. They are not scripts. A copied sentence without comparable judgment or experience will not supply upper-IC evidence.

## Scenario

A company owns 128 H100-class accelerators in 16 eight-accelerator nodes. The first-year plan cannot use public-cloud overflow.

The company wants a 14-billion-parameter reasoning model for coding, quantitative analysis, structured research, and internal decision support. It already has a licensed 14B dense base model and a 3B production assistant.

The cluster must handle continued training, post-training, rollout generation, verifier work, evaluation, canaries, and online inference. Launch demand is 20 million requests per day with a four-times peak factor.

Inputs have a 700-token median, a 6,000-token p95, and a 24,000-token p99. Outputs have a 220-token median and a 1,400-token p95. Some coding and research jobs can run asynchronously for 60 seconds.

The team has exact checkers for parts of arithmetic and code. Human review is limited. It has no general verifier for research quality, safety, or intent.

The interviewer asks the candidate to design the two-quarter program and its first production release.

## Observer scorecard

Score each dimension from 0 to 2 after every challenged turn.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Framing | Maximizes model score | Names a budget | Defines product utility, hard constraints, and non-goals |
| Capacity accounting | Lists accelerators | Estimates one workload | Reconciles training, serving, evaluation, and reserve |
| Training program | Proposes one large run | Adds tuning stages | Uses an evidence ladder, recovery, and stopping rules |
| Routing and test-time compute | Gives one budget to all | Adds a router | Allocates marginal compute by task, risk, support, and deadline |
| Verification and evaluation | Trusts one judge | Mentions verifier error | Defines support, abstention, protected audits, and product evidence |
| Serving and reliability | Says batch and autoscale | Adds queueing | Connects prefill, decode, KV, admission, degradation, and recovery |
| Economics | Reports token counts | Adds average cost | Uses slice-level quality-cost-latency curves and tail limits |
| Rollout and recovery | Says canary and rollback | Names stages | Versions the policy, sets stop gates, and rehearses restoration |
| Ownership and delegation | Candidate owns everything | Names partner teams | Gives domain owners decisions, evidence, and escalation paths |
| Durability | Says multi-year | Documents a roadmap | Preserves doctrine, succession, portability, and reversal under change |

A strong staff answer should finish high on capacity, verification, serving, and recovery. A principal answer should also finish high on economics, ownership, and portfolio decisions. Senior-principal evidence depends on the employer and requires durable direction through other technical leaders.

## Challenged turn 1: frame the fixed decision

**Interviewer:** You have 128 accelerators and six months. Train and serve the reasoning model. Start wherever you want.

**Candidate:**

> I will frame this as a capacity portfolio, not a model-training project. The same accelerator-hour can support continued training, rollout generation, evaluation, or a production request, so every proposal has an opportunity cost.
>
> I need four clarifications. Which task families create product value? Which wrong answers have material consequences? Which workloads need interactive latency? Is the cluster a hard physical limit or only a financial target?
>
> I will assume the cluster is a hard first-year limit. Coding, quantitative analysis, and research synthesis are the launch families. Routine requests are interactive. Long coding and research tasks may be asynchronous. Consequential internal decisions require stronger evidence or human review.
>
> My initial decision is to start from the licensed 14B model. I would not pretrain from random initialization. I would reserve online inference and recovery capacity first, then fund a staged continuation and post-training program. The launch policy will spend extra inference compute only where measured marginal gain supports it.

### Annotation

The candidate changes the unit of planning from “one model” to “one fixed capacity portfolio.” They identify task, consequence, latency, and hardness of the constraint before discussing a framework.

The answer also makes two choices. It reuses the licensed base model and protects production before speculative training. Those choices can be tested and reversed later.

The candidate separates difficult tasks from consequential tasks. This prevents an easy financial action from receiving weak controls merely because the model answers it quickly.

**Level signal:** strong staff framing with an early principal portfolio signal. The principal signal is the allocation boundary across research and production, not the vocabulary.

**Score movement:** Framing moves from 0 to 2. Capacity accounting moves from 0 to 1 because the candidate names competing envelopes without numbers.

### Weak alternative

> I would fine-tune the 14B model with supervised examples, apply RL, deploy it with a high-throughput inference engine, and autoscale the endpoints.

This answer lists plausible work. It does not state what the cluster can support, what traffic receives the model, or what gets displaced when training runs.

### Retry drill

Use a different fixed-resource prompt. Spend two minutes naming the unit of scarcity, product outcome, hard constraints, and first rejected option. Do not name a library until those are clear.

## Challenged turn 2: reconcile the arithmetic

**Interviewer:** That sounds orderly, but I need numbers. How much capacity exists, and can your continued-training run fit?

**Candidate:**

> The quarter has $128 \times 24 \times 90 = 276{,}480$ raw accelerator-hours. I would remove 10% for maintenance, failed hosts, upgrades, checkpoint movement, and fragmentation. That leaves about 248,800 usable hours.
>
> For a 14B dense model continued on 120B tokens, the parameter-matrix estimate is $6ND$. That gives about $1.0 \times 10^{22}$ FLOPs.
>
> A six-node job uses 48 accelerators. At one petaFLOP per accelerator and 35% measured model FLOPs utilization, it provides about $1.68 \times 10^{16}$ useful FLOP/s. The simple estimate is about 167 hours.
>
> I would plan eight to ten days. The extra time covers long-context attention, vocabulary work, communication, checkpoints, data stalls, and bounded restart risk. I would replace this estimate with measured tokens per second from short 14B runs.
>
> The main run consumes roughly 9,200 to 11,500 accelerator-hours. That fits within a larger continued-training envelope, but it is not the whole program. Scale studies, ablations, failed runs, and checkpoint evaluation also consume hours.

### Annotation

The candidate uses the dense-model approximation correctly and labels it as a planning estimate. They do not confuse peak device throughput with sustained job throughput.

The answer connects wall time with accelerator-hours. Fixed-cluster planning needs both. A ten-day run can fit in total hours while still colliding with production at the wrong time.

The candidate also refuses false precision. Attention, data, collectives, and checkpoint behavior need measurement on the actual architecture and stack.

**Level signal:** strong staff systems depth. The arithmetic is necessary for higher scope, though arithmetic alone does not establish principal judgment.

**Score movement:** Capacity accounting moves from 1 to 2. Training program moves from 0 to 1 because the candidate reserves room around the promoted run.

### Weak alternative

> A 14B model is small enough for 128 H100s. The run should take a few days, and we can use all accelerators when traffic is low.

“Small enough” does not provide a schedule. Using every accelerator removes production and recovery capacity. The estimate also omits sustained efficiency and non-final runs.

### Retry drill

Recompute the plan with 64 accelerators, 20% maintenance loss, and 60B tokens. Say which numbers scale linearly and which still require a benchmark.

## Challenged turn 3: defend the portfolio allocation

**Interviewer:** The research lead wants 70% of the cluster for training because model quality is the product. Why refuse?

**Candidate:**

> I would ask for the marginal quality curve and price the displaced product evidence. A checkpoint cannot create value if we cannot evaluate or serve it under the launch workload.
>
> My first usable-capacity budget is 55% online serving, 14% continued training, 12% post-training, 9% evaluation and verifiers, and 10% recovery reserve. These are envelopes rather than fixed partitions.
>
> Continued training gets an experiment ladder. Small mixture studies lead to short 14B continuations. Only one candidate receives the 120B-token run. Checkpoints along that run can stop the schedule if gains flatten or general capability falls.
>
> Post-training includes rollout generation, reference-policy calls, verifier execution, and model updates. Counting only gradient steps would underprice it. Evaluation gets protected capacity because verifier audits and held-out cases cannot wait until the final checkpoint.
>
> I would move more than 14% into training only after smaller runs predict enough hard-task gain to repay the lost serving or post-training work. I would also require a protected recovery floor. A promising run is not permission to spend incident capacity.

### Annotation

The candidate answers the political challenge with an allocation rule rather than saying research is wrong. They ask for marginal gain and name the opportunity cost.

The envelopes cover the full lifecycle. Rollouts and evaluation are treated as compute consumers. The recovery reserve has a purpose and cannot be borrowed informally.

The checkpoint stopping rule protects the budget from a run that continues because its original plan said 120B tokens. This is a portfolio decision under evidence.

**Level signal:** principal. The candidate changes research, evaluation, platform, and product roadmaps through one capacity rule.

**Score movement:** Training program moves from 1 to 2. Economics moves from 0 to 1. Ownership and delegation moves from 0 to 1 because research must present evidence within a shared rule.

### Weak alternative

> I would compromise at 50% for training and 50% for serving. Both teams get a fair share.

A symmetric split ignores rollout generation, evaluation, maintenance, and recovery. It also treats organizational fairness as a substitute for expected product value.

### Retry drill

Create a five-envelope budget for a 32-accelerator cluster. For each envelope, state one event that can increase it and one event that can reduce it.

## Challenged turn 4: route when difficulty estimates are wrong

**Interviewer:** Your router will misclassify hard tasks as easy. Why should we trust it with the budget?

**Candidate:**

> I would not trust one router score as truth. The route combines product class, risk, prompt structure, required tools, historical outcomes, first-pass checks, candidate disagreement, deadline, and current capacity.
>
> I would start with four policies. Routine requests use the 3B model or one short 14B pass. Hard requests receive tools or one bounded revision. A smaller verified-search slice receives two to four candidates. Unsupported or consequential tasks abstain or move to specialist review.
>
> Router false negatives are cheap-routed tasks that would have gained from more compute. I estimate them by sending a random sample of cheap traffic through a stronger shadow policy. False positives are expensive-routed tasks that the cheap route already solves. I estimate those with paired replay.
>
> The router is recalibrated for each model and verifier bundle. Post-training can make yesterday's hard task routine. Prompt length alone is a weak proxy.
>
> Risk remains a separate axis. A simple high-impact task may require an exact check or approval without a long generation. A difficult low-impact puzzle may receive more search without action authority.

### Annotation

The candidate accepts router error and designs measurement around it. Shadow counterfactuals estimate missed quality and wasted compute.

The answer uses several bounded policies instead of a continuous token promise that is hard to operate. Model, verifier, deadline, and capacity can change the chosen tier.

Separating risk from difficulty protects the system from a common mistake. Compute effort and action authority solve different problems.

**Level signal:** staff mechanism with principal product policy. The higher-level signal comes from one shared routing contract that still permits product-specific consequence rules.

**Score movement:** Routing and test-time compute moves from 0 to 2. Economics moves from 1 to 2 because routing is tied to marginal benefit and waste.

### Weak alternative

> The 14B model can score its own confidence. Low-confidence prompts get more tokens, and high-confidence prompts use the short path.

Self-confidence can be miscalibrated and manipulated. The answer provides no counterfactual estimate, support check, or separate treatment of consequence.

### Retry drill

Design a router without using model confidence. Then add confidence as one feature and state the test that could remove it.

## Challenged turn 5: challenge best-of-N

**Interviewer:** Best-of-8 has the highest benchmark score. Why would you ship only two or four candidates?

**Candidate:**

> I need the quality gain per total accelerator-second on each task slice. I also need p95 latency, candidate correlation, verifier selection error, KV demand, and the production route mix.
>
> Under independent samples with success probability $p$, an oracle sees at least one success with probability $1-(1-p)^N$. That is an upper bound. Our samples share a model, prompt, data, and decoding policy. Their errors can be highly correlated.
>
> The verifier is also imperfect. Best-of-8 can contain a correct candidate and still return an incorrect one. Larger $N$ gives the verifier more candidates to rank and more chances to encounter an exploit.
>
> I would plot $N=1,2,4,8$ for supported task families. Coding may gain through executable tests. Research synthesis may plateau because its semantic judge cannot reliably rank subtle factual errors.
>
> I would ship the smallest $N$ on the local quality-cost frontier. If best-of-8 remains valuable for a small asynchronous high-value slice, it can live there with a hard branch and wall-time budget. It should not become the default from one average benchmark.

### Annotation

The candidate knows the optimistic sampling formula and immediately states why production falls below it. Correlation and selection error can dominate the extra oracle coverage.

They also connect candidate count to KV memory and latency. Best-of-N is a serving policy, not an isolated evaluation trick.

The answer permits best-of-8 where evidence supports it. This is stronger than rejecting expensive methods categorically.

**Level signal:** strong staff and principal economics. The candidate uses one framework to narrow or expand the method by task family.

**Score movement:** Routing and test-time compute remains 2. Verification and evaluation moves from 0 to 1. Serving and reliability moves from 0 to 1 because branch memory enters the choice.

### Weak alternative

> Eight samples are more diverse, so they are more likely to include the right answer. We should optimize batching to make them cheap.

More samples can repeat one error. Better batching cannot remove generation, decode state, verifier cost, or selection mistakes.

### Retry drill

Explain when best-of-2 beats one stronger model. Then reverse the conclusion by changing verifier quality, candidate correlation, or latency.

## Challenged turn 6: bound verifier authority

**Interviewer:** The safety judge approves a candidate and says its hidden reasoning looks honest. Can the system return it?

**Candidate:**

> No. Hidden reasoning is not a reliable safety boundary. It may be unavailable, incomplete, unstable across samples, or shaped to satisfy training. Longer reasoning can rationalize a wrong action as easily as it can repair one.
>
> I need the verifier's support contract. Which task families, formats, environments, and risk classes were used to measure its false accepts and false rejects? What does it do outside that support?
>
> For code, I would first check policy, sandbox limits, compilation, protected tests, and final repository state. A calibrated semantic judge can compare maintainability among candidates that pass those checks. It cannot override a failed hard constraint.
>
> For research synthesis, the judge may check citation entailment and coverage on measured domains. If source quality or claim type is outside support, it should abstain. Human reversal sampling estimates drift.
>
> I would record observable actions, tool arguments, verifier evidence, support status, artifacts, citations, final state, and user-visible rationale. A severe false accept blocks rollout. If a reward verifier is compromised, I also re-audit checkpoints trained against it.

### Annotation

The candidate rejects the interviewer's unsafe premise directly. They do not treat private reasoning as proof of intent, correctness, or policy compliance.

The response defines verifier authority through measured support. It separates deterministic constraints from semantic ranking and includes abstention.

The candidate follows a verifier failure backward into post-training. Repairing the production gate may not remove behavior reinforced by the old reward.

**Level signal:** retained technical and safety depth at principal scope. The answer connects training, evaluation, serving, and incident response.

**Score movement:** Verification and evaluation moves from 1 to 2. Rollout and recovery moves from 0 to 1 because severe false accepts and trained-policy impact receive stop actions.

### Weak alternative

> I would use a second independent judge. If both judges agree and the chain of thought appears safe, the answer can pass.

Two judges can share training data and failure modes. Agreement does not define support, and hidden reasoning does not enforce observable behavior.

### Retry drill

Pick one verifier. Write its supported tasks, known exclusions, false-accept target, false-reject target, and abstention behavior. Add one valid unusual solution that it must not reject.

## Challenged turn 7: protect latency and KV memory

**Interviewer:** Peak traffic arrives with many 32K prompts. Hundreds of interactive requests are decoding. What happens?

**Candidate:**

> Admission decides before the queues become universal timeouts. It reserves a bounded prompt, output, branch count, deadline, and KV allowance for the selected route.
>
> Suppose the 14B model has 40 layers, eight KV heads, head size 128, and BF16 KV values. The cache uses about $2TLH_{kv}d_hb$, or 160 KiB per token before overhead. A 32K sequence is about 5 GiB. Four diverged branches can consume much more after their shared prefix.
>
> I would chunk long prefills so they cannot stall active decode. Interactive decode gets a protected scheduling share. Long prompts use a separate class or asynchronous queue. Tenant quotas cap active sequences and reserved KV.
>
> The admission result can be admit, bounded queue, smaller candidate budget, cheaper supported route, asynchronous offer, or early rejection. If predicted start time already misses the SLO, the service should not pretend that queueing is success.
>
> Cancellation releases branches and KV pages. I would start with co-located prefill and decode. I would consider separate pools only when traces show that phase isolation repays KV transfer and another queue boundary.

### Annotation

The answer turns a traffic spike into an explicit resource decision. It calculates the KV scale and connects best-of-N to branch memory.

The candidate protects active decode and avoids admitting work that cannot meet its deadline. They do not assume autoscaling because the cluster is fixed.

The answer also avoids premature disaggregation. A more complex architecture requires measured scheduling interference.

**Level signal:** strong staff inference design. The policy becomes principal-level when it governs task and tenant priorities across products.

**Score movement:** Serving and reliability moves from 1 to 2. Capacity accounting remains 2 because the online constraint is reconciled with the physical cluster.

### Weak alternative

> Continuous batching and paged attention will handle the spike. We can increase batch size and queue the 32K prompts until a GPU is free.

These mechanisms improve use of memory and compute. They do not decide whether a request can meet its deadline or prevent long prefills from harming active decode.

### Retry drill

Recompute KV memory for 16 KV heads and FP8 cache values. Explain which result changes admission and which still needs a load test.

## Challenged turn 8: degrade and recover under two failures

**Interviewer:** One node fails during peak traffic. At the same time, the semantic verifier starts timing out. Keep the service useful.

**Candidate:**

> I would remove the failed node from admission and claim protected recovery capacity. Checkpointable training stops at its next safe boundary. Offline evaluation and broad shadow traffic pause first.
>
> For low-risk supported traffic, I reduce optional inference work in order. Best-of-4 becomes best-of-2, optional revision stops, routine tasks move to the 3B policy, and long jobs enter the asynchronous queue. I reject early before queue age guarantees a missed SLO.
>
> I do not treat a timed-out semantic verifier as a pass. Tasks with deterministic checks can continue through those checks. Tasks that require the semantic verifier either use an audited fallback, return a clearly bounded draft, or move to human review.
>
> Consequential tasks keep policy, sandbox, and final-state checks. If their required evidence is unavailable, they fail closed.
>
> Every response records the degraded policy. After recovery, I compare route share, accepted quality, verifier support, and delayed user corrections. I restore the verifier only after health checks and protected canaries pass.

### Annotation

The candidate degrades optional compute before required controls. This preserves a useful smaller product and avoids silent safety weakening.

Training, evaluation, routing, and serving participate in one incident order. The recovery reserve now has an operational purpose.

Recording the degraded policy prevents the quality team from blaming a model release for an overload route. Post-incident evaluation includes delayed product evidence.

**Level signal:** staff operational depth with principal cross-portfolio priority. The response coordinates several owners without erasing their responsibilities.

**Score movement:** Rollout and recovery moves from 1 to 2. Serving and reliability remains 2. Training program remains 2 because offline work yields correctly.

### Weak alternative

> Route around the failed node and skip the verifier temporarily. The model can still answer, and we can review logs after the incident.

Skipping a required verifier silently changes authority. The answer does not reduce demand, protect deadlines, or distinguish low-risk from consequential tasks.

### Retry drill

Write an overload order with nine steps. Mark three controls that can never be disabled for latency. Then remove two more nodes and revise the order.

## Challenged turn 9: decide build, buy, and ownership

**Interviewer:** A vendor offers training, reasoning inference, routing, and evaluation. The research lead wants to buy everything. The platform lead wants to build everything. Who decides?

**Candidate:**

> I would split the decision by capability and assign owners. Generic training frameworks, kernels, serving runtimes, queueing, storage, telemetry, and sandbox foundations are strong adopt or buy candidates. We should not rebuild them for prestige.
>
> The company must own task value, consequence classes, verifier support, protected evaluations, capacity allocation, degraded behavior, and release authority. A vendor can implement parts of those systems, but accountability remains internal.
>
> Model training owns continuation recipe and checkpoint lineage. Post-training owns reward and rollout policy. Evaluation owns protected suites and verifier authority. Inference owns scheduling and admission. Product owns utility, latency, and acceptable abstention. Reliability owns reserve and incident command.
>
> A principal release owner joins the manifest and evidence. They do not take every domain decision. Cross-domain disputes use declared evidence and escalation paths.
>
> I would pilot the vendor behind exportable checkpoint, event, verifier, and manifest interfaces. Expansion depends on quality per total compute, regional policy, recovery, support load, and exit cost. If our wrapper becomes a second weaker platform, we narrow it.

### Annotation

The candidate avoids a binary platform choice. Commodity mechanisms and company-specific decision systems receive different treatment.

Ownership follows semantics. Evaluation can block a release, product defines utility, and reliability protects reserve. The release owner integrates evidence without absorbing every domain.

The vendor pilot has expansion and contraction evidence. Portability is concrete because the candidate names the artifacts that must move.

**Level signal:** principal. The answer defines technical boundaries, decision rights, and investment evidence across organizations.

**Score movement:** Ownership and delegation moves from 1 to 2. Durability moves from 0 to 1 because portability and exit cost enter the architecture.

### Weak alternative

> Form a steering committee with research and platform leaders. They can compare vendor features with our internal roadmap and vote.

A committee does not define who owns verifier error, product utility, capacity risk, or recovery. Voting can hide incompatible evidence and accountability.

### Retry drill

Split the system into eight capabilities. Choose build, buy, adopt, or keep local for each. Name the decision owner, migration cost, and exit artifact.

## Challenged turn 10: preserve direction after external change

**Interviewer:** Six months later, a stronger external model becomes available. Power limits cut cluster capacity by 20%. The executive sponsor leaves. What survives, and what changes?

**Candidate:**

> The specific 14B checkpoint and allocation percentages can change. The operating doctrine should survive until evidence overturns it.
>
> I would preserve five rules. Compare capability at fixed total compute. Keep verifier authority within measured support. Protect production and recovery before speculative scale. Separate task difficulty from consequence. Give every major allocation a reversal condition.
>
> The external model enters the same task-slice, latency, cost, support, data-policy, and recovery evaluation. We can route approved slices to it without replacing internal identities or outcome measures. If it wins broadly and exit terms are acceptable, continued-training capacity can contract.
>
> A 20% capacity cut triggers the predeclared portfolio order. Pause low-value continuation extensions, reduce expensive low-gain routes, preserve protected evaluations, and resize product commitments. I would not hide the cut through longer queues.
>
> Principal owners for training, evaluation, inference, product, and reliability make decisions in their domains. A recorded interface and escalation process replaces sponsor-dependent negotiation. My contribution is a durable decision system that another technical leader can run, challenge, and reverse.
>
> The largest residual risk is stale evidence. If the evaluation suite and route labels stop representing production, the doctrine can allocate capacity precisely toward the wrong outcome.

### Annotation

The candidate separates durable decision rules from replaceable models and percentages. External model quality can reverse internal investment without discarding evaluation, identity, or ownership contracts.

The capacity cut has a declared order. The candidate does not preserve every roadmap through queue debt or an informal demand for more hardware.

The sponsor's departure tests succession. Several principal owners retain real authority, and another leader can operate the process from recorded evidence.

The close names a residual risk that can invalidate the entire portfolio. Stale evaluation affects training, routing, vendor comparison, and rollout.

**Level signal:** company-dependent senior principal. The evidence is durable direction through principal-owned domains, external change, succession, and reversal. Some employers would calibrate the same scope as principal.

**Score movement:** Durability moves from 1 to 2. Framing remains 2 because the close returns to product evidence. All dimensions now finish at 2 in this teaching reference.

### Weak alternative

> I would make the case to the new sponsor that reasoning is strategic. We should preserve the roadmap, buy more efficient hardware, and use the external model as a temporary fallback.

This protects the prior plan before evaluating new evidence. It depends on sponsorship and assumes the internal roadmap should survive a capacity cut.

### Retry drill

Repeat your closing with three changes: a new vendor, a 30% budget cut, and the loss of one senior owner. State what remains stable, what reopens, and who decides.

## Final scorecard

| Dimension | Start | Finish | Evidence from the ten turns |
| --- | ---: | ---: | --- |
| Framing | 0 | 2 | Capacity portfolio, task value, consequence, latency, and non-goals |
| Capacity accounting | 0 | 2 | Quarterly hours, maintenance, $6ND$, utilization, and online collision |
| Training program | 0 | 2 | Experiment ladder, checkpoint stopping, rollout accounting, and recovery |
| Routing and test-time compute | 0 | 2 | Difficulty and risk axes, shadow counterfactuals, bounded best-of-N |
| Verification and evaluation | 0 | 2 | Support contracts, false accepts, abstention, protected checks, and audits |
| Serving and reliability | 0 | 2 | KV arithmetic, prefill and decode, admission, degradation, and node loss |
| Economics | 0 | 2 | Marginal quality per total compute, opportunity cost, and slice frontiers |
| Rollout and recovery | 0 | 2 | Stop gates, degraded-policy identity, verifier removal, and restoration |
| Ownership and delegation | 0 | 2 | Domain decisions, independent evaluation, release integration, escalation |
| Durability | 0 | 2 | Portable evidence, external model response, capacity cut, and succession |

This perfect reference score does not mean a real candidate should deliver every detail. A live interviewer chooses a few branches. The candidate must maintain a coherent decision while moving between arithmetic, mechanism, operation, and scope.

## Why the answer clears staff

The answer clears a strong staff bar because it turns a fixed cluster into an executable system.

The candidate:

- computes the capacity and training order of magnitude;
- states caveats instead of hiding uncertainty;
- builds a staged continuation and post-training plan;
- designs bounded routing and test-time policies;
- defines verifier support and failure handling;
- calculates KV pressure;
- defines admission, degradation, and recovery;
- assigns subsystem ownership;
- preserves enough depth under hostile follow-up.

A staff candidate can stop after delivering this mechanism across several teams. They do not need to claim multi-portfolio technical doctrine.

## Why the answer clears principal

The answer clears principal because the candidate repeatedly chooses boundaries and opportunity costs across organizations.

They decide:

- how much capacity research, post-training, evaluation, serving, and recovery receive;
- which capabilities are shared and which remain product-owned;
- when training should stop in favor of serving or verification work;
- how task families receive different compute policies;
- which verifier evidence can block a launch;
- what the vendor may own and which decision contracts remain internal;
- how several staff and principal owners retain accountability;
- which evidence expands, narrows, or reverses the program.

The candidate does not call broad scope sufficient. They retain direct depth in compute estimates, verifier behavior, and KV admission.

## Why the final turn can clear company-dependent senior principal

“Senior principal” is not a stable title across companies. The final turn shows an archetype rather than a universal level rule.

The higher-scope evidence includes:

- a small technical doctrine above replaceable implementations;
- several principal-owned portfolios with real decision rights;
- a response to vendor, hardware, power, and leadership change;
- portability of evaluation and release evidence;
- a predefined order for contraction;
- succession that lets another leader operate and challenge the system;
- willingness to reverse internal model investment when external evidence wins.

Some companies expect this behavior from principal engineers. Others reserve it for senior principal, distinguished engineer, or a research director with retained technical depth.

## What the mock does not prove

A synthetic design answer cannot prove a career record.

It does not prove that the candidate has:

- shipped a model under a real capacity constraint;
- persuaded research and product leaders to change budgets;
- built a verifier with measured false accepts;
- recovered a distributed training run;
- operated a fixed inference cluster during an incident;
- delegated direction to other principal engineers;
- reversed a multi-quarter investment;
- created a system that survived their departure.

Project and behavioral rounds need real evidence. The candidate should partition personal decisions, team work, partner authority, measured outcomes, mistakes, and changes after their direct involvement.

The mock also does not establish that its numeric assumptions fit another model. Hardware throughput, model architecture, context, data, route mix, and product demand must be remeasured.

## Observer guidance

Before the session:

1. Choose two depth turns and two scope turns.
2. Hide the synthetic candidate response.
3. Prepare one changed condition that is not in this page.
4. Decide which score dimensions the change tests.
5. Score independently before discussing feedback.

During the session, interrupt vague claims with concrete probes:

- “What gets less capacity?”
- “What number would you calculate?”
- “What does the verifier support?”
- “What happens after timeout?”
- “Which queue misses its deadline?”
- “Who owns this decision?”
- “What evidence stops the run?”
- “What survives a vendor switch?”

After the session, assign three repairs at most. Choose one arithmetic or mechanism repair, one operating repair, and one scope or ownership repair.

## Strong performance signals

- Answers the changed condition before returning to the prepared structure.
- Preserves fixed-capacity arithmetic across every proposal.
- Distinguishes training FLOPs from serving behavior.
- Prices rollout generation, verifier calls, and failed runs.
- Treats more tokens and samples as measured options.
- Defines verifier support and abstention.
- Rejects hidden reasoning as the safety boundary.
- Connects best-of-N with correlation, selection error, latency, and KV.
- Degrades optional compute before required controls.
- Keeps protected evaluation during a portfolio cut.
- Gives owners real decisions rather than advisory roles.
- Names evidence that reverses the candidate's preferred plan.

## Weak performance signals

- Repeats the original architecture after a condition changes.
- Uses peak hardware specifications as delivered throughput.
- Spends the whole cluster on one promoted run.
- Treats post-training as a small gradient update.
- Routes only from self-reported model confidence.
- Assumes every extra candidate improves the selected result.
- Calls judge agreement correctness.
- Uses private reasoning as intent evidence.
- Queues every overload request indefinitely.
- Drops evaluation or recovery whenever research asks for more capacity.
- Creates a committee instead of assigning decisions.
- Preserves a roadmap because a sponsor once approved it.

## Spaced transfer practice

The goal is transfer to new constraints. Repeating the same wording can improve fluency while hiding weak reasoning.

### Day 0: cold 60-minute mock

Use the original scenario. Record the full answer. Score every dimension before reading the synthetic responses.

Choose one depth gap and one scope gap. Do not repair ten things at once.

### Day 2: arithmetic compression

Rebuild the capacity, $6ND$, and KV calculations from memory. Use a 64-accelerator cluster and an 8B model.

Finish in eight minutes. State every assumption that could change the result.

### Day 5: verifier transfer

Replace coding with scientific literature synthesis. Design a verifier support contract for claims, citations, and uncertainty.

Explain what remains outside support. Give one false accept, one false reject, and one abstention.

### Day 9: serving transfer

Keep the model fixed and change traffic to long-document batch analysis. Reconsider interactive SLOs, queue policy, prefill, decode, and admission.

Do not copy the original route percentages.

### Day 14: organizational transfer

Assume research, platform, and product report to separate executives. Give principal owners real authority and define two cross-domain escalation paths.

Then remove the executive sponsor. Explain what still operates.

### Day 21: reversal practice

Start from a plan you favor. Introduce evidence that makes it wrong. Reverse or narrow it without defending sunk cost.

Examples include a verifier exploit, lower product demand, a stronger vendor model, or a failed continuation curve.

### Day 30: mixed upper-IC loop

Run this design beside one technical depth round and one real project deep dive. The design tests hypothetical judgment. The project round tests whether similar behavior happened in practice.

Compare both answers for consistency. A candidate who praises reversal in design should be able to discuss a real decision they changed.

## Additional changed-condition prompts

Use one prompt per later retry.

1. The cluster drops to 48 accelerators, and interactive traffic cannot shrink.
2. The 3B model improves enough to solve 90% of routine tasks.
3. The 14B continuation gains on code and loses on multilingual research.
4. A verifier exploit affected two prior post-training checkpoints.
5. Best-of-2 beats best-of-8 after a diversity collapse.
6. A 128K context requirement arrives from one high-value customer.
7. A region prohibits external model calls and stored prompts.
8. An exact theorem checker becomes available for one task family.
9. The inference runtime cannot preempt or migrate active sequences.
10. Human review capacity falls by half during launch month.
11. The evaluation owner and model owner disagree on a severe failure.
12. Product teams want to buy priority during overload.
13. A new accelerator doubles compute and leaves network bandwidth unchanged.
14. The company requires a 30% power reduction next quarter.
15. The senior sponsor asks to remove abstention because users dislike it.

For each retry, state what changed, what remains invariant, what loses capacity, who decides, and what evidence confirms the new plan.

---

*Related: [train and serve a reasoning model under a fixed compute budget](/questions/design-reasoning-model-fixed-budget/), [test-time compute, search, and verifiers](/concepts/test-time-compute-search-verifiers/), [design a production LLM inference service](/questions/design-production-llm-inference-service/), [senior through senior-principal ML scope](/guides/l5-vs-l6-faang-ml/), and [Transformer compute and memory accounting](/concepts/transformer-compute-memory-accounting/).*
