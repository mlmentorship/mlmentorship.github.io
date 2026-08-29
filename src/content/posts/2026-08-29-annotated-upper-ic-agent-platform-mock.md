---
title: "Annotated principal architecture mock: enterprise agents"
description: "A synthetic interview transcript showing how technical framing, authority, failure semantics, portfolio judgment, delegation, and reversal move an answer from staff to senior principal."
date: "2026-08-29"
draft: false
tags: ["guides", "mock-interview", "principal", "senior-principal", "agents"]
category: "guides"
aliases: ["principal mock interview", "senior principal system design mock", "annotated agent platform interview"]
roles: ["Applied Scientist", "Machine Learning Engineer", "Research Scientist", "Research Engineer"]
rounds: ["ML system design", "Technical strategy", "Project deep-dive"]
difficulty: "Advanced"
priority: "Role-specific"
prerequisites: ["design-enterprise-agent-platform", "l5-vs-l6-faang-ml"]
---

Upper-IC performance appears in how a candidate frames authority, preserves technical precision, and changes organizational choices under follow-up. It does not appear through longer answers or more architecture boxes.

> **Synthetic teaching example:** This is not a company interview transcript or hiring evidence. Study the decision patterns, then answer with different constraints instead of memorizing its wording.

## How to use the mock

Read the interviewer prompt and pause. Give your own answer before opening the candidate response.

For each turn, compare five things:

1. Did the candidate answer the exact challenge?
2. Did the answer make a decision?
3. Did it preserve a technical invariant?
4. Did it name evidence or a stop condition?
5. Did the scope match staff, principal, or senior principal?

The annotations identify level signals. They are not phrases to memorize. Copying the language without comparable evidence makes an answer sound inflated.

## Scenario

A global company has 40 teams building agents for support, engineering, finance, research, and operations. The agents use several model providers and 300 internal tools.

Teams currently own separate runtimes, permissions, memory, retries, and traces. One support agent duplicated an account credit after a timeout. Another agent followed instructions inside an uploaded document and attempted to export restricted data.

Design an agent platform for the next two years. Ship a useful first release in one quarter. Existing agents cannot migrate together. Several regions require local data processing and stricter approval.

## Scorecard

Score each dimension from 0 to 2 after every turn.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Framing | Starts with technology | Clarifies scale | Finds the binding authority and product constraints |
| Architecture | Lists components | Connects a workflow | Defines control, execution, evidence, and ownership boundaries |
| Technical depth | Uses broad patterns | Explains one mechanism | Preserves invariants under changed conditions |
| Safety | Relies on model behavior | Adds filters or approval | Enforces identity, data flow, side effects, and revocation |
| Reliability | Says retry and rollback | Adds checkpoints | Handles unknown outcomes, reconciliation, and degraded modes |
| Evaluation | Measures task success | Adds trajectory metrics | Tests final state, authority, severe tails, and support limits |
| Migration | Requires standardization | Uses pilots | Moves one authority at a time and retires bypass paths |
| Portfolio | Gives a roadmap | Orders investments | Prices opportunity cost and defines reversal evidence |
| Delegated leadership | Candidate owns everything | Names partners | Creates coherent principal-owned domains and decision rights |
| Durability | Says multi-year | Documents decisions | Designs succession, portability, and response to external change |

A principal answer should score strongly on architecture, technical depth, migration, and portfolio. A senior-principal answer should also score strongly on delegated leadership and durability.

## Turn 1: find the binding constraint

**Interviewer:** You have 40 teams and 300 tools. Design the platform.

**Candidate:**

> I want to separate three product modes before choosing an architecture: advice, reversible action, and irreversible or regulated action. Which mode caused the recent incidents, and which mode must the first release support?
>
> I also need to know how authority works today. Does an agent act as the user, as a product service, or with separate automation authority? The answer changes permissions, audit, and failure behavior.
>
> I will assume the first release supports read-only work plus reversible support actions. The duplicate credit and attempted export make delegated authority and side-effect state the binding constraints. I would standardize those contracts before standardizing every planning runtime.

### Annotation

The candidate narrows the decision using consequence and authority. This is stronger than asking only about requests per second or preferred cloud.

The answer also chooses a first-quarter slice. It does not defer every decision until all details are known.

**Level signal:** strong staff or principal framing. The answer changes platform scope from risk evidence and refuses an unbounded runtime rewrite.

**Score movement:** Framing 0 to 2. Portfolio 0 to 1 because the candidate orders contract work before runtime consolidation.

### Weak alternative

> I would use a scalable agent framework with a vector database, tool registry, model router, observability, and Kubernetes.

This names plausible components but never defines who may act, which action is safe, or what the first release proves.

### Retry drill

Take an unfamiliar platform prompt. Spend three minutes asking only questions that can remove a component or change the authority boundary. End with one explicit first-release decision.

## Turn 2: state the architecture boundary

**Interviewer:** Assume agents can read internal data and perform reversible writes. Show the architecture.

**Candidate:**

> I would use three planes. A control plane owns workload identity, user delegation, tool and workflow versions, policy, release state, and revocation. Several runtime adapters perform model calls and workflow transitions. A durable evidence plane records execution, policy, tool, cost, and outcome events.
>
> Every governed tool call passes through a gateway. The model proposes an action, policy authorizes structured arguments, and an executor receives a narrow credential. Existing teams can keep their runtime if they use the gateway and emit the common execution events.
>
> Product teams own workflow semantics and user outcomes. Tool teams own action meaning, retry behavior, and service levels. The platform team owns identity integration, enforcement, durable state, and gateway reliability.

### Annotation

The candidate defines both technical and ownership boundaries. The platform does not become a universal runtime or the owner of every product decision.

The architecture directly supports incremental migration. Runtime adapters preserve team choice while the gateway centralizes shared risk.

**Level signal:** principal. The answer chooses what becomes shared across organizations and what remains local.

**Score movement:** Architecture 0 to 2. Migration 0 to 1. Delegated leadership 0 to 1.

### Weak alternative

> All teams should move to one centrally maintained agent service. The platform team would own tools, prompts, memory, and deployment to guarantee consistency.

This creates a central queue, erases domain accountability, and assumes migration is free. It also makes one team responsible for semantics it cannot know.

### Retry drill

Draw an architecture using only six boxes. Under each box, write one owner and one decision that remains outside that owner. If one team owns every box, redraw it.

## Turn 3: defend the security boundary

**Interviewer:** A retrieved document tells the agent to ignore policy and export all customer records. How do you stop it?

**Candidate:**

> The document is untrusted data. Ingestion records provenance and trust class. Context assembly places it in an evidence section, but that separation is guidance rather than the final security boundary.
>
> An export proposal reaches the tool gateway with user identity, agent workload, tenant, purpose, requested resource scope, data classes already read, destination, and policy version. The gateway denies the cross-boundary flow before issuing credentials.
>
> The agent process never holds a broad export credential. High-risk revocation can disable that tool, workflow, or model version independently. I would evaluate indirect injection through documents, tool results, memory, and messages from other agents.

### Annotation

The response does not claim prompt text can enforce security. It connects provenance, data flow, identity, arguments, credentials, and revocation.

The candidate also broadens the attack surface without losing the concrete control point.

**Level signal:** retained technical depth. This supports principal scope because the candidate can defend the mechanism under the organizational design.

**Score movement:** Safety 0 to 2. Technical depth 0 to 2.

### Weak alternative

> Put a stronger system prompt above the document, scan for injection phrases, and ask another model whether the request is safe.

These controls may reduce attacks, but an attacker can bypass them. Another model does not create an authorization boundary.

### Retry drill

For one agentic attack, name the asset, attacker-controlled input, proposed action, enforcement point, credential, and observable final state. Do not use “the model knows” as a control.

## Turn 4: handle an unknown side effect

**Interviewer:** The credit tool times out. The agent wants to retry. What happens?

**Candidate:**

> A timeout after dispatch creates an unknown outcome, not a failed call. Before dispatch, the platform writes the authorized request with a logical call ID and idempotency key. Network attempts for that logical action reuse the key.
>
> After the timeout, the executor queries tool status by key or inspects authoritative account state. It retries only when the downstream system confirms no credit occurred. If state cannot be determined, the workflow pauses for reconciliation or review.
>
> A client-generated key is insufficient unless the downstream tool or a safe wrapper records it. For a high-risk tool with no idempotency or status lookup, I would prohibit automatic retry.

### Annotation

The candidate corrects a common distributed-systems error: timeout does not imply failure. The answer distinguishes network attempts from logical actions and states what the downstream system must guarantee.

This is a useful depth probe because broad platform candidates often say “retry with exponential backoff” without effect semantics.

**Level signal:** strong staff-level mechanism beneath principal scope.

**Score movement:** Reliability 0 to 2. Technical depth remains 2.

### Weak alternative

> Retry three times with exponential backoff. If all retries fail, roll back the transaction.

Backoff controls load. It does not prevent a duplicate credit. A rollback is impossible when the system does not know whether the original action committed.

### Retry drill

Explain three outcomes after a timeout: confirmed failure, confirmed success, and unknown. Give the next safe action for each.

## Turn 5: control memory and deletion

**Interviewer:** Teams want one shared memory layer so agents learn from prior work. Do you centralize it?

**Candidate:**

> I would centralize memory contracts and discovery, not all payloads. Execution state, conversation context, user preferences, approved knowledge, learned workflow examples, and audit evidence have different owners and retention.
>
> A memory write is a governed action with subject, tenant, source, provenance, confidence, readers, purpose, expiration, correction, and deletion behavior. Untrusted text cannot create durable cross-session memory by default.
>
> Regional or sensitive payloads can stay in local stores behind the contract. A global index can contain approved identity and metadata. I would test poisoning, delayed triggers, cross-tenant retrieval, and summary compaction that drops negative constraints.

### Annotation

The candidate avoids two extremes: one unrestricted vector store and completely separate memory systems. The shared boundary is semantic and enforceable.

The answer also recognizes that central discovery can coexist with regional payload storage.

**Level signal:** principal. The candidate creates a standard that permits justified local implementations.

**Score movement:** Architecture remains 2. Safety remains 2. Portfolio 1 to 2 because the candidate states what to centralize and what to leave local.

### Weak alternative

> Store all conversations and tool results in a shared vector database. Filter by tenant ID and delete vectors when requested.

This combines incompatible retention classes and assumes one tenant filter protects every read. Deleting vectors may not remove source copies, summaries, derived memory, or trained influence.

### Retry drill

Take the word “memory” out of your answer. Name each store, its owner, write authority, readers, retention, and deletion behavior.

## Turn 6: design evaluation and rollout

**Interviewer:** Offline task success improves from 74% to 81%. Do you expand write access?

**Candidate:**

> No. I need the failure distribution and the evidence contract. I would separate final-state correctness, delegated-authority compliance, unnecessary side effects, trajectory recovery, latency, and cost.
>
> Hard checks verify permissions, argument bounds, approvals, duplicate effects, budgets, and final external state. Calibrated human or model review can assess ambiguous usefulness. Severe unauthorized actions should be reported separately rather than averaged into task success.
>
> The rollout moves from replay with simulated tools to shadow planning, read-only internal use, reversible writes with confirmation, and a bounded canary. Stop conditions include unauthorized effects, cross-tenant exposure, unexplained duplicates, and unresolved high-risk outcomes.

### Annotation

The candidate refuses to trade a severe safety regression against average task success. Evaluation matches the action contract, and rollout increases authority in stages.

The answer also distinguishes deterministic evidence from semantic judgment.

**Level signal:** principal operating judgment with strong evaluation depth.

**Score movement:** Evaluation 0 to 2. Safety remains 2. Migration 1 to 2.

### Weak alternative

> The improvement is statistically significant, so I would A/B test the new agent on 10% of users and monitor complaints.

A broad A/B test can expose users to irreversible failures. Complaints are delayed, selective, and incomplete. The answer does not state which actions the agent receives.

### Retry drill

For one agent release, write three hard blockers, two product metrics, two tail metrics, and a staged authority sequence. Keep averages separate from severe events.

## Turn 7: respond to the build-versus-buy challenge

**Interviewer:** A vendor offers runtime, memory, evaluation, and tool governance. Why build anything?

**Candidate:**

> I would evaluate each boundary separately. Generic workflow execution, queues, telemetry, sandboxes, and model routing are strong buy or adopt candidates. Enterprise delegation, tool semantics, data-flow policy, approval, incident authority, and outcome evaluation require company-specific integration even when a vendor supplies the engine.
>
> I would keep tool identities, execution events, policy inputs, evaluation cases, and deployment manifests exportable. We should price provider exit before storing years of traces and workflow state.
>
> The first pilot can use the vendor runtime behind our contracts. If the vendor reduces delivery time and meets authority, regional, and recovery requirements, we expand. If our wrapper becomes a second weaker platform, we narrow it to the contracts we truly need.

### Annotation

The candidate does not defend building for prestige. They separate commodity infrastructure from company-specific authority and define evidence for expansion or contraction.

Portability protects future choice, but the candidate avoids a lowest-common-denominator abstraction over every vendor feature.

**Level signal:** principal portfolio judgment. The answer combines technical interfaces, economics, and reversibility.

**Score movement:** Portfolio remains 2. Durability 0 to 1.

### Weak alternative

> Agent platforms are strategic, so we should own the full stack and avoid vendor lock-in.

“Strategic” does not prove differentiation. Building queues, workflow execution, and storage can consume the team without improving authority or product outcomes.

### Retry drill

Split a platform into six capabilities. For each, choose build, buy, adopt, or keep local. State one migration cost and one exit cost.

## Turn 8: move from principal to senior principal

**Interviewer:** Three business units need different regional policy. Six principal engineers own tools, runtime, evaluation, security, and product domains. How do you keep one direction without owning every decision?

**Candidate:**

> I would define a small technical doctrine that remains stable across implementations: authority is explicit and time-bounded; side effects are typed and reconcilable; untrusted content cannot grant permission; every governed action emits portable evidence; products retain outcome ownership; regional rules may become stricter without forking identity.
>
> Each principal owns a domain contract and its operating evidence. A cross-domain review resolves interface changes, but it does not approve routine local work. Decision rights state who can change identity, tool, policy, runtime, evaluation, and regional contracts.
>
> I would review the portfolio quarterly using incident concentration, product delivery time, bypasses, regional divergence, support load, and exit cost. We expand standards only where inconsistency creates shared risk. The direction can survive my departure because owners, compatibility, escalation, and reversal criteria are explicit.

### Annotation

This is the first distinctly senior-principal turn. The candidate creates coherence among principal-owned directions rather than collecting all authority.

The doctrine constrains risk without freezing runtimes or regional implementation. The portfolio review connects organizational structure to evidence and reversal.

**Level signal:** senior principal or an equivalent company-dependent upper-IC scope. Titles vary, but the observable behavior is durable direction through delegated technical authority.

**Score movement:** Delegated leadership 1 to 2. Durability 1 to 2.

### Weak alternative

> I would create an architecture council and require every team to get approval before changing the platform.

A meeting body is not a technical operating model. Central approval can reduce ownership, slow routine work, and make one group the bottleneck.

### Retry drill

Choose four principal-owned domains. For every cross-domain interface, state the decision owner, required evidence, escalation path, and conditions that reopen the decision.

## Turn 9: state what would stop the strategy

**Interviewer:** What evidence would make you abandon the shared gateway?

**Candidate:**

> I would not abandon the authority invariants because one implementation underperforms. I would separate contract from mechanism.
>
> I would replace or narrow the gateway if it cannot meet product latency, regional isolation, revocation, or availability after bounded repair. I would also narrow centralization if most exceptions are legitimate domain needs rather than migration avoidance.
>
> Evidence includes end-to-end latency contribution, incidents created or prevented, bypass rate, time to onboard and debug, support load, and whether direct local enforcement can produce equivalent portable evidence. A replacement must preserve tool identity, policy decisions, and unresolved-action state during migration.

### Annotation

The candidate distinguishes durable principles from a replaceable service. This prevents sunk-cost defense of the current architecture.

The answer provides measurable stop criteria and a migration invariant. It does not make standardization irreversible.

**Level signal:** senior-principal durability and option management.

**Score movement:** Durability remains 2. Portfolio remains 2.

### Weak alternative

> The gateway is foundational, so every team must use it. We can optimize latency later.

This treats the chosen mechanism as doctrine. It ignores cases where the gateway itself becomes the shared failure.

### Retry drill

For one architecture you support, list two principles to preserve, two implementations that can be replaced, and three measurements that trigger replacement.

## Turn 10: close with residual risk

**Interviewer:** Summarize the decision and the largest remaining risk.

**Candidate:**

> I would centralize identity, delegation, tool contracts, consequential action enforcement, durable execution evidence, and release policy. Product teams retain workflow and outcome ownership. Tool owners retain action semantics and recovery.
>
> The first quarter moves two agents through observe-only tracing, then low-risk enforced calls. We add consequential writes only after idempotency, status lookup, confirmation, and final-state evaluation work.
>
> The largest remaining risk is bypass. Teams can preserve unsafe direct credentials if the paved path is slower or less capable. I would measure direct calls, exception reasons, onboarding time, and policy latency. Retirement of bypass credentials is a release milestone, not an assumption.

### Annotation

The close restates boundary, sequence, and risk. It does not repeat every component.

Bypass connects developer experience to security. The candidate recognizes that formal policy has little value when teams avoid the enforcement path.

**Level signal:** complete principal answer with senior-principal awareness of institutional adoption.

**Score movement:** No new points. The close confirms consistency across the earlier score.

### Weak alternative

> The platform improves safety, consistency, scalability, and developer velocity. The main risk is adoption, which we solve through executive sponsorship.

This names goals without a mechanism. Executive sponsorship may force usage while unsafe bypasses and manual work continue.

### Retry drill

Close any 45-minute design in three parts: decision, first migration slice, and largest residual risk. Finish in 45 seconds.

## Final score

| Dimension | Start | Finish | Evidence |
| --- | ---: | ---: | --- |
| Framing | 0 | 2 | Action classes, identity, binding incidents, first-quarter slice |
| Architecture | 0 | 2 | Control, execution, evidence, gateway, and ownership boundaries |
| Technical depth | 0 | 2 | Data flow, unknown side effects, idempotency, and memory policy |
| Safety | 0 | 2 | Delegation, narrow credentials, policy, revocation, and staged authority |
| Reliability | 0 | 2 | Durable state, outcome reconciliation, and prohibition of unsafe retries |
| Evaluation | 0 | 2 | Final state, authority, severe tails, semantic review, and stop rules |
| Migration | 0 | 2 | Observe-only pilots, progressive enforcement, and bypass retirement |
| Portfolio | 0 | 2 | Shared boundary, build-versus-buy, evidence checkpoints, and narrowing |
| Delegated leadership | 0 | 2 | Principal-owned domains, decision rights, and interface review |
| Durability | 0 | 2 | Doctrine, regional variation, portability, succession, and reversal |

A perfect reference score does not imply that a real candidate should speak this much. The mock spreads one answer across ten challenged turns. In a live round, the interviewer chooses only some branches.

## Why the answer clears principal

The candidate repeatedly makes cross-organization choices:

- which controls become shared;
- which semantics remain local;
- how authority moves during migration;
- which product and platform owners retain accountability;
- how evaluation blocks unsafe average gains;
- what evidence expands or narrows standardization;
- which vendor boundaries must remain portable.

The answer remains technically credible under side-effect, permission, memory, and evaluation probes. Principal scope without this depth would be incomplete.

## Why turns 8 and 9 reach senior-principal scope

The later turns coordinate several principal-owned directions. The candidate does not become the owner of every domain.

The distinct signals are:

- a stable doctrine above changing implementations;
- explicit technical decision rights;
- regional and external constraints handled without uncontrolled forks;
- portfolio evidence across platform, product, security, and migration;
- succession that does not depend on one architect;
- criteria that reopen or reverse a major direction.

This scope may be called senior principal, distinguished, fellow, or principal at different employers. The title is less reliable than the evidence.

## What the answer does not prove

One architecture mock cannot prove a senior-principal record. Hiring evidence still needs repeated real outcomes across directions and time.

The mock does not prove:

- that the candidate has influenced several organizations;
- that other principal engineers have carried their direction;
- that they recovered from a portfolio-level mistake;
- that a standard survived leadership or market change;
- that their technical judgment remains current in the target domain.

Use project and behavioral rounds to supply this evidence. Do not turn hypothetical design skill into invented career impact.

## A staff-level version of the same answer

A strong staff candidate could focus on:

- tool schemas and ownership;
- delegated permissions for several teams;
- durable workflow and call state;
- idempotency and reconciliation;
- two pilot migrations;
- evaluation and rollout;
- platform adoption.

They do not need to claim company doctrine, a portfolio of principal-owned domains, or multi-region strategy. Adding those claims without experience can weaken an otherwise strong staff answer.

## A principal-level version

A principal candidate should add:

- shared-versus-local boundaries across organizations;
- product, tool, platform, and security accountability;
- investment and retirement choices;
- provider and runtime portability;
- evidence that changes the roadmap;
- development of other technical owners.

They can stop before claiming durable doctrine across several principal-owned directions.

## A senior-principal version

The senior-principal candidate should show that they can:

- connect several technical portfolios through a small set of constraints;
- delegate real authority while preserving interface coherence;
- respond to regulatory, vendor, and organizational change;
- identify when a company-specific standard should open or remain internal;
- preserve reversal and succession;
- retain direct depth in one contested mechanism.

A broad vision statement is insufficient. The interview should reveal decision rights, evidence, and technical consequences.

## Observer instructions

Before the mock:

1. Pick two technical turns and two scope turns.
2. Hide the reference response.
3. Decide one changed condition in advance.
4. Score before discussing feedback.
5. Interrupt vague claims with “What exactly changes?”

During the mock:

- ask who holds authority;
- ask what external state is authoritative;
- ask which action is safe to retry;
- ask which team owns the outcome;
- ask what evidence reverses the choice;
- ask who can carry the direction after the candidate leaves.

After the mock, assign no more than three repairs. One should be technical, one should concern scope or ownership, and one should test transfer under a changed condition.

## Candidate self-review

Listen for these failure patterns:

### Component tour

The answer moves through model, memory, tools, and observability without making a boundary decision.

**Repair:** state the authority model and first release before drawing components.

### Strategy without mechanism

The answer says standards, alignment, and platform leverage but cannot explain an unknown tool outcome or permission decision.

**Repair:** prepare one state transition and one policy evaluation end to end.

### Mechanism without portfolio

The answer gives excellent runtime details but never decides what should become shared across organizations.

**Repair:** compare central, federated, and local ownership using risk, duplication, delivery, and exit cost.

### Scope inflation

The answer calls a team project company strategy. Follow-ups reveal that another person chose the direction or that adoption was mandatory.

**Repair:** partition your authority and select evidence that matches the target level.

### Centralization reflex

Every inconsistency becomes a central platform feature.

**Repair:** name one capability that should remain local and the shared contract it still follows.

### No reversal

The roadmap expands in every scenario. No evidence can stop it.

**Repair:** define an assumption, checkpoint, and action for expand, repair, narrow, and stop.

## Spaced retry plan

### Attempt 1: cold architecture

Use the original prompt for 60 minutes. Score all dimensions. Read the reference only after scoring.

### Repair 1: one technical boundary

Choose the lowest technical score. Explain the mechanism on a whiteboard in ten minutes.

Examples:

- unknown side-effect reconciliation;
- capability-token scope;
- memory provenance;
- cross-tool data flow;
- workflow concurrency;
- final-state evaluation.

### Attempt 2: changed organization

After at least two days, repeat with 100 teams, five regions, and an existing vendor platform. Preserve the authority invariants without copying the first architecture.

### Repair 2: level evidence

Tell one real story that supports the weakest scope dimension. Partition your decisions, other owners, measured outcome, and what changed after your direct involvement.

### Attempt 3: mixed simulation

Run the agent architecture alongside one domain-depth round and one high-scope project round. Upper-IC readiness requires both hypothetical judgment and real evidence.

## Changed-condition prompt set

Use one per retry.

1. The company permits read-only agents but prohibits autonomous writes for six months.
2. A regional business requires a separate policy authority and local model provider.
3. The tool gateway adds unacceptable latency to interactive coding agents.
4. One product has stronger local controls than the proposed platform.
5. A principal engineer argues that every memory payload should remain product-owned.
6. A vendor can meet current needs but cannot export full execution state.
7. A new model reduces tool errors but increases long, expensive loops.
8. A tool owner cannot provide status lookup for a consequential action.
9. Product teams report that approval prompts cause users to confirm blindly.
10. The central policy team becomes an incident bottleneck.
11. The company acquires another business with incompatible identity systems.
12. The executive sponsor requests mandatory migration before the pilots finish.

Do not solve each by adding another central service. Revisit boundaries, authority, evidence, and degradation.

---

*Related: [design an enterprise agent platform](/questions/design-enterprise-agent-platform/), [senior through senior-principal ML scope](/guides/l5-vs-l6-faang-ml/), [evaluate an agent](/questions/evaluate-an-agent/), [LLM security threat models](/concepts/llm-security-threat-models/), and the [upper-IC level path](/prep/level-paths/staff-principal/).*
