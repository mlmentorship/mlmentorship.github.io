# ML and AI interview landscape, August 2026

## Evidence review, role-specific preparation system, and mlmentorship gap audit

**Research cutoff:** August 27, 2026

**Scope:** ML and AI engineering, science, research, systems, product, data, safety, multimodal, and leadership roles at frontier AI labs and major technology companies.

**Ethics:** This review describes public formats and representative competency archetypes. It does not reproduce leaked prompts, NDA-covered material, or proprietary question banks.

**Implementation update, August 29, 2026:** The declared P0 concept set is complete, and the repository now includes nine deep system-design cases. Upper-IC preparation extends through company-dependent senior-principal scope with three annotated transfer mocks, domain cases, simulations, Workbook routing, and story evidence.

---

## Executive conclusions

1. **There is no single ML interview loop.** Role family, team mandate, level, and assessment format predict the loop better than the company name or job title.
2. **Traditional coding remains a gate.** Recent reports and official guides still show data structures and algorithms in product MLE, Applied Scientist, Research Engineer, Research Scientist, deep-learning, and infrastructure loops. SQL remains central in product-data and experimentation roles.
3. **New formats are additive, not substitutes.** AI-assisted codebase work, realistic debugging, research work samples, presentations, and take-home defenses are being added while unaided coding and fundamentals persist.
4. **The frontier-lab bar is hybrid.** Current roles increasingly combine scientific judgment, production code, evaluation, distributed systems, model behavior, and operational ownership.
5. **ML system design is no longer a model-selection exercise.** Strong answers connect objective, data, labels, baseline, model, offline evaluation, serving, cost, monitoring, online evidence, fallback, and iteration.
6. **Project and research discussions are level interviews.** Interviewers use them to infer actual ownership, scope, technical judgment, failed approaches, influence, and whether the candidate's claimed impact is defensible.
7. **AI-use policy is part of the assessment.** OpenAI is assessment-specific; Anthropic and Google DeepMind default to no AI unless expressly permitted; Meta authorizes a built-in assistant for select roles; Microsoft, Google, Uber, and NVIDIA prohibit unapproved outside help. Silence is not permission.
8. **The most frequent failure is shallow execution under pressure.** Common misses include solving the wrong variant, failing edge cases, accepting generated code without understanding it, skipping problem framing, weak testing, running out of time, and underpreparing behavioral evidence.
9. **Preparation should be evidence-driven.** A candidate should map the actual loop, run cold timed baselines, allocate work to weighted deficits, use delayed closed-book retries, and complete at least one full simulation before declaring readiness.
10. **mlmentorship is already strong on the modern ML core.** Its largest remaining risks are incomplete general coding and practical software gates, no Product Data Scientist path, a capped eight-week workload model, a small company registry, and too few deep system-design cases.

---

## 1. Research method and confidence

### 1.1 Evidence hierarchy

| Tier | Source type | How it is used |
| --- | --- | --- |
| **T1A** | Official company interview guide or candidate policy | Authoritative for published stages, timing, tools, and candidate rules. |
| **T1B** | Current official job description | Authoritative for role scope and expected work, but not proof of interview rounds. |
| **T2** | Method-disclosed synthesis using current engineers or multiple candidate reports | Useful for likely round composition when clearly labeled as reported. |
| **T3** | Detailed first-person candidate account | Evidence for one role, team, geography, and date only. |
| **T4** | Anonymous aggregate, gated review, search snippet, or opaque commercial guide | Discovery or weak corroboration only. Never sufficient for a firm claim. |

### 1.2 Source set

The review covered:

- frontier organizations: OpenAI, Anthropic, Google DeepMind, Meta AI/FAIR, xAI/SpaceXAI, Microsoft AI, Mistral AI, Cohere, Scale AI, and NVIDIA research/model teams;
- major technology companies: Google, Meta product ML and data science, Amazon/AWS, Microsoft, Apple, Netflix, Uber, ByteDance/TikTok, LinkedIn, Stripe, Airbnb, and Databricks;
- role evidence from Pinterest, Canva, Physical Intelligence, ElevenLabs, and current model, safety, retrieval, robotics, and infrastructure postings;
- recent first-person and methods-based material from interviewing.io, personal blogs, Reddit, Glassdoor, Blind, Hacker News, and X where directly accessible.

### 1.3 Important limitations

- A company page often describes a generic process while teams choose the technical stations.
- Candidate reports are self-selected and usually omit interviewer rubrics.
- Candidates rarely receive reliable rejection reasons. Their causal interpretation of a rejection is a hypothesis.
- Glassdoor, Blind, X, and some Substack pages are partially gated. Search snippets were not treated as evidence.
- Current job descriptions show what a team needs, not necessarily how it tests it.
- Recruiter-provided instructions for the exact role always supersede this report.

---

## 2. What changed in 2025-2026

### 2.1 Algorithms survived AI

The strongest cross-company data point is interviewing.io's 2025 survey of 67 first-hand respondents, including 52 at FAANG companies. None of the 52 reported that their company had removed algorithmic interviews. Fifty-eight percent said they had changed the questions, commonly by adding custom context, multiple stages, variants, and deeper follow-ups.

The practical conclusion is not "memorize more company-tagged questions." It is:

- recognize underlying patterns despite unfamiliar terminology;
- clarify the contract before coding;
- produce a correct baseline quickly;
- test edge and boundary cases;
- explain complexity and alternatives;
- adapt when a constraint changes;
- debug without immediately reaching for external help.

### 2.2 AI-assisted interviewing became a separate skill

Three policies illustrate the market:

- **Meta:** select technical interviews include an authorized CoderPad assistant. The candidate remains responsible for planning, review, tests, and explanation.
- **OpenAI:** the official guide says tool rules vary by assessment. Some formats intentionally permit AI; others assess independent problem solving.
- **Anthropic:** take-homes and live interviews are no-AI by default unless the instructions explicitly permit it. The performance-team exercise is a documented exception.

The new station tests codebase mapping, decomposition, bounded delegation, critical review, verification, and recovery when the model is wrong. It does not remove the need to code independently.

### 2.3 Work samples moved closer to real work

Public and reported formats include:

- multi-file repositories with failing tests;
- progressive specifications and hidden evaluators;
- training-loop diagnosis;
- black-box model investigation;
- accelerator traces and optimization;
- notebooks, APIs, datasets, or paper-based research tasks;
- technical presentations and artifact defenses;
- paid work trials at some startups.

The artifact alone is increasingly insufficient. Reviewers add a defense, extension, code review, changed requirement, or presentation to determine whether the candidate owns the work.

### 2.4 Questions probe one level deeper

AI-generated surface fluency has increased the value of follow-ups such as:

- Why does this mechanism work?
- Which assumption is doing the work?
- What breaks at the boundary?
- What would change your decision?
- What does this line do, and what happens if it is removed?
- Which result distinguishes your hypothesis from the strongest alternative?
- How would this fail operationally?

### 2.5 Exact fit matters more in tight headcount markets

Recent senior-candidate accounts consistently report that a technically good loop can still lose to a candidate whose prior work matches the live team mandate. Preparation cannot manufacture years of relevant work, but it can make existing fit legible through:

- a role-specific resume;
- one evidence-dense project or research contribution;
- a concise project narrative;
- relevant artifacts;
- a clear account of why this team and why now;
- references who directly observed the work.

---

## 3. The canonical end-to-end hiring funnel

A modern process is usually assembled from the following stages. Not every company uses every stage.

1. **Application review**
   - Resume, publications, portfolio, GitHub, exceptional-work statement, or referral.
   - Signal: relevant scope, trajectory, proof, and exact team fit.
2. **Recruiter or hiring-manager screen**
   - Background, motivation, role calibration, logistics, level, and sometimes short technical questions.
3. **Pre-onsite assessment**
   - Online assessment, live coding, take-home, research screen, or one to three technical calls.
4. **Final loop**
   - Four to eight stations selected from coding, ML implementation, math, ML breadth, research, system design, product/experimentation, project presentation, behavioral, values, and manager conversations.
5. **Decision and calibration**
   - Interview debrief, hiring committee or bar-raiser process, level decision, and sometimes team matching.
6. **References**
   - Optional at OpenAI, mandatory at Mistral, explicit in some Anthropic programs, Databricks, and multiple reported frontier processes.
7. **Team match and offer**
   - Team conversations, role scoping, level and compensation, offer comparison, negotiation, and start-date alignment.

The practical risk is optimizing only for stage 4. Strong candidates also fail at application fit, recruiter narrative, references, team match, and offer coordination.

---

## 4. Interview-station map

### 4.1 Station matrix

| Station | What it tests | Common role families | Representative public-safe archetypes | Frequent failure |
| --- | --- | --- | --- | --- |
| **Application and proof of work** | Relevance, trajectory, ownership, verifiable evidence | All, especially frontier labs and senior roles | Summarize one exceptional contribution in 100 words; attach the artifact that best supports it | Technology list with no decision, evidence, or attributable outcome |
| **Recruiter/HM screen** | Role fit, motivation, communication, level, resume truth | All | Walk through the career as an argument for this role; explain why this team rather than generic AI | Chronological biography, generic enthusiasm, unclear role target |
| **General DSA coding** | Problem decomposition, data structures, algorithms, correctness, complexity | MLE, RE, AS, RS, infra | Streaming aggregation, graph traversal, dependency ordering, interval processing, cache or parsing problems | Memorized pattern does not transfer to variant; no tests; poor pace |
| **Progressive practical/OOP coding** | APIs, state, invariants, extensibility, ambiguity, concurrency | SWE-leaning MLE/RE, OpenAI/Anthropic-style reports, startups | Build a small stateful service in levels, then add concurrency, failure handling, or scale | Finishes functions but makes brittle domain or state choices |
| **ML implementation** | Tensor contracts, numerical stability, model mechanics, testing | RE, RS, foundation-model, MLE | Implement a decoder block, KV cache, beam search, LoRA adapter, metrics accumulator, or classical algorithm | Framework familiarity without from-scratch fluency; shape bugs |
| **ML debugging** | Diagnostic order, causal isolation, measurements, code reading | RE, post-training, systems, applied ML | Diagnose a model that will not learn, a loss spike, train/eval skew, or distributed-only failure | Random fixes, no first-bad-transition analysis, treats symptom as cause |
| **SQL and data manipulation** | Data semantics, joins, windows, aggregation, experiment tables | Product DS, experimentation, some MLE | Define a metric from event tables; detect exposure/assignment mismatch; build cohort retention | Correct syntax against the wrong unit or denominator |
| **Probability/statistics/math** | Derivation, assumptions, uncertainty, intuition, sanity checks | AS, RS, RE, DS, specialist research | Derive a loss, reason about estimators, sampling, confidence, KL, gradients, or matrix structure | Recites formula but cannot derive or interpret it |
| **ML breadth** | Mechanisms, alternatives, failure conditions, field fluency | Most ML roles | Loss choice, regularization, calibration, optimization, representations, modern architectures | Definitions with no mechanism or boundary condition |
| **ML model/system design** | Full lifecycle and product judgment | MLE, AS, RE, ranking, applied AI | Design fraud detection, recommendation, retrieval, moderation, forecasting, or an agent evaluation system | Jumps to a fashionable model before objective, data, labels, or baseline |
| **ML infrastructure/system design** | Scale, distributed state, reliability, throughput, cost | Platform MLE, RE, training, inference | Feature/data platform, training service, inference scheduler, model registry, fault recovery | Model-centric answer that ignores ordinary systems concerns |
| **General distributed-system design** | Storage, consistency, concurrency, queues, partitioning, recovery | SWE-heavy MLE/RE, platform and staff roles | Design a distributed cache, task scheduler, event pipeline, or versioned store | Names technologies without invariants or failure semantics |
| **Product and experimentation** | Metrics, causal validity, guardrails, ship decision | AS, DS, product MLE, ranking | Offline improves while online declines; define randomization, exposure, power, slices, and decision rule | Statistical significance without practical decision logic |
| **Research/paper discussion** | Claim/evidence separation, prior work, limitations, taste | RS, RE, safety, advanced AS | Critique a paper; defend a paper on the resume; identify the central threat to validity | Summary instead of critique; attacks peripheral details |
| **Research brainstorm/work sample** | Hypothesis generation, discriminating experiments, uncertainty | RS, safety, model behavior, RE | Given an observed behavior, generate competing hypotheses and minimal separating probes | One favorite explanation, no controls, no falsifiable prediction |
| **Project deep dive/job talk** | Scope, ownership, judgment, technical depth, impact | All senior roles; core for RS/AS/frontier | Present three consequential decisions, one failed approach, evidence, and bounded outcome | Project diary, inflated "we," no failed path, no attribution |
| **Behavioral/leadership** | Conflict, ambiguity, influence, failure, mentoring, judgment | All | Tell a decision story, then defend it under skeptical follow-up | Generic STAR shell, no real stakes, team activity without personal action |
| **Values/mission** | Independent judgment, principles under tension, ability to update | Anthropic/Mistral/frontier and leadership roles | Discuss a real tension with costs on both sides and evidence that changed a belief | Company slogans, fan behavior, or performative agreement |
| **AI-assisted codebase** | Navigation, planning, delegation, review, verification | Select Meta/OpenAI/startup formats | Extend an unfamiliar repository, diagnose tests, benchmark baseline, and review generated changes | Lets the agent own the plan; cannot explain code; skips diff and tests |
| **Take-home/work-trial defense** | Authorship, prioritization, code/research judgment, extension | Startups, selected lab/team processes | Explain what was omitted, defend trade-offs, respond to a new constraint, modify the artifact live | Over-polished artifact with no decision log or time-box disclosure |
| **References/team match/offer** | Independent evidence, live fit, manager/team quality, negotiation | Senior/frontier and committee-based hiring | Map claims to witnesses; evaluate first-year scope and team health; compare complete offers | Treats reference check as an offer; optimizes compensation before role/team |

### 4.2 What strong performance looks like across stations

A reusable answer pattern is:

1. clarify the objective and contract;
2. state assumptions and constraints;
3. propose the simplest viable baseline;
4. explain the mechanism or invariant;
5. identify alternatives and a decision criterion;
6. test or measure the result;
7. cover edge cases and failure modes;
8. update when the interviewer changes a condition;
9. close with the decision and residual risk.

This is not a script. Different stations emphasize different steps.

---

## 5. Role-family interview map

| Role family | Typical loop emphasis | Technical core | Specialist depth | Proof that carries weight |
| --- | --- | --- | --- | --- |
| **Product/generalist MLE** | DSA or practical coding, ML breadth, ML design, project, behavioral | Software correctness plus full ML lifecycle | Product/domain modeling and production operations | Shipped model with data, evaluation, serving, monitoring, and business outcome |
| **Applied Scientist** | Coding, math/stats, ML breadth, experiment/model design, project, leadership | Scientific reasoning plus product and shipping judgment | Domain literature, experimental design, causal validity | Research or modeling contribution that reached a real decision or deployment |
| **Research Scientist** | ML coding, math, technical breadth, research discussion, brainstorm, job talk | Original hypothesis and evidence plus executable experimentation | Research agenda and current subfield depth | One or two defensible papers/projects with code, ablations, limitations, and future direction |
| **Research Engineer** | Strong coding, ML implementation/debugging, systems, research depth, project | Production-grade experimental code and systems reasoning | Distributed training, inference, evals, or team-specific research | Difficult implementation, open-source work, reproducible system, or research-to-production result |
| **Foundation-model pre/post-training** | Python/ML coding, training debugging, math, data/reward/eval design, systems | Transformers, optimization, data, SFT/preference/RL methods, behavior | Graders, reward hacking, synthetic data, scaling, distributed execution | Reproducible training/eval loop with held-out families and failure analysis |
| **ML systems/distributed training** | Systems coding, quantitative design, incident debugging, project | Parallelism, collectives, memory, checkpoints, scheduler, reliability | Step-time/MFU analysis, state consistency, cluster failure | Scaling curves, profiler traces, recovery design, or framework contribution |
| **Inference/performance/kernel** | C++/Python/CUDA-style coding, profiling, architecture, estimation | Hardware hierarchy, roofline, kernels, batching, KV cache, quantization | Compiler/runtime details and numerical correctness | Before/after benchmark with traces, hardware disclosure, and correctness tests |
| **LLM/agent application engineer** | Production coding, architecture, eval/debugging, product/customer case | APIs, retrieval, tools, state, orchestration, observability, cost | Security, permissions, workflow integration, model failure | Deployed application with real evals, traces, adversarial tests, and fallbacks |
| **Safety/alignment/evals** | Coding/data analysis, experiment design, threat model, research, mission | Measurement, uncertainty, model behavior, misuse, control and safeguards | Alignment science, adaptive red teams, eval infrastructure, policy boundaries | Executable eval or threat model with false positives, residual risk, and mitigation evidence |
| **Product Data Scientist/experimentation** | SQL/Python, statistics, experiment/product case, behavioral | Metric semantics, causal inference, experimentation, communication | Marketplace/network effects, forecasting, heterogeneous treatment | Decision memo with reproducible analysis, sensitivity checks, and a clear recommendation |
| **Recommendations/search/ranking** | Coding, ranking fundamentals, ML design, experimentation, project | Retrieval, ranking losses, sampling, bias, ANN, online metrics | Multi-objective ecosystems, exploration, cold start, generative ranking | Two-stage system with temporal evaluation, latency, bias slices, and launch plan |
| **CV/multimodal/speech** | Coding, modality fundamentals, paper/model design, data/eval, deployment | Vision/audio representations, fusion, augmentation, robustness, metrics | ASR decoding, geometry, video, generative modeling, edge constraints | Dataset card, modality ablations, corruption/slice tests, latency and demo |
| **Robotics/embodied AI** | Python/C++, controls/robot learning, full-stack debugging, real-time design | Perception, policy, state estimation, control, sim-to-real, safety | Hardware interfaces, teleoperation/data, planning, latency/jitter | Logs and repeated trials, failure taxonomy, timing evidence, safety envelope |
| **Staff through senior-principal IC** | Architecture, technical strategy, high-scope project defense, influence, and retained domain depth | Same domain bar plus problem selection, delegated authority, and organizational leverage | Cross-team standards, multi-year portfolios, external change, succession, and reversal | Repeated impact, wrong bets, principal-owned directions, adaptable standards, and measurable adoption |
| **ML/research manager** | Technical design, people/execution cases, hiring/coaching, portfolio judgment | Enough depth to challenge architecture and science | Team building, performance, roadmap and resource allocation | Specific examples of hiring, feedback, underperformance, organizational change, and delivery |

### 5.1 Shared core versus specialist overlays

Every candidate needs some version of:

- **C1: implementation** - correct, tested code and debugging;
- **C2: ML/math** - mechanisms, assumptions, and uncertainty;
- **C3: data/evaluation** - labels, splits, metrics, leakage, slices, and experiments;
- **C4: systems** - scale, reliability, latency, cost, monitoring, security, and fallback;
- **C5: ownership** - decisions, conflict, failure, influence, and measurable impact;
- **C6: dual-mode AI fluency** - independent competence plus responsible AI collaboration when authorized.

The candidate should then add one primary specialist overlay and at most one adjacent overlay. Preparing for "AI in general" creates breadth without a hiring thesis.

---

## 6. Frontier AI lab process overview

### 6.1 Company map

| Organization | Officially confirmed | Currently reported or role-specific | Confidence and caution |
| --- | --- | --- | --- |
| **OpenAI** | Application review, introductory call, one or more skills assessments, then usually 4-6 hours with 4-6 people over 1-2 days. Assessments may include pair coding, take-homes, and technical tests. Engineering is evaluated on design, code quality, performance, tests, communication, and collaboration. References may be requested. | Current method reports describe practical multi-part coding, system design, project presentation, leadership/collaboration, and a beta agentic-codebase format. Some RE reports add ML debugging, probability, statistics, or information theory. | High for official skeleton; moderate for reported station mix. Team instructions control. |
| **Anthropic** | Technical interviews use Colab or CodeSignal; syntax lookup is allowed. Interviews are remote. AI is prohibited in take-homes and live interviews unless expressly permitted. The performance team has a specialized, AI-permitted take-home; current Fellows materials include technical assessment, research discussion, and references. | Current SWE reports describe progressive practical coding, project depth, system design, possible second coding, and values. A Fellows account described research ideation plus black-box investigation and presentation. | High for policy/tools and special public formats; moderate for generic SWE loop; never generalize one team exercise. |
| **Google DeepMind** | 30-minute recruiter call, possible HM call, 2-3 skills interviews, final meetings with team leads/leadership, then decision. Official guide says 4-10 weeks, mostly virtual. AI is for preparation only; no AI in live interviews or interview tasks unless told otherwise. | RE syntheses report executable algorithmic coding, mathematical ML, ML design, and team conversations. RS mixes can instead emphasize research discussion and job talk. | High for process/policy; moderate for exact technical composition. |
| **Meta AI/FAIR/MSL** | Generic Meta flow is recruiter, screen, and virtual loop. Research page confirms screen plus final loop but not station mix. Select roles use an authorized CoderPad assistant. | Reports show combinations of conventional coding, AI-assisted codebase work, ML/research design, project/paper depth, and behavioral interviews. Systems research can look like an SWE loop. | Moderate. Product MLE guidance is more standardized than FAIR/MSL research hiring. |
| **xAI/SpaceXAI** | Technical staff review the CV and exceptional-work statement, a short screen includes technical questions/background, then technical interviews probe depth, complex problem solving, and critical thinking. | Public evidence is too sparse for a stable round recipe. Current model-training roles request a preferred language and a 100-word exceptional-work claim. | High for the sparse official stages; low for question mix. |
| **Microsoft AI** | Microsoft-wide process is virtual and assesses problem solving, design, coding, testing, AI/ML/data science where relevant, resume evidence, and competencies. | Current post-training, applied-science, and alignment roles have materially different work. Multi-report syntheses describe implementation, debugging, ML design, and project depth, but not one MAI-wide loop. | High for Microsoft framework; moderate-to-low for MAI-specific sequence. |
| **Mistral AI** | Intro conversation, then typically 2-5 technical exercises designed to reflect real work, a values conversation, and mandatory references before offer. | Exact technical environment is not public. A successful RE candidate's broader 2025 search includes coding, practical work, system design, quizzes, culture, and references, but those formats cannot all be attributed to Mistral. | High for official sequence; low for exercise contents. |
| **Cohere** | Application review, recruiter, possible take-home, hiring manager, final team conversations, then offer. Video interviews and scheduling flexibility are confirmed. | Current roles span post-training, evaluation, RL environments, safety, and inference, implying very different specialist depth. | High for generic skeleton; low for technical stations. |
| **Scale AI** | No public company-wide map. One current AI Controls and Monitoring RS posting explicitly says interviews assess practical ML prototyping/debugging, research concepts, and culture, without LeetCode-style questions. | Other evaluation, post-training, agent, and ML-systems roles do not publish their loops. | High only for the named role; do not generalize. |
| **NVIDIA research/model teams** | Phone interviews followed by virtual or in-person interviews; several 30-60 minute sessions; coding may use HackerRank, whiteboard, or supplied laptop. An in-person office interview is required before a full-time offer. Unapproved outside tools such as ChatGPT can disqualify a candidate. | Reports add coding, system design, domain fundamentals, project depth, GPU/performance or research specialization. | High for mechanics/policy; moderate for team composition. |

### 6.2 Frontier-lab technical themes visible in current roles

Current job descriptions repeatedly emphasize:

- RL environments, graders, preference data, synthetic data, and post-training;
- model behavior, monitorability, scalable oversight, control, and safeguards;
- distributed RL/pretraining, orchestration, checkpointing, and failure recovery;
- evaluation systems and statistical validity;
- inference efficiency, schedulers, kernels, and accelerators;
- retrieval/search, coding agents, sandboxing, and tool use;
- foundation-model data provenance, filtering, mixture, and deletion;
- multimodal perception, robotics, and real-world deployment.

These are domain signals, not guaranteed interview questions. They should determine the specialist overlay after the recruiter confirms the round formats.

---

## 7. Major technology company overview

| Company | Best-supported current process for ML-adjacent roles | Role-specific emphasis | Confidence/caution |
| --- | --- | --- | --- |
| **Google** | Official process is usually 6-8 weeks, with possible assessment, recruiter calls, structured panel, review, and team/offer stages. Reported MLE loops commonly combine multiple DSA rounds, ML/domain depth, ML design, and behavioral. | Product MLE remains SWE plus ML. Research variants may add a talk or deeper model implementation. Team match can extend the practical timeline. | High generic; moderate MLE composition. |
| **Meta product ML** | Official MLE initial interview is 45 minutes; full loop is up to six 45-minute interviews selected from coding, ML system design, and behavioral. Product DS has a 45-minute analytical/technical screen and reported finals across SQL, analytical execution/reasoning, and behavior. | Fast unaided coding, end-to-end ML design, product metrics, ownership. Select roles add authorized AI-native coding/design. | High for MLE and product DS. |
| **Amazon/AWS** | Official AS/RS process: 1-2 technical phone screens of 60 minutes, then four 55-minute loop interviews and a targeted decision within five business days. | AS includes problem solving/coding, science breadth/depth, real-world formulation, tech talk, and Leadership Principles. RS adds experimental design and data-driven modeling. AWS/platform roles add distributed systems and operations. | Very high for AS/RS; moderate for MLE variants. |
| **Microsoft** | Entirely virtual framework covering problem solving, design, coding, testing, algorithms, data structures, distributed systems, AI/ML, data science, resume and competencies as role-relevant. | Teams vary widely. MLE/AI reports include model components, debugging, ML design, project depth, and standard coding. | High framework; moderate station sequence. |
| **Apple** | No public company-wide loop. Recent multi-report syntheses show recruiter/HM, several technical screens, and team-owned finals. | Vectorized/tensor coding, product-specific ML design, project depth, on-device or platform constraints. Team and job description are especially predictive. | Moderate MLE; low universal sequence. |
| **Netflix** | No official adult sequence. Current org pages separate member ML, ML platform, experimentation/data platform, and data science. | Reports suggest theory/code or SQL/data, culture, ML/platform design, fundamentals, HM/team fit. Culture and senior judgment matter. | Low-to-moderate due small samples. |
| **Uber** | Official role-specific guides provide recruiter and technical screens, then level-dependent loops. Science hiring includes live coding, a no-code analytical jam session, and 4-6 team interviews. | L3-L6+ progressively add new-problem design, prior-system depth, architecture, impact, scope, and leadership. DS emphasizes experiments/causal decisions; AS adds models and engineering. | Very high. |
| **ByteDance/TikTok** | Technical assessment/task plus several interviews is common; official FAQ says the end-to-end process is usually about one month. | Coding, algorithms, design, team technology, ML/domain fundamentals, project/research depth; research candidates may add a job talk. | High generic; moderate MLE/RS details. |
| **LinkedIn** | No current official role-specific map. Recent small-sample reports show recruiter, mixed technical screen, then coding, probability/ML, ML design, and behavioral. | Ranking, platform, and GenAI teams diverge. Some invitation-specific reports include a browser assistant. | Moderate MLE; low DS and AI-policy generalization. |
| **Stripe** | No stable official interview guide. Broad engineering evidence shows practical incremental coding, repository debugging, API work, system design, project/HM. | ML roles layer fraud/risk, evaluation, ML design, or infrastructure onto a production-engineering bar. Both manual and explicitly AI-assisted rounds are reported. | Medium generic; low-to-moderate ML-specific. |
| **Airbnb** | No official sequence. Small current samples report screen, coding, one or two ML design rounds, project/experience, core values, and HM. | Marketplace data, labels, cold start, experiments, end-to-end design and prior-work scope. | Low-to-moderate. |
| **Databricks** | Official FAQ: usually 2-3 months; recruiter, possible HM/assessment, 4-6 final interviews, references, decision. CoderPad and CoderPad Draw are named in engineering guidance. | Coding, algorithms, systems programming, architecture, domain deep dive, cross-functional/HM. Infra roles emphasize concurrency, durability, distributed systems and recovery. | High general/infra; lower ML-specific. |
| **Pinterest** | Current job descriptions emphasize retrieval/ranking, experiments, product metrics, and staff/principal influence; public standardized loop detail is limited. | Product MLE and ranking specialization, with standard coding/design/behavior reported in cross-company accounts. | Strong role evidence, weaker process evidence. |
| **Canva** | Official 2025 engineering post says designated backend, frontend, and ML interviews expect AI use in realistic ambiguous coding tasks. | AI collaboration is assessed as a work mode, but fundamentals and ownership still matter. | High for the published format, not every role. |

---

## 8. Candidate AI-use policy snapshot

| Organization | Publicly supportable rule as of August 27, 2026 |
| --- | --- |
| **OpenAI** | Assessment-specific. Some interviews intentionally allow AI; others assess independent problem solving. Written preparation materials control. |
| **Anthropic** | AI may help preparation and refine an applicant-authored draft. Take-homes and live interviews are no-AI unless the assessment explicitly permits it. |
| **Google DeepMind** | AI may be used for preparation. No AI in live interviews or interview tasks unless told otherwise. |
| **Meta** | Select roles use the assistant built into the authorized CoderPad environment. Outside AI is not authorized. |
| **Google** | Current official guidance prohibits Search, AI, or collective outside assistance during interviews. |
| **Microsoft** | Responsible AI preparation is encouraged. No outside assistance during assessments/interviews unless explicitly permitted. |
| **NVIDIA** | Unapproved outside tools, explicitly including ChatGPT, may result in disqualification. |
| **Uber** | AI may be used for preparation; ChatGPT, automated response generation, and code completion are prohibited during interviews. |
| **Canva** | Designated engineering/ML formats explicitly expect AI use. Follow the exact invitation. |
| **Mistral, Cohere, Scale, xAI, Apple, Netflix, TikTok, LinkedIn, Stripe, Airbnb, Databricks** | No sufficiently clear public company-wide candidate rule was found. Ask. Absence is not permission. |

**Operating rule:** unless the exact round's written instructions authorize a tool, assume no external AI, search, autocomplete, documentation, or private notes.

---

## 9. Representative question archetypes

These are original practice categories derived from public competency evidence, not recovered interview prompts.

### 9.1 General coding

- Process an unbounded event stream with bounded memory.
- Maintain top-k or heavy hitters with updates and deletes.
- Resolve dependencies and explain cycle behavior.
- Parse nested or versioned records with malformed input.
- Build a cache or scheduler, then change capacity or fairness constraints.
- Extend a stateful API across several requirement levels.

**Expected follow-ups:** complexity, memory, concurrency, persistence, test design, alternative data structure, changed input representation.

### 9.2 ML implementation

- Implement logistic regression, k-means, KNN, sampling, or metrics without a high-level estimator.
- Implement attention, a transformer block, normalization, decoding, KV caching, LoRA, or reverse-mode autodiff.
- Vectorize a slow numerical implementation.
- Write a training/evaluation loop with correct mode, gradient, masking, and device behavior.
- Build a mergeable distributed metric accumulator.

**Expected follow-ups:** shapes, stability, masking, gradient flow, time/memory, edge cases, testing against a reference.

### 9.3 ML debugging

- Loss is flat, exploding, or non-finite.
- Offline metric improves but online outcome degrades.
- Single-device reproduction passes while a distributed run fails.
- A resumed run diverges from the original.
- An inference optimization changes output quality.
- A grader can be exploited by the training policy.

**Expected behavior:** construct a timeline, identify the first bad transition, partition causal families, instrument before editing, test one family at a time, preserve scientific validity.

### 9.4 ML breadth and math

- Explain a mechanism and derive its key equation.
- Compare losses or regularizers under explicit assumptions.
- Derive an estimator, gradient, likelihood, or posterior.
- Discuss calibration, uncertainty, sampling, bias/variance, or causal threats.
- Change one assumption and predict what breaks.

**Expected behavior:** define symbols, state assumptions, derive without skipping the central identity, check dimensions/limits, interpret the result.

### 9.5 ML design

- Recommendation/search/ranking with cold start, position bias, multi-objective value, and feedback loops.
- Fraud/abuse/moderation with delayed labels, adaptive adversaries, and asymmetric costs.
- RAG/agent product with permissions, evals, latency, observability, prompt injection, and escalation.
- Forecasting/personalization with leakage-resistant splits and online validation.
- Multimodal/speech system with missing modalities, shift, real-time constraints, and calibration.

**Expected behavior:** objective and user first, then data/labels, baseline, model, offline evidence, serving, monitoring, experiment, fallback and iteration.

### 9.6 Infrastructure and performance

- Design a feature or training-data platform.
- Train a model that does not fit on one accelerator.
- Design fault-tolerant collectives and checkpoint recovery.
- Schedule heterogeneous LLM requests under KV-memory, fairness, and latency limits.
- Use a trace and roofline model to prioritize a kernel optimization.

**Expected behavior:** quantify workload, identify the limiting resource, compare architectures, define state consistency and recovery, measure before and after.

### 9.7 Product data and experimentation

- Define a product metric from raw events and write the query.
- Diagnose sample-ratio mismatch or exposure dilution.
- Choose randomization unit, duration, power, and guardrails.
- Handle network effects, novelty, carryover, interference, and repeated peeking.
- Decide ship/iterate/stop when metrics conflict.

### 9.8 Research

- Defend the strongest claim in prior work and its weakest assumption.
- Critique whether baselines, compute, seeds, and ablations support a paper's claim.
- Generate competing hypotheses for an observed model behavior.
- Design a minimum experiment that separates those hypotheses.
- Interpret hypothetical results and choose the next experiment.
- Explain why the research question matters and where it should not generalize.

### 9.9 Behavioral and level calibration

- A wrong bet or project the candidate stopped.
- Conflict with a senior peer or cross-functional team.
- Quality/safety versus speed.
- Production incident and recovery.
- Ambiguous scope the candidate clarified.
- Mentoring or raising another person's effectiveness.
- Technical direction adopted across teams.

At senior and staff levels, every story should expose a consequential decision, bounded personal ownership, evidence, impact, and a changed belief.

---

## 10. What recent candidate reports add

### 10.1 High-signal cross-account observations

- **Alisa Liu, June 2026:** 11 companies and 57 interviews for RS/MTS roles. ML coding was the most common station; general coding, technical/research discussion, math, job talks, and behavioral also appeared.
- **Silvia Sapora, June 2026:** successful RS search across DeepMind, Isomorphic Labs, Cohere, Meta, and a startup. Reports 3-8 technical interviews, ranging from LeetCode and ML coding to theory, debugging, and modern architectures. Strong survivor and elite-profile bias must be kept in mind.
- **Yong Zheng-Xin, June 2026:** safety/RS search encountered system design, parallel programming, AI-agent evaluation, paid work trials, and many non-safety technical stations. Timing and live headcount materially affected the process.
- **Yuan Meng, February 2026:** senior MLE/RE cycle reports standard coding/ML-design/fundamentals/behavior plus practical OOP, ML-infrastructure design, modern ML coding, presentations, agentic coding, references, and tighter fit-based team matching.
- **Max Mynter, September 2025:** an 18-month path to a Mistral RE role, with roughly 60 touchpoints across 40 companies. Emphasizes LeetCode, CS systems foundations, public artifacts, open-source work, practical assessments, take-homes, culture, and references. His specific assessment list spans multiple unnamed employers and must not be attributed to Mistral.

### 10.2 Anecdotal platform observations

Recent directly readable or partially readable accounts include:

- Meta RS loops with conventional coding plus AI-enabled codebase work and ML design;
- Amazon AS loops reported with five interviews despite the official four-interview description;
- NVIDIA deep-learning hiring retaining DSA before domain depth;
- Anthropic RE progressive implementation/refactoring screens;
- Meta product DS rounds mixing coding, SQL, product analytics, and behavior;
- specialist robotics, CV, and speech loops combining standard coding with domain, systems, and deployment questions;
- substantial take-home and presentation burdens at some data/AI employers.

These establish possibility and recurring shape, not prevalence.

### 10.3 X and social-media evidence

X was useful mainly for discovery, hiring announcements, and links to full first-person essays. Short tweets without method or context were not used to establish process facts. Karan's July 2026 X article provides directional candidate/interviewer advice but no disclosed denominator; Max Mynter's X announcement corroborates the outcome described in his longer post. This is a deliberate evidence choice: virality is not reliability.

---

## 11. Recurring failure modes

### Coding

- Recognizing a familiar pattern and solving the wrong variant.
- Coding before clarifying inputs, output, mutation, ordering, scale, and error behavior.
- Producing pseudocode when runnable code is expected.
- Ignoring empty, duplicate, boundary, overflow, or large-input cases.
- Reaching an acceptable baseline too late to handle follow-ups.
- Being unable to debug calmly after the first failed test.

### ML implementation

- Weak PyTorch/NumPy fluency despite strong conceptual knowledge.
- Shape and broadcasting errors.
- Incorrect causal masks, train/eval mode, gradient handling, or decoding semantics.
- Using a library call that hides the mechanism being assessed.
- No reference implementation, invariant, or numerical check.

### Design

- Starting with architecture instead of user, objective, and failure cost.
- Treating labels as free and immediate.
- Optimizing one offline metric with no product or safety guardrails.
- Naming technologies without workload estimates or decision criteria.
- Omitting deployment, monitoring, rollback, and ownership.
- Failing to manage the 45-minute conversation.

### Research

- Summarizing a paper rather than testing its central claim.
- Suggesting ablations that do not distinguish hypotheses.
- Ignoring matched compute, tuning budget, variance, leakage, or selection effects.
- Defending every choice instead of acknowledging limitations.
- Knowing a subfield deeply but failing broad ML/math questions.

### AI-assisted rounds

- Prompting before reading the task and repository.
- Delegating the plan rather than a bounded subproblem.
- Accepting large diffs without line-by-line review.
- Being unable to explain generated code.
- Re-prompting repeatedly instead of diagnosing the evidence.
- Skipping tests, benchmarks, or security review because the code "looks right."

### Behavioral/project rounds

- A polished opening that does not answer the question.
- Excessive setup and no decision.
- Unbounded "we" and inflated ownership.
- Metrics with no baseline, denominator, attribution, or caveat.
- No failure, conflict, changed belief, or reflection.
- Treating mission/values as a loyalty test rather than a reasoning test.

### Process management

- Starting top-choice companies before calibrating on lower-stakes loops.
- Allowing take-homes and work trials to collide.
- Assuming headcount will remain open.
- Waiting until the end to identify references.
- Treating a passed committee or reference check as a guaranteed offer.
- Failing to coordinate timelines or prepare for team matching.

---

## 12. Comprehensive preparation system

### 12.1 Step 1: define the role contract

For 10-15 target postings, extract:

- primary verbs: research, build, optimize, deploy, evaluate, lead;
- artifacts: papers, models, systems, kernels, evals, experiments, products;
- constraints: scale, latency, safety, customer interaction, hardware, on-call;
- required stack and interview language;
- expected level and scope;
- evidence that the team is hiring for a live mandate.

Choose one primary role family and at most one adjacent family.

### 12.2 Step 2: obtain the actual loop map

Ask the recruiter:

1. What are the exact station names, durations, and order?
2. Is coding DSA, practical/OOP, ML implementation, debugging, or a mixture?
3. Is code executable? Which language, framework, editor, and documentation are allowed?
4. Is any AI tool authorized? Is outside AI forbidden?
5. Is design model-centric, infrastructure-centric, or general distributed systems?
6. Is there a research discussion, brainstorm, take-home, job talk, or project defense?
7. What is the expected presentation format and audience?
8. What does each station score?
9. Is hiring team-specific or followed by team match?
10. Are references required, and at what stage?

### 12.3 Step 3: run a cold diagnostic

Complete, without notes:

- 45-minute general coding problem;
- 45-minute practical or ML implementation problem;
- 30-minute mixed ML/math oral;
- 45-minute ML or infrastructure design;
- 30-minute specialist case;
- 20-minute project/research presentation plus skeptical follow-ups;
- four behavioral prompts;
- SQL case if relevant;
- AI-assisted repository task if confirmed.

Score 0-4 on correctness, framing, depth, evidence/testing, trade-offs, communication, and ownership. External scoring is preferable.

### 12.4 Step 4: allocate by weighted deficit

For module score $s_m$, target score $3$, and role relevance $w_m$:

$$
h_m \propto w_m(3-s_m)_+
$$

A useful weekly allocation is:

- 60-70% on the weakest high-weight stations;
- 20-25% on integrated mocks, project/research defense, and behavior;
- 10-15% on company/team research and process coordination.

Do not allocate time equally across topics.

### 12.5 Step 5: use an attempt-diagnose-repair-retry loop

1. Attempt under the target constraints.
2. Score before reading a solution.
3. Classify the failure: knowledge, representation, implementation, debugging, estimation, communication, or time.
4. Repair one mechanism or subskill.
5. Attempt a changed-surface problem.
6. Retry from a blank page after spacing.
7. Retest later in a mixed session with no category cue.

Reading creates familiarity. Interviews require retrieval and transfer.

### 12.6 Step 6: build one evidence-dense artifact

The artifact should include:

- problem and constraints;
- data provenance or synthetic-data rationale;
- simple baseline;
- reproducible training/analysis/build;
- evaluation protocol and uncertainty;
- error slices and failure examples;
- architecture and operations;
- tests and automation;
- latency/cost/resource measurements;
- decision log and rejected alternatives;
- concise demo and README.

Role-specific additions:

- **RS/AS:** related work, falsifiable hypothesis, ablations, null results;
- **post-training/safety:** grader validation, attack cases, contamination and residual risk;
- **systems/kernel:** profiler traces, numerical checks, reproducible benchmark;
- **DS/ranking:** temporal split, causal assumptions, SRM/power and launch plan;
- **multimodal/robotics:** shift/corruption slices, timing, real-world failure logs;
- **staff/manager:** sanitized strategy or architecture memo showing organizational scope.

### 12.7 Step 7: simulate stations, then the loop

- Begin with individual weak stations.
- Use unseen prompts and different observers.
- Require interruption and changed constraints.
- Review recordings within 24 hours.
- Run a full loop early enough to repair it.
- Practice both unaided and AI-assisted modes when the market requires both, while following the actual employer policy.

---

## 13. Role-specific time allocation

These are starting allocations, not employer rubrics.

| Role | Coding | ML/math/research | Design/systems | Product/eval | Project/behavior |
| --- | ---: | ---: | ---: | ---: | ---: |
| Product MLE | 30% | 15% | 25% | 15% | 15% |
| Applied Scientist | 15% | 25% | 20% | 25% | 15% |
| Research Scientist | 20% | 35% | 10% | 15% | 20% |
| Research Engineer | 30% | 20% | 25% | 10% | 15% |
| Post-training | 25% | 25% | 20% | 20% | 10% |
| ML systems/training | 30% | 15% | 35% | 5% | 15% |
| Inference/kernel | 35% | 15% | 30% | 5% | 15% |
| LLM/agent application | 30% | 10% | 25% | 20% | 15% |
| Safety/evals | 20% | 25% | 15% | 25% | 15% |
| Product Data Scientist | 20% SQL/Python | 25% statistics | 5% | 35% | 15% |
| Ranking/search | 25% | 15% | 25% | 20% | 15% |
| CV/speech/multimodal | 25% | 25% | 20% | 15% | 15% |
| Robotics | 30% Python/C++ | 20% | 30% | 5% | 15% |
| Staff through senior principal | 15% | 15% | 30% | 10% | 30% |
| Manager | 10% | 10% | 20% | 10% | 50% people/execution |

Within "coding," reserve a separate DSA lane whenever the loop includes it. ML coding does not substitute for general coding, and vice versa.

---

## 14. Two-, four-, eight-, and twelve-week plans

### Capacity tiers

| Capacity | Hours/week | Use |
| --- | ---: | --- |
| Maintenance | 5-7 | Preserve readiness; not enough for several foundational gaps. |
| Standard | 8-12 | One or two bounded deficits while employed. |
| Intensive | 13-18 | Broad loop, role-adjacent move, or several weak stations. |
| Transition | 19-25 | Deep role change or full-time preparation. Protect sustainability. |

If the plan below requires more weekly capacity than available, extend the horizon rather than deleting simulations and retries.

### 14.1 Two weeks: consolidation only

**Entry condition:** foundations already exist; every critical station has at least one workable timed baseline.

**Outputs:**

- exact round map and AI policy;
- two coding repetitions in each confirmed coding mode;
- two design/specialist stations;
- six to eight story outlines;
- one project/research defense;
- one full simulation early in week 2;
- two targeted repairs and delayed retries;
- logistics, references, and company packet.

**Do not:** start a large project or attempt broad foundational rebuilding.

### 14.2 Four weeks: bounded repair

- **Week 1 - diagnose:** cold station baselines, role/company map, ranked deficits, story inventory.
- **Week 2 - repair:** deepest two deficits, while maintaining one coding and one presentation/behavior lane.
- **Week 3 - interleave:** random mixed stations, specialist cases, project defense, company-specific transfer.
- **Week 4 - simulate:** full loop in first half, repair two failures, delayed retry, taper.

**Minimum standard outputs:** 8-12 coding attempts across relevant modes, 4 oral/math sets, 3 designs, 3 specialist cases, 2 project/behavior sessions, and 2 full or half-loop simulations.

### 14.3 Eight weeks: breadth plus specialty

- **Weeks 1-2:** shared core and diagnostic repair.
- **Weeks 3-4:** specialist theory and implementation.
- **Weeks 5-6:** evidence-dense artifact or professional-case reconstruction; repeated design and project defense.
- **Week 7:** mixed full-loop simulation and bounded repair.
- **Week 8:** final retries, company packets, team-match questions, logistics, recovery.

This is appropriate for an adjacent-role move only at intensive capacity. At eight hours per week, 64 hours is usually a broad refresh, not a deep transition.

### 14.4 Twelve weeks: deep transition

- **Weeks 1-3:** DSA/software, Python/SQL where required, math/stats, and ML foundations.
- **Weeks 4-6:** target-role implementation and systems.
- **Weeks 7-9:** specialist project, evaluation, and public/sanitized artifact.
- **Weeks 10-11:** station mocks, research/project defense, and 3-4 full simulations.
- **Week 12:** targeted repair, application sequencing, team scorecards, negotiation preparation, and taper.

This is the appropriate default for backend-to-ML, DS-to-RE, MLE-to-kernel, generalist-to-robotics, or other deep family changes.

---

## 15. Readiness gates

A candidate should not average away a fatal weak station.

For each required module, use the median of the last three unseen attempts. A practical readiness condition is:

$$
R = \min_{m \in \text{required modules}} \operatorname{median}(\text{last three unseen scores in }m)
$$

On a 0-4 rubric, target $R \ge 3$ and no critical failure in the last two full-loop simulations.

| Area | Suggested exit evidence |
| --- | --- |
| General coding | Three unseen 40-45 minute tasks with correct runnable code, complexity, tests, and no critical bug. |
| Practical/OOP coding | Two multi-stage tasks with stable APIs/invariants and one concurrency or changed-requirement follow-up. |
| ML implementation | Three unseen primitives and two debugging tasks completed under target tooling. |
| SQL/data | Three timed cases with correct grain, denominator, joins/windows, and validation queries. |
| ML/math breadth | At least 85% on a mixed closed-book set plus mechanism-level explanations and changed-assumption follow-ups. |
| Design | Three unseen cases at 3/4 or better on framing, data, model, evaluation, system, and operations. |
| Research | Two paper defenses, three brainstorm/ablation cases, and one 20-30 minute job talk with Q&A. |
| Specialist depth | Three unseen role-specific cases plus one inspectable artifact. |
| Behavioral | Eight non-duplicative stories covering impact, ambiguity, conflict, failure, incident, trade-off, mentoring, and stopped work. |
| AI-assisted codebase | Two unfamiliar repository tasks with explicit plan, bounded delegation, reviewed diff, tests, and full explanation. |
| Full loop | Two simulations with no failed critical station and no station below 2.5/4. |
| Company readiness | Written AI/tool policy, five relevant team artifacts, exact station map, ten substantive questions, and reference plan. |

A critical failure includes non-running code, invalid experiment logic, leakage, ignoring a primary system constraint, unsafe unbounded tool access, or inability to identify personal contribution.

---

## 16. Application, sequencing, team match, and offer strategy

### Before applications

- Build a role-family scorecard from live postings.
- Tailor the top third of the resume to the target mandate.
- Put the strongest relevant artifact near the top.
- Make every ownership and impact claim independently defensible.
- Contact references before processes begin.
- Use referrals and warm conversations for information, not as a substitute for fit.

### Process sequencing

- Start with credible lower-stakes companies, not companies one would never join.
- Batch processes so top choices enter finals in a similar window.
- Ask about live headcount and team specificity before delaying.
- Avoid overlapping multiple heavy take-homes or work trials.
- Record station notes immediately, without storing proprietary prompts.
- Recompute the prep allocation weekly from observed misses.

### Team-match scorecard

Evaluate:

- exact first-year problem and success criteria;
- research/modeling/engineering/product mix;
- manager quality and decision style;
- technical maturity and honest limitations;
- access to compute, data, hardware, and users;
- ownership boundaries and cross-team dependencies;
- publication/open-source/IP policy;
- on-call, office, travel, and work-trial expectations;
- promotion evidence and scope at the offered level;
- team health, attrition, and roadmap stability.

### Offer comparison

Compare the complete package:

- role, team, manager, level, and scope;
- base, target bonus, equity instrument and amount;
- vesting, cliffs, refreshers, liquidity, and exercise window;
- sign-on, relocation, benefits, start date, and location requirements;
- publication and outside-work constraints;
- downside cases for private-company options.

Resolve role/team/level before optimizing individual compensation components. Use truthful market or competing-offer evidence and never fabricate leverage.

---

# Part II: mlmentorship comparison and gap audit

## 17. Verified inventory

Repository inspection updated on August 29, 2026 found:

- **283 published posts**
   - 85 interview questions;
  - 13 guides;
   - 185 concepts;
- **four ordered role paths:** Applied Scientist, Machine Learning Engineer, Research Scientist, Research Engineer;
- **14 configurable interview rounds,** including `technical-strategy` and `systems-infrastructure`;
- **2-, 4-, and 8-week plans** with 5/8/12-hour workload options;
- **role-aware readiness scoring, spaced retry workflow, story bank, final-week checklist, and simulations;**
- **frontier process cards for OpenAI, Anthropic, DeepMind, Meta, and xAI;**
- **executable labs** for agentic codebase work, broken training, black-box research, ML implementation, post-training environment design, inference scheduling, math oral, technical presentation, values, and accelerator performance;
- **nine deep end-to-end system-design case studies,** exceeding the roadmap target of eight.

### Important correction from an intermediate audit

`systems-infrastructure` is not missing or hidden. It is declared in the round type, defined as an interview round, and used by MLE and RE default overlays. The real gap is broader general distributed-system and practical software coverage, not configurability of the existing ML-infrastructure station.

---

## 18. What mlmentorship already does exceptionally well

### 18.1 Modern ML implementation

The site directly covers the gap repeatedly reported by 2026 RS/RE candidates:

- attention and transformer decoder;
- KV-cache decoding;
- beam search;
- LoRA;
- reverse-mode autodiff;
- top-k retrieval and streaming metrics;
- training-loop and frontier-loss debugging.

This is more current than most generic MLE prep resources.

### 18.2 Frontier work samples

The public labs closely match current format evidence without laundering leaked prompts:

- unfamiliar codebase plus tests and authorized agent;
- black-box model behavior investigation;
- accelerator optimization with traces;
- post-training environments and graders;
- inference scheduler design;
- technical presentation and values defense.

### 18.3 LLM production, training, and systems

Coverage is deep across evals, RAG, hallucinations, inference cost, batching, KV memory, quantization, distributed training, fault recovery, data curation, post-training, safety, and agent evaluation.

### 18.4 Ranking/search/recommendation

The curriculum covers retrieval and ranking architecture, two-tower/cross-encoder choices, metrics, negative sampling, cold start, exploration, bias, feedback loops, and online validation.

### 18.5 Practice mechanics

The readiness check, round-specific rubrics, attempt-diagnose-repair workflow, spaced retries, simulations, story bank, and go/risk/extend gates are unusually operational. The site correctly discourages passive reading and indiscriminate completion of every lab.

### 18.6 Level calibration

Answer rubrics from mid-level through company-dependent senior-principal scope cover bounded execution, autonomous ownership, cross-team influence, killed work, portfolio judgment, delegated authority, succession, and retained hands-on depth.

---

## 19. Coverage matrix

### 19.1 Role families

| Role family | Current coverage | Assessment |
| --- | --- | --- |
| Applied Scientist | **Deep** | Ordered path, design, experimentation, project, behavior, breadth. |
| Product/generalist MLE | **Deep for ML; incomplete for SWE gates** | Strong ML implementation/design/production. General DSA, OOP, concurrency, and backend practical work are externalized but not integrated into readiness. |
| Research Engineer | **Deep for ML systems; incomplete for generic software variants** | Strong implementation, math, research, training/inference. Needs broader practical software/system formats. |
| Research Scientist | **Deep research core; specialist depth varies** | Ordered path, readiness overlay, simulation, research critique, ablation, math oral, black-box lab, implementation, and presentation are present. Current-subfield depth still comes from the candidate's own papers and team scope. |
| Foundation-model/post-training | **Deep technical, moderate pathway** | Excellent concepts and labs; role is a domain overlay rather than a first-class path. |
| ML systems/distributed training | **Deep technical** | Strong training, inference, failure, and lineage content; less ordinary distributed storage, concurrency, API, and multi-region platform design. |
| Inference/performance/kernel | **Deep** | Strong architecture, hardware, scheduler, trace, and accelerator practice. Could add actual CUDA/Triton implementation for specialist candidates. |
| LLM/agent application engineer | **Deep** | Covers evals, RAG, reasoning systems, coding products, agent authority, runtime safety, inference, security, and codebase work. Generic full-stack application work remains external. |
| Safety/alignment/evals | **Deep technically, moderate pathway** | Covers model behavior, control, monitorability, red teams, graders, runtime safety controls, evaluation, and values. A dedicated role path and more quantitative eval statistics remain useful. |
| Product Data Scientist/experimentation | **Moderate for ML statistics; incomplete for the full loop** | Experiment, uncertainty, thresholds, and causal reasoning are strong. There is still no role path, SQL/data station, or product-analytics simulation. |
| Recommendations/search/ranking | **Deep** | Covers retrieval, ranking objectives, negative sampling, position bias, counterfactual evaluation, multi-task learning, cold start, and product design. |
| CV/multimodal/speech | **Moderate-to-deep multimodal, moderate speech** | A deep real-time multimodal case joins solid foundations and speech questions. An executable modality lab and specialist loop remain absent. |
| Robotics/embodied AI | **Light** | One strong policy-learning concept but no controls, state estimation, hardware-debugging, sim-to-real, or C++/real-time practice path. |
| Staff/principal/senior-principal IC | **Deep** | Calibration, story evidence, simulations, three annotated mocks, and cases across platforms, agents, reasoning, ranking, data, multimodal systems, coding products, and safety test transfer across domains. |
| ML/research manager | **Minimal** | Hiring-manager audience content exists, but not candidate preparation for people leadership, portfolio, hiring, underperformance, and organization design. |

### 19.2 Interview stations

| Station | Current coverage | Gap status |
| --- | --- | --- |
| Application/proof of work | **Moderate** | Strong exceptional-work/reference guide, but little resume/application funnel tooling. |
| Recruiter/HM screen | **Minimal** | No dedicated career narrative, why-role, level, or recruiter-screen practice. |
| General DSA | **None by design** | Dedicated external content is reasonable; omitting it from readiness and plans is not. |
| Progressive OOP/practical software | **Minimal** | Major modern MLE/RE gap. |
| ML implementation/debugging | **Deep** | Core strength. |
| SQL/data manipulation | **None by design** | Critical for product DS; useful for some MLE/AS loops. |
| Probability/math/stats | **Deep for ML math and core applied statistics** | Moments, distributions, bootstrap inference, hypothesis tests, and causal identification are covered. Power, sequential testing, and variance reduction remain thinner. |
| ML breadth | **Deep** | Core strength. |
| ML model/system design | **Deep case coverage** | Nine long-form cases cover ranking, ML platforms, agents, reasoning, multimodal systems, foundation-model data, coding products, ecosystem health, and runtime safety. |
| ML infrastructure | **Deep for frontier training/inference; moderate-to-deep for ML platforms** | Data lineage and point-in-time contracts are covered. Ordinary queueing, multi-region state, and general systems remain thinner. |
| Product/experimentation | **Deep conceptually** | Causal inference, thresholds, delayed labels, selective labels, and feedback loops are explicit. No SQL-backed product case or Product DS loop exists. |
| Research paper discussion | **Moderate** | Critique and ablation exist; no worked paper-defense packet. |
| Research brainstorm | **Moderate** | Black-box investigation is strong; no clean from-spec ideation/experiment-design lab. |
| Project/job talk | **Deep** | Core strength. |
| Behavioral/values | **Deep** | Strong story bank and values packet; lacks observed exemplar mocks. |
| AI-assisted codebase | **Deep** | Core strength. |
| Take-home/work-trial defense | **Light** | No dedicated time-box, decision log, authorship defense, or live-extension guide. |
| References | **Moderate** | Strong claim/artifact/reference coherence guidance. |
| Team match/offer/negotiation | **Minimal** | High-value post-loop gap. |

---

## 20. Prioritized mlmentorship gaps

### P0: false-readiness and policy risks

#### P0.1 Integrate general coding into readiness without becoming a LeetCode site

**Evidence:** DSA remains in official and reported Google, Meta, Amazon, TikTok, Microsoft, NVIDIA, MLE, AS, RE, and RS processes. The site explicitly sends users elsewhere, but its readiness check can still declare an MLE or RE ready without any general-coding evidence.

**Recommended change:**

- add a `general-coding` round and rating lane;
- link to a small vetted external roadmap rather than writing hundreds of problems;
- require a timed baseline and exit gate when the recruiter confirms DSA;
- distinguish DSA, practical/OOP, ML implementation, and AI-assisted codebase work.

#### P0.2 Add progressive practical software/OOP/concurrency coverage

**Evidence:** Current OpenAI/Anthropic method reports and multi-company senior-MLE accounts repeatedly identify progressive APIs, stateful systems, concurrency, code reading, and practical multi-part coding. Existing ML primitives do not train this skill.

**Recommended change:**

- one progressive stateful-service lab;
- one concurrency/debugging lab;
- one unfamiliar non-ML codebase exercise;
- practical software as a selectable MLE/RE station.

#### P0.3 Create a first-class Research Scientist path (completed August 28, 2026)

**Evidence:** Named 2026 RS accounts show a distinct mixture of ML coding, broad technical discussion, research defense, math, brainstorms, job talks, and behavior.

**Implemented:**

- RS role overlay, readiness weights, ordered path, and simulation;
- existing paper critique, black-box investigation, math, implementation, presentation, and behavioral material assembled into one sequence;
- explicit warning that publications do not replace interview performance.

#### P0.4 Refresh candidate AI policies immediately (completed August 28, 2026)

Three current source mismatches can affect integrity:

1. **OpenAI:** current official guide positively confirms assessment-specific AI rules; the site frames this mainly as absence of one universal rule plus a reported pilot.
2. **DeepMind:** the official PDF now explicitly says no AI in live interviews or tasks unless told otherwise; the site says the public overview does not state a universal policy.
3. **Meta:** the official careers page says **select roles** include an authorized assistant; site language such as "many interviews" risks overstating prevalence.

The source registry, public cards, verification date, and caution text now match the current official wording.

### P1: major scope and conversion gaps

#### P1.1 Add a Product Data Scientist/experimentation path or draw a firm boundary (partially completed August 28, 2026)

The site now covers causal inference, bootstrap uncertainty, distribution choice, and cost-aware decisions. A full Product Data Scientist path still needs SQL, product analytics cases, and a DS simulation. If this remains out of scope, the readiness tool should state that clearly and route to vetted resources.

#### P1.2 Add a 12-week/deep-transition plan

The readiness form accepts "12+ weeks" but always maps recommendations to 2, 4, or 8 weeks. The eight-week plan assumes 64 standard hours and calls itself suitable for role transitions. That is adequate for refresh or adjacent moves, but not many deep transitions.

Add:

- a 12-week plan;
- a transition-capacity tier;
- explicit entry conditions for each horizon;
- a warning that 64 hours is not a foundations rebuild;
- station-level readiness gates based on multiple unseen attempts.

#### P1.3 Expand the process registry

The current five organizations are valuable but incomplete. Add official, carefully caveated cards for:

- Microsoft AI;
- Mistral AI;
- Cohere;
- Scale AI;
- NVIDIA research/model teams.

A separate major-tech page should cover the best official role-specific sources: Meta MLE/DS, Amazon AS/RS, Uber ML/Sciences, Microsoft, Google, TikTok, Databricks, and NVIDIA.

#### P1.4 Add take-home and work-trial defense

Cover:

- ethical time-boxing and disclosure;
- requirement and decision log;
- tests and reproducibility;
- artifact walkthrough;
- omitted work and trade-offs;
- likely reviewer challenges;
- changed-requirement extension;
- AI-use disclosure;
- when a work trial is unreasonable.

#### P1.5 Add team-match, process management, and offer material

The technical curriculum ends before a consequential part of the funnel. Add application batching, live-headcount questions, reference timing, team scorecard, offer components, private-equity risk, negotiation preparation, and truthful timeline coordination.

#### P1.6 Complete the deep system-design backlog (completed August 29, 2026)

Nine long-form cases now exceed the original target of eight. The final release added a reasoning model under fixed compute, a real-time multimodal assistant, short-form video ecosystem ranking, a foundation-model data platform, an AI coding product, and an agent safety control plane.

#### P1.7 Add general ML-platform and distributed-system depth (partially completed August 28, 2026)

ML data lineage, versioning, point-in-time correctness, replay, deletion, and rollback are now covered. Frontier distributed training is also strong. Remaining practical platform cases include:

- event and feature pipelines;
- batch/stream consistency;
- versioned datasets and models;
- queueing and backpressure;
- cache/state semantics;
- orchestration and idempotency;
- online/offline data quality;
- multi-region serving and failure recovery.

### P2: specialist and quality expansion

#### P2.1 Specialist executable packets

Add one reviewed packet each for:

- CV/multimodal;
- speech;
- robotics/embodied AI;
- quantitative safety/evals.

The current roadmap correctly asks for expert review before presenting these as authoritative.

#### P2.2 Upper-IC and manager paths (upper-IC completed August 29, 2026)

Staff through senior-principal preparation now includes cross-organization architecture, portfolio trade-offs, technical strategy, delegated authority, reversibility, succession, a level path, simulations, and story evidence. Manager preparation still needs hiring, coaching, underperformance, organizational design, and 30/60/90-day planning.

#### P2.3 Observed mock exemplars (partially completed August 29, 2026)

One synthetic upper-IC architecture mock is complete. Remaining consented or synthetic annotated transcripts should show:

- a weak-to-strong repair;
- a project answer losing ownership under follow-up;
- an AI-assisted diff review;
- a research brainstorm with competing hypotheses.

#### P2.4 Application and recruiter-screen practice

Add a role narrative, resume-to-round traceability check, "why this team" worksheet, and recruiter questions on process, level, headcount, tools, and timing.

---

## 21. Content and source corrections beyond missing coverage

### 21.1 Role taxonomy needs a frontier-era revision

The guide's fixed research/modeling/engineering/product percentages are useful intuition but can imply more precision than exists. Current roles often collapse these boundaries. Replace or supplement the percentages with observable role-contract axes:

- research agenda ownership;
- model/data experimentation;
- production/software ownership;
- systems/performance ownership;
- product/customer ownership;
- safety/evaluation authority;
- publication/open-source expectation.

The Research Scientist description should also acknowledge productized frontier research, eval infrastructure, and researcher-written production code.

### 21.2 Broad compensation claims need sourcing or removal

The claim that frontier labs pay roughly twice FAANG compensation for equivalent roles is too broad without a dated, level-, location-, and equity-adjusted source. Private-company equity is not directly comparable to public RSUs. This belongs in a sourced compensation methodology, not a universal multiplier.

### 21.3 Preserve the site's evidence discipline

The strongest existing editorial choice is separating official, method-based, and first-person evidence. Keep that distinction, add access dates, and never promote one role-specific posting or one candidate's experience into company policy.

---

## 22. Recommended implementation sequence

### Release 1: prevent false readiness

1. Refresh OpenAI, DeepMind, and Meta AI-policy text.
2. Add general-coding and practical-software stations to readiness and simulations.
3. Add the Research Scientist role overlay and simulation.
4. Add a 12-week transition plan.

### Release 2: complete the funnel

5. Add take-home/work-trial defense.
6. Add recruiter/application and team-match/offer pages.
7. Expand frontier registry to Microsoft AI, Mistral, Cohere, Scale, and NVIDIA.
8. Add a top-tech official-process guide.

### Release 3: deepen transfer

9. Publish the seven remaining deep system-design cases with observer packets.
10. Add Product DS/SQL or an explicit external route.
11. Add practical OOP/concurrency and general ML-platform labs.
12. Add specialist-reviewed multimodal, speech, robotics, and safety packets.

### Release 4: improve feedback quality

13. Publish further observed mock exemplars only with informed consent; three synthetic upper-IC transfer mocks are complete.
14. Add the manager pathway; the upper-IC overlay through senior principal is complete.
15. Measure starts, completions, retries, and station-level perceived usefulness without collecting candidate answers.

---

## 23. Selected source ledger

All links were accessed or rechecked on August 27, 2026 unless otherwise noted.

### Official frontier sources

- [OpenAI interview guide](https://openai.com/interview-guide/)
- [Anthropic careers](https://www.anthropic.com/careers)
- [Anthropic candidate AI guidance](https://www.anthropic.com/candidate-ai-guidance), updated July 10, 2025
- [Anthropic: Designing AI-resistant technical evaluations](https://www.anthropic.com/engineering/AI-resistant-technical-evaluations), January 21, 2026
- [Anthropic original performance take-home](https://github.com/anthropics/original_performance_takehome)
- [Google DeepMind careers](https://deepmind.google/careers/)
- [Google DeepMind official interview PDF](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf)
- [Meta hiring process and AI FAQ](https://www.metacareers.com/hiring-process/)
- [Meta MLE initial interview](https://www.metacareers.com/ML-prep-initial/)
- [Meta MLE full loop](https://www.metacareers.com/ML-prep-onsite/)
- [Meta Data Scientist initial interview](https://www.metacareers.com/DS-prep-initial/)
- [xAI/SpaceXAI careers](https://x.ai/careers)
- [Mistral AI careers](https://mistral.ai/careers)
- [Cohere careers](https://cohere.com/careers)
- [NVIDIA hiring process](https://www.nvidia.com/en-us/about-nvidia/careers/how-we-hire/)

### Official major-tech sources

- [Google hiring process](https://www.google.com/about/careers/applications/how-we-hire/)
- [Google interview guidance](https://www.google.com/about/careers/applications/interview-tips/)
- [Amazon Applied Scientist prep](https://www.amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep)
- [Amazon Research Scientist prep](https://www.amazon.jobs/content/en/how-we-hire/research-scientist-interview-prep)
- [Microsoft hiring and Candidate Code of Conduct](https://careers.microsoft.com/v2/global/en/hiring-tips.html)
- [Microsoft technical interviewing](https://careers.microsoft.com/v2/global/en/hiring-tips/technical-interviewing.html)
- [Uber ML and AI Engineering interview guide](https://jobs.uber.com/en/uber-interview-guide/ml-ai-engineering-interview-guide/)
- [Uber Sciences interview guide](https://jobs.uber.com/en/uber-interview-guide/sciences-interview-guide/)
- [TikTok FAQ](https://lifeattiktok.com/faq/?language=en)
- [TikTok technical interview tips](https://lifeattiktok.com/campus/interview-tips)
- [Databricks interview preparation](https://www.databricks.com/company/careers/interview-prep)
- [Canva: AI in interviews](https://www.canva.dev/blog/engineering/yes-you-can-use-ai-in-our-interviews/), June 11, 2025

### Method-based reports

- [interviewing.io: how AI is changing interviews](https://interviewing.io/blog/how-is-ai-changing-interview-processes-not-much-and-a-whole-lot), September 17, 2025, updated October 8, 2025; 67 first-hand survey responses
- [interviewing.io OpenAI process](https://interviewing.io/openai-interview-questions), updated June 22, 2026
- [interviewing.io Anthropic process](https://interviewing.io/anthropic-interview-questions), updated June 16, 2026
- [interviewing.io: becoming an MLE at FAANG](https://interviewing.io/blog/becoming-an-mle-at-faang-what-you-need-to-know), July 16, 2025
- [Google DeepMind RE synthesis](https://igotanoffer.com/en/advice/google-deepmind-research-engineer-interview), updated June 15, 2026
- [NVIDIA process synthesis](https://igotanoffer.com/en/advice/nvidia-interview-process), updated August 24, 2026

### Recent multi-report company syntheses

These sources use self-selected candidate submissions and are less authoritative than official guides. Their sample sizes and dates make them useful for role variants, not company-wide guarantees.

- [Google MLE process](https://trueinterview.io/blog/google-machine-learning-engineer-interview-process), July 24, 2026; 10 reported experiences
- [Amazon MLE process](https://trueinterview.io/blog/amazon-machine-learning-engineer-interview-process), July 24, 2026; 12 reported experiences
- [Amazon Research Scientist process](https://trueinterview.io/blog/amazon-research-scientist-interview-process), July 24, 2026; 12 reported experiences
- [Microsoft MLE process](https://trueinterview.io/blog/microsoft-machine-learning-engineer-interview-process), July 24, 2026; 8 reported experiences
- [Microsoft AI Engineer process](https://trueinterview.io/blog/microsoft-ai-engineer-interview-process), July 24, 2026; 8 reported experiences
- [Apple MLE process](https://trueinterview.io/blog/apple-machine-learning-engineer-interview-process), July 24, 2026; 16 reported experiences
- [NVIDIA MLE process](https://trueinterview.io/blog/nvidia-machine-learning-engineer-interview-process), July 24, 2026; 8 reported experiences
- [Netflix MLE process](https://trueinterview.io/blog/netflix-machine-learning-engineer-interview-process), July 24, 2026; 4 reported experiences
- [Uber MLE process](https://trueinterview.io/blog/uber-machine-learning-engineer-interview-process), July 24, 2026; 6 reported experiences
- [ByteDance MLE process](https://trueinterview.io/blog/bytedance-machine-learning-engineer-interview-process), July 24, 2026; 13 reported experiences
- [LinkedIn MLE process](https://trueinterview.io/blog/linkedin-machine-learning-engineer-interview-process), July 24, 2026; 8 reported experiences
- [Stripe general engineering process](https://trueinterview.io/blog/stripe-interview-process), July 24, 2026; 51 candidate reports and 46 question records
- [Airbnb MLE process](https://trueinterview.io/blog/airbnb-machine-learning-engineer-interview-process), July 31, 2026; 4 reported experiences
- [Databricks infrastructure process](https://trueinterview.io/blog/databricks-infrastructure-engineer-interview-process), July 24, 2026; 49 reported experiences

### Named first-person accounts

- [Alisa Liu: Notes on the Industry Job Search](https://alisawuffles.github.io/blog/job-search/), June 20, 2026
- [Silvia Sapora: ML Job Interviews](https://silviasapora.github.io/blog/ml-interviews.html), June 2026
- [Yong Zheng-Xin: Surprising lessons from my research scientist job search](https://yongzx.github.io/blog/2026/06/24/job-search/), June 24, 2026
- [Yuan Meng: MLE Interview 2.0](https://www.yuan-meng.com/posts/mle_interviews_2.0/), February 1, 2026
- [Max Mynter: Becoming a Research Engineer at a Big LLM Lab](https://maxmynter.substack.com/p/becoming-a-research-engineer-at-a), September 25, 2025
- [Andrey Goncharov: Anthropic Fellows interview account](https://blog.faillearnrepeat.net/blog/i-failed-my-anthropic-interview-and-came-to-tell-you-all-about-it-so-you-dont-have-to), February 12, 2025, updated September 2025
- [Yong and Joseph: AI safety technical interviews](https://www.lesswrong.com/posts/dvsFfGuXXyHYkyifp/tips-for-cracking-the-ai-safety-technical-interview-1), June 16, 2026

### Community and anecdotal corroboration

- [Meta Research Scientist account](https://www.reddit.com/r/leetcode/comments/1k4imgn/meta_research_scientist_interview_experience/), April 21, 2025
- [Meta ML Research Scientist with AI round](https://www.reddit.com/r/leetcode/comments/1r37w7q/meta_ml_research_scientist_interview_experience/), February 12, 2026
- [Multi-company ML/GenAI compilation](https://www.reddit.com/r/developersIndia/comments/1q065gd/my_ml_engineer_interviews_compilation_along_with/), December 31, 2025
- [Amazon Applied Scientist loop discussion](https://www.reddit.com/r/MachineLearning/comments/1pfqphi/d_amazon_applied_scientist_1_interview_loop/), December 6, 2025
- [NVIDIA Senior Deep Learning Engineer account](https://www.reddit.com/r/CUDA/comments/1oof56x/my_interview_process_with_nvidia_for_senior_deep/), November 4, 2025
- [Hacker News discussion of AI and take-homes](https://news.ycombinator.com/item?id=42909166), February 2, 2025
- [CV/ML specialist preparation discussion](https://www.reddit.com/r/computervision/comments/1hc1d8m/advice_on_preparing_for_cvml_interviews_at_major/), December 11, 2024
- [Glassdoor Amazon Applied Scientist report](https://www.glassdoor.com/Interview/Amazon-Interview-E6036-RVW105161157.htm), August 12, 2026; anonymous single report
- [Glassdoor Meta Data Scientist report](https://www.glassdoor.com/Interview/Meta-Interview-E40772-RVW105080175.htm), August 6, 2026; anonymous single report
- [Glassdoor Meta MLE report](https://www.glassdoor.com/Interview/Meta-Interview-E40772-RVW104642584.htm), July 4, 2026; anonymous single report
- [Glassdoor Anthropic Research Engineer report](https://www.glassdoor.com/Interview/Anthropic-Interview-E8109027-RVW103598339.htm), April 16, 2026; anonymous single report
- [Blind Microsoft Applied Scientist 2 account](https://www.teamblind.com/post/insane-interview-with-microsoft-applied-scientist-2-6uop2f24), November 9, 2025; anonymous single report
- [Blind OpenAI/Anthropic reference-check account](https://www.teamblind.com/post/openaianthropic-offer-chances-reference-check-go224g3c), July 15, 2025; anonymous counterexample to treating references as an offer
- [Blind Meta Robotics Studio loop](https://www.teamblind.com/post/meta-robotics-studio-final-loop-interview-help-mtt4gzy1), July 15, 2026; incomplete single process
- [Karan's X article on AI/ML interview strategy](https://x.com/kmeanskaran/article/2078728719448015134), July 19, 2026; directional advice with no disclosed denominator

### Representative current role-scope sources

These establish what teams currently hire to do. They do not prove a dedicated interview station.

- [OpenAI Frontier Evals and Environments Research Engineer](https://openai.com/careers/research-engineer-frontier-evals-and-environments-san-francisco/)
- [OpenAI RL Training Infrastructure Software Engineer](https://openai.com/careers/software-engineer-rl-training-infra-san-francisco/)
- [OpenAI Model Inference Software Engineer](https://openai.com/careers/software-engineer-model-inference-san-francisco/)
- [OpenAI Agent Post-Training Research](https://openai.com/careers/agent-post-training-research-san-francisco/)
- [Anthropic Production Model Post-Training Research Engineer](https://job-boards.greenhouse.io/anthropic/jobs/5112018008)
- [Anthropic Pretraining Scaling Research Engineer](https://job-boards.greenhouse.io/anthropic/jobs/4938432008)
- [Anthropic Model Evaluations Research Engineer](https://job-boards.greenhouse.io/anthropic/jobs/5198255008)
- [Scale AI Controls and Monitoring Research Scientist](https://scale.com/careers/4675694005)
- [Physical Intelligence Applied Researcher](https://jobs.ashbyhq.com/physicalintelligence/1a7a181f-c318-4e0b-9516-c7111b3e3968)
- [ElevenLabs Research Engineer](https://jobs.ashbyhq.com/elevenlabs/3d650946-5ac2-4729-9ae4-129c43fcd0b5)

### Learning-method evidence

- [Retrieval-practice review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12292765/), July 17, 2025
- [Systematic review of distributed and retrieval practice](https://pmc.ncbi.nlm.nih.gov/articles/PMC11078833/), 2024

---

## Bottom line

A candidate maximizing success in 2026 should prepare for a **portfolio of assessment modes**, not a bag of ML questions:

- unaided general coding;
- practical and ML-specific implementation;
- deep mechanisms and derivations;
- end-to-end ML and infrastructure design;
- product/experiment judgment;
- research and project defense;
- behavioral and values evidence;
- AI-assisted repository work when explicitly authorized;
- references, team match, and offer execution.

mlmentorship already covers the frontier ML center of that portfolio unusually well. Its next highest-return work is to close the remaining edges that cause false confidence: general and practical software gates, a Product DS pathway, longer transition plans, broader process coverage, and deeper realistic cases.