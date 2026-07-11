export interface FrontierLabFile {
  label: string;
  href: string;
  note: string;
}

export interface FrontierLab {
  slug: string;
  title: string;
  description: string;
  eyebrow: string;
  duration: string;
  format: string;
  bottomLine: string;
  protocol: string[];
  deliverables: string[];
  gates: string[];
  files: FrontierLabFile[];
  questionHref: string;
  questionLabel: string;
  externalSource?: { label: string; href: string; note: string };
}

export const FRONTIER_LABS: FrontierLab[] = [
  {
    slug: 'agentic-codebase',
    title: 'Agentic ML codebase lab',
    description: 'Extend an unfamiliar ML evaluation package with an AI coding agent while preserving correctness, scope, and ownership.',
    eyebrow: 'AI-native engineering work sample',
    duration: '60 minutes',
    format: 'Existing multi-file Python package, failing tests, one bug, one feature',
    bottomLine: 'The model can write code. Your signal is whether you can map the system, delegate bounded work, reject bad changes, and prove the result.',
    protocol: [
      'Run the tests and write a five-line codebase map before opening the agent.',
      'State the merge invariant and slice-report contract in your own words.',
      'Delegate one bounded behavior at a time. Include the file, constraint, and verification command.',
      'Inspect every diff. Revert unrelated edits and explain one rejected suggestion.',
      'Add one edge test the agent did not propose, then run focused and full verification.',
      'End with a two-minute summary of the change, residual risk, and what you would inspect next.',
    ],
    deliverables: [
      'A repaired merge operation and implemented slice report.',
      'One candidate-written regression test.',
      'A short prompt log showing bounded delegation.',
      'A final diff you can explain line by line.',
    ],
    gates: [
      'Merged shard metrics equal single-pass metrics.',
      'Low-support slices cannot trip the guardrail.',
      'The public API remains stable.',
      'No generated change survives without review and a verification path.',
    ],
    files: [
      { label: 'Assignment README', href: '/labs/agentic-eval-service/README.md', note: 'Contract, rules, timing, and scoring.' },
      { label: 'Metrics implementation', href: '/labs/agentic-eval-service/ml_eval/metrics.py', note: 'Contains the distributed merge bug.' },
      { label: 'Report implementation', href: '/labs/agentic-eval-service/ml_eval/report.py', note: 'Contains the bounded feature task.' },
      { label: 'Public tests', href: '/labs/agentic-eval-service/tests/test_metrics.py', note: 'Expected to fail before repair.' },
    ],
    questionHref: '/questions/agentic-ml-codebase-interview/',
    questionLabel: 'Read the interview rubric',
  },
  {
    slug: 'broken-training',
    title: 'Broken frontier LLM training lab',
    description: 'Diagnose interacting data, masking, loss, accumulation, scheduler, and evaluation defects in a tiny causal LM.',
    eyebrow: 'LLM training debugging work sample',
    duration: '75 minutes',
    format: 'PyTorch package, failing contract tests, source review, one-batch overfit',
    bottomLine: 'A training run that executes is not a training run that is correct. Repair semantics before tuning optimization.',
    protocol: [
      'Run contract tests and classify failures as data, model, update, or evaluation defects.',
      'Trace one token through input, target, mask, logits, loss, and gradient.',
      'Repair label and causal-mask semantics before changing the optimizer loop.',
      'Make accumulation, clipping, and scheduler cadence agree on what an update means.',
      'Restore evaluation isolation and caller mode.',
      'Overfit one tiny batch, then record the first remaining multi-GPU risk.',
    ],
    deliverables: [
      'Repaired code and one new regression test.',
      'A defect table with symptom, mechanism, and verification.',
      'A one-batch overfit result.',
      'A prioritized plan for a distributed recurrence.',
    ],
    gates: [
      'Targets are next tokens and padded targets are ignored.',
      'Prefix outputs cannot depend on future tokens.',
      'Cross-entropy receives raw logits.',
      'One accumulation window creates one optimizer and scheduler step.',
    ],
    files: [
      { label: 'Assignment README', href: '/labs/broken-llm-training/README.md', note: 'Diagnostic order and deliverables.' },
      { label: 'Data contract', href: '/labs/broken-llm-training/tiny_lm/data.py', note: 'Contains label defects.' },
      { label: 'Tiny causal LM', href: '/labs/broken-llm-training/tiny_lm/model.py', note: 'Contains attention and logits defects.' },
      { label: 'Training protocol', href: '/labs/broken-llm-training/train.py', note: 'Contains update and evaluation defects.' },
      { label: 'Public tests', href: '/labs/broken-llm-training/tests/test_contracts.py', note: 'Covers only part of the failure surface.' },
    ],
    questionHref: '/questions/debug-frontier-llm-training-run/',
    questionLabel: 'Read the diagnostic framework',
  },
  {
    slug: 'black-box',
    title: 'Black-box model behavior lab',
    description: 'Turn model observations into falsifiable hypotheses, discriminating probes, boundary conditions, and a short research readout.',
    eyebrow: 'Research work sample',
    duration: '90 minutes',
    format: 'Query-only behavior API, experiment template, five-minute readout',
    bottomLine: 'The goal is not prompt volume. The goal is the smallest experiment that separates your leading explanations.',
    protocol: [
      'Run one control and record the exact observation without interpretation.',
      'Write three competing hypotheses that predict different results.',
      'Change one factor at a time before testing interactions.',
      'Use controls or repeats where variance could produce a false pattern.',
      'Find one boundary where the leading claim stops holding.',
      'Present the claim, evidence, uncertainty, and next discriminating experiment in five minutes.',
    ],
    deliverables: [
      'A probe table with predictions and observations.',
      'One interaction effect and one negative result.',
      'A bounded behavioral claim, not an architecture guess.',
      'A five-minute research readout.',
    ],
    gates: [
      'Every claim has a probe that could have falsified it.',
      'Observation and interpretation remain separate.',
      'The readout exposes uncertainty and boundary conditions.',
      'The next experiment is chosen by expected information gain.',
    ],
    files: [
      { label: 'Assignment README', href: '/labs/black-box-behavior/README.md', note: 'Rules and readout contract.' },
      { label: 'Query client', href: '/labs/black-box-behavior/query.py', note: 'Use this interface without opening the oracle.' },
      { label: 'Probe template', href: '/labs/black-box-behavior/probe_template.py', note: 'A minimal experiment ledger.' },
    ],
    questionHref: '/questions/investigate-black-box-model-behavior/',
    questionLabel: 'Read the research rubric',
  },
  {
    slug: 'implementation',
    title: 'Frontier ML implementation set',
    description: 'Implement a decoder block, incremental KV cache, beam search, LoRA adapter, and reverse-mode autodiff against executable contracts.',
    eyebrow: 'ML primitives in code',
    duration: 'Five sessions, 35 to 50 minutes each',
    format: 'Starter modules plus public unit tests',
    bottomLine: 'These exercises test whether modern ML abstractions are mechanisms you can build and debug, not names you can recite.',
    protocol: [
      'Choose one primitive and restate its tensor or graph contract before coding.',
      'Implement the smallest correct baseline without importing the target abstraction.',
      'Run the focused test, localize the first failure, and add two edge tests.',
      'State time, memory, and numerical behavior.',
      'Handle one changed constraint without replacing the design.',
      'Explain how the production implementation differs from the toy version.',
    ],
    deliverables: [
      'A passing implementation for each chosen primitive.',
      'Two candidate-written tests per primitive.',
      'A complexity and numerical-stability note.',
      'One production follow-up per primitive.',
    ],
    gates: [
      'Tests cover shape, edge behavior, and gradient or probability semantics.',
      'Causal and incremental paths agree with full computation.',
      'Frozen and trainable parameters are explicit.',
      'You can explain every operation without framework magic.',
    ],
    files: [
      { label: 'Set README', href: '/labs/ml-implementation/README.md', note: 'Exercise list and graduation rules.' },
      { label: 'Decoder starter', href: '/labs/ml-implementation/decoder.py', note: 'Causal attention and decoder block.' },
      { label: 'KV-cache starter', href: '/labs/ml-implementation/kv_cache.py', note: 'Incremental decode equivalence.' },
      { label: 'Beam-search starter', href: '/labs/ml-implementation/beam_search.py', note: 'Bounded hypotheses and EOS.' },
      { label: 'LoRA starter', href: '/labs/ml-implementation/lora.py', note: 'Frozen base plus low-rank update.' },
      { label: 'Autograd starter', href: '/labs/ml-implementation/autograd.py', note: 'Scalar computation graph and reverse pass.' },
    ],
    questionHref: '/questions/implement-transformer-decoder/',
    questionLabel: 'Start with the decoder rubric',
  },
  {
    slug: 'inference-scheduler',
    title: 'LLM inference scheduler lab',
    description: 'Implement KV admission, chunked prefill, continuous batching, tenant fairness, and overload behavior for a model server.',
    eyebrow: 'Inference systems work sample',
    duration: '75 minutes plus 30-minute design follow-up',
    format: 'Scheduler starter, tests, and an arrival trace',
    bottomLine: 'The hard part is not naming vLLM. It is turning latency, throughput, fairness, and KV capacity into one explicit scheduling policy.',
    protocol: [
      'Derive worst-case KV reservation per request before writing scheduler code.',
      'Implement admission and release invariants.',
      'Keep decode moving while chunking long prefills.',
      'Rotate across tenants without violating per-tenant FIFO.',
      'Run the supplied tests, then replay the arrival trace conceptually.',
      'Defend overload shedding, SLO metrics, and one alternative scheduler.',
    ],
    deliverables: [
      'A passing scheduler implementation.',
      'A policy note for time to first token and inter-token latency.',
      'An overload and tenant-isolation design.',
      'A bottleneck dashboard with four discriminating metrics.',
    ],
    gates: [
      'Reserved blocks never exceed capacity.',
      'Long prefill cannot starve active decode.',
      'Finished requests release memory immediately.',
      'The policy has an explicit rejection or degradation point.',
    ],
    files: [
      { label: 'Assignment README', href: '/labs/inference-scheduler/README.md', note: 'System contract and follow-ups.' },
      { label: 'Scheduler starter', href: '/labs/inference-scheduler/scheduler.py', note: 'Admission and iteration scheduling.' },
      { label: 'Public tests', href: '/labs/inference-scheduler/tests/test_scheduler.py', note: 'Capacity, fairness, and chunking.' },
      { label: 'Arrival trace', href: '/labs/inference-scheduler/arrival_trace.json', note: 'Mixed prompts, tenants, and generations.' },
    ],
    questionHref: '/questions/design-production-llm-inference-service/',
    questionLabel: 'Read the full system-design answer',
  },
  {
    slug: 'accelerator',
    title: 'Accelerator performance lab',
    description: 'Use a trace, bottleneck model, and correctness-preserving experiments to optimize Anthropic’s released simulated-accelerator challenge.',
    eyebrow: 'Performance engineering work sample',
    duration: 'Two hours for a timed attempt, then open-ended',
    format: 'Public Anthropic challenge plus an original experiment worksheet',
    bottomLine: 'Performance work starts with a verified bottleneck and ends with a measured speedup. Clever code without a trace or correctness gate is noise.',
    protocol: [
      'Clone the official public challenge and run its submission tests unchanged.',
      'Record baseline cycles and map execution units, dependencies, and scratchpad limits.',
      'Use the trace to choose one bottleneck hypothesis.',
      'Predict the effect before changing code.',
      'Measure, verify, and either keep or revert the change.',
      'Review agent changes for weakened tests or machine constraints.',
    ],
    deliverables: [
      'A verified cycle count and unchanged official tests.',
      'An experiment ledger with predictions and measurements.',
      'One failed optimization and what it taught you.',
      'A short explanation of simulator-to-hardware limits.',
    ],
    gates: [
      'The official verification command passes.',
      'Every kept change has a measured effect.',
      'Scratchpad and dependency constraints remain valid.',
      'The candidate can distinguish instruction count, memory traffic, and critical-path depth.',
    ],
    files: [
      { label: 'Experiment worksheet', href: '/labs/accelerator-performance/worksheet.md', note: 'Bottleneck, prediction, measurement, and debrief.' },
    ],
    questionHref: '/questions/optimize-accelerator-workload/',
    questionLabel: 'Read the performance rubric',
    externalSource: {
      label: 'Anthropic original performance take-home',
      href: 'https://github.com/anthropics/original_performance_takehome',
      note: 'Official public challenge. Follow its current verification instructions.',
    },
  },
  {
    slug: 'math-oral',
    title: 'Timed ML math oral',
    description: 'Derive, sanity-check, and interpret eight ML results under oral follow-up rather than silently reproducing memorized algebra.',
    eyebrow: 'Math and statistics screen',
    duration: 'Eight separate drills, 8 to 15 minutes each',
    format: 'Prompt deck, observer follow-up, scored oral derivation',
    bottomLine: 'A correct last line is not enough. The signal is assumptions, key identities, sanity checks, and what the result says about model behavior.',
    protocol: [
      'Draw one prompt without previewing its checks.',
      'Define symbols, dimensions, and assumptions aloud.',
      'Derive without notes while the observer interrupts one skipped step.',
      'Check dimensions and one boundary or special case.',
      'Interpret which term changes model behavior and why.',
      'Answer one follow-up that changes an assumption.',
    ],
    deliverables: [
      'A recorded or observed derivation.',
      'A 0 to 2 score for six dimensions.',
      'One corrected derivation written from memory.',
      'A spaced retry on another day.',
    ],
    gates: [
      'No zero in setup or derivation.',
      'At least 9 of 12 scorecard points.',
      'A changed-assumption follow-up is handled cleanly.',
      'A second spaced success is recorded before graduation.',
    ],
    files: [
      { label: 'Prompt deck', href: '/labs/math-oral/prompts.json', note: 'Eight derivations with observer checks.' },
      { label: 'Scorecard', href: '/labs/math-oral/scorecard.md', note: 'Six dimensions and a passing gate.' },
    ],
    questionHref: '/questions/derive-ml-math-under-pressure/',
    questionLabel: 'Read the oral-screen strategy',
  },
  {
    slug: 'post-training-environment',
    title: 'Post-training environment and grader lab',
    description: 'Design an RL environment and repair a grader that rewards successful but unsafe tool use.',
    eyebrow: 'Post-training research work sample',
    duration: '60 minutes plus 20-minute research discussion',
    format: 'Broken structured grader, adversarial episodes, environment-design readout',
    bottomLine: 'A scalar reward is an attack surface. Separate task success, policy compliance, and process evidence before training optimizes the wrong behavior.',
    protocol: [
      'State the capability, prohibited behavior, state, action space, and terminal condition.',
      'Run tests and identify how outcome quality hides process violations.',
      'Implement hard disqualification gates and bounded component scores.',
      'Add adversarial episodes for reward hacking and grader blind spots.',
      'Separate deterministic checks, model-graded judgments, and human review.',
      'Explain grader versioning, drift, and online validation.',
    ],
    deliverables: [
      'A repaired structured grader.',
      'Two new adversarial episodes.',
      'An environment and coverage specification.',
      'A decision rule for whether the training intervention helped.',
    ],
    gates: [
      'Unsafe process cannot be offset by final-answer quality.',
      'Evidence accompanies every gate and penalty.',
      'The environment has explicit reset, timeout, and terminal semantics.',
      'Grader and environment versions are part of every result.',
    ],
    files: [
      { label: 'Assignment README', href: '/labs/post-training-environment/README.md', note: 'Environment and grader contract.' },
      { label: 'Broken grader', href: '/labs/post-training-environment/grader.py', note: 'Scalar reward hides unsafe process.' },
      { label: 'Public tests', href: '/labs/post-training-environment/tests/test_grader.py', note: 'Policy gates and component behavior.' },
    ],
    questionHref: '/questions/design-post-training-data-and-rl-environment/',
    questionLabel: 'Read the post-training design rubric',
  },
];

export function getFrontierLab(slug: string): FrontierLab | undefined {
  return FRONTIER_LABS.find((lab) => lab.slug === slug);
}
