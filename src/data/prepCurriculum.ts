export type RoleId = 'applied-scientist' | 'ml-engineer' | 'research-scientist' | 'research-engineer';
export const ROLE_ORDER: RoleId[] = ['applied-scientist', 'ml-engineer', 'research-scientist', 'research-engineer'];
export type LevelId = 'l4' | 'l5' | 'l6' | 'l7' | 'l8';
export type AreaId =
  | 'fundamentals'
  | 'production'
  | 'llm-systems'
  | 'recsys-search'
  | 'system-design'
  | 'experimentation'
  | 'behavioral'
  | 'math-research'
  | 'coding';
export type RoundId =
  | 'ml-breadth'
  | 'coding'
  | 'agentic-codebase'
  | 'ml-system-design'
  | 'project-deep-dive'
  | 'technical-strategy'
  | 'technical-presentation'
  | 'behavioral'
  | 'values-mission'
  | 'product-experimentation'
  | 'research-depth'
  | 'research-work-sample'
  | 'math-oral'
  | 'systems-infrastructure';
export type PracticeModeId =
  | 'breadth'
  | 'coding'
  | 'agentic'
  | 'system-design'
  | 'project'
  | 'presentation'
  | 'behavioral'
  | 'values'
  | 'experimentation'
  | 'research'
  | 'infrastructure'
  | 'math';

export interface PracticeRubricItem {
  id: string;
  label: string;
  question: string;
  coaching: string;
}

export interface PracticeModeDefinition {
  id: PracticeModeId;
  label: string;
  minutes: number;
  instruction: string;
  rubric: PracticeRubricItem[];
}

export const KNOWLEDGE_AREAS: Array<{
  id: AreaId;
  label: string;
  link: string;
  description: string;
  starterLinks: Array<{ label: string; href: string }>;
}> = [
  { id: 'fundamentals', label: 'ML fundamentals', link: '/questions/#ml-fundamentals', description: 'Explain mechanisms, assumptions, and failure modes, not just formulas.', starterLinks: [{ label: 'Bias–variance', href: '/questions/bias-variance-tradeoff/' }, { label: 'Choose a loss', href: '/questions/how-to-choose-loss-function/' }] },
  { id: 'production', label: 'Production & debugging', link: '/questions/#deep-learning-production', description: 'Diagnose data, training, serving, monitoring, and operational failures.', starterLinks: [{ label: 'Model not learning', href: '/questions/debug-model-not-learning/' }, { label: 'ML monitoring', href: '/questions/design-ml-monitoring/' }] },
  { id: 'llm-systems', label: 'LLM systems', link: '/questions/#llm-systems', description: 'Reason about evals, retrieval, serving cost, reliability, and safety.', starterLinks: [{ label: 'Evaluate an LLM app', href: '/questions/how-would-you-evaluate-an-llm-application/' }, { label: 'Fine-tune vs RAG', href: '/questions/fine-tune-vs-prompt-vs-rag/' }] },
  { id: 'recsys-search', label: 'Retrieval, search & recommendations', link: '/questions/#recsys-and-search', description: 'Connect retrieval and ranking choices to metrics and feedback loops.', starterLinks: [{ label: 'YouTube recommender', href: '/questions/design-youtube-recommender/' }, { label: 'Evaluate a ranker', href: '/questions/evaluate-search-ranker/' }] },
  { id: 'system-design', label: 'ML system design', link: '/questions/#ml-system-design', description: 'Scope users and constraints before designing data, models, serving, and iteration.', starterLinks: [{ label: 'Fraud detection', href: '/questions/design-fraud-detection/' }, { label: 'Fixed-budget system', href: '/questions/design-ml-system-fixed-budget/' }] },
  { id: 'experimentation', label: 'Product & experimentation', link: '/questions/#product-and-experimentation', description: 'Choose success metrics, guardrails, experiment units, and decision rules.', starterLinks: [{ label: 'Design an A/B test', href: '/questions/design-ml-ab-test/' }, { label: 'Offline–online gap', href: '/questions/debug-offline-online-metric-gap/' }] },
  { id: 'behavioral', label: 'Behavioral & leadership', link: '/questions/#behavioral', description: 'Use specific decisions and evidence that hold up under skeptical follow-up.', starterLinks: [{ label: 'Ambitious project', href: '/questions/most-ambitious-project/' }, { label: 'Disagreement', href: '/questions/disagreed-with-senior/' }] },
  { id: 'math-research', label: 'Math & research depth', link: '/questions/#math-and-research', description: 'Derive, critique evidence, and design experiments from first principles.', starterLinks: [{ label: 'Design an ablation', href: '/questions/design-ablation-study/' }, { label: 'Critique a paper', href: '/questions/critique-ml-paper/' }] },
  { id: 'coding', label: 'ML implementation', link: '/questions/#ml-implementation', description: 'Implement ML primitives and evaluation systems with correct, testable, resource-aware code.', starterLinks: [{ label: 'Debug training loop', href: '/questions/debug-training-loop/' }, { label: 'Batched top-k retrieval', href: '/questions/implement-batched-top-k/' }] },
];

export const INTERVIEW_ROUNDS: Array<{
  id: RoundId;
  label: string;
  description: string;
  minutes: number;
  areaWeights: Partial<Record<AreaId, number>>;
  starterLinks: Array<{ label: string; href: string }>;
}> = [
  {
    id: 'ml-breadth',
    label: 'ML breadth',
    description: 'Mechanisms, model choices, metrics, and “why” follow-ups.',
    minutes: 45,
    areaWeights: { fundamentals: .75, 'math-research': .25 },
    starterLinks: [
      { label: 'Bias–variance trade-off', href: '/questions/bias-variance-tradeoff/' },
      { label: 'Choose a loss function', href: '/questions/how-to-choose-loss-function/' },
    ],
  },
  {
    id: 'coding',
    label: 'ML implementation / coding',
    description: 'ML primitives, evaluation systems, debugging, and resource-aware implementation under time pressure.',
    minutes: 45,
    areaWeights: { coding: .8, fundamentals: .1, production: .1 },
    starterLinks: [
      { label: 'Debug a training loop', href: '/questions/debug-training-loop/' },
      { label: 'Implement attention', href: '/questions/implement-attention-from-scratch/' },
      { label: 'Memory-bounded top-k retrieval', href: '/questions/implement-batched-top-k/' },
      { label: 'Mergeable model metrics', href: '/questions/implement-streaming-classification-metrics/' },
    ],
  },
  {
    id: 'agentic-codebase',
    label: 'Agentic codebase',
    description: 'Navigate an unfamiliar ML codebase, direct an AI coding agent, review its work, and ship a tested change.',
    minutes: 60,
    areaWeights: { coding: .55, production: .3, 'system-design': .15 },
    starterLinks: [
      { label: 'Extend an ML evaluation codebase', href: '/questions/agentic-ml-codebase-interview/' },
      { label: 'Debug a frontier training run', href: '/questions/debug-frontier-llm-training-run/' },
    ],
  },
  {
    id: 'ml-system-design',
    label: 'ML system design',
    description: 'Requirements, data, modeling, evaluation, serving, monitoring, and iteration.',
    minutes: 45,
    areaWeights: { 'system-design': .65, production: .2, experimentation: .15 },
    starterLinks: [
      { label: 'Design a multi-team ML platform', href: '/questions/design-multi-team-ml-platform/' },
      { label: 'Train and serve a reasoning model', href: '/questions/design-reasoning-model-fixed-budget/' },
      { label: 'Design a real-time multimodal assistant', href: '/questions/design-real-time-multimodal-assistant/' },
      { label: 'Design short-form video recommendation', href: '/questions/design-short-form-video-ecosystem/' },
      { label: 'Design a foundation-model data platform', href: '/questions/design-foundation-model-data-platform/' },
      { label: 'Design an AI coding product', href: '/questions/design-ai-coding-product/' },
      { label: 'Design an agent safety control plane', href: '/questions/design-agent-safety-control-plane/' },
      { label: 'Design fraud detection', href: '/questions/design-fraud-detection/' },
      { label: 'Design ML monitoring', href: '/questions/design-ml-monitoring/' },
    ],
  },
  {
    id: 'project-deep-dive',
    label: 'Project deep-dive',
    description: 'Ownership, technical decisions, failures, evidence, influence, and reflection.',
    minutes: 45,
    areaWeights: { behavioral: .55, production: .2, 'system-design': .15, experimentation: .1 },
    starterLinks: [
      { label: 'Most ambitious project', href: '/questions/most-ambitious-project/' },
      { label: 'Scope an ambiguous problem', href: '/questions/scope-ambiguous-problem/' },
    ],
  },
  {
    id: 'technical-strategy',
    label: 'Technical strategy',
    description: 'Shared boundaries, investment order, delegated authority, decision checkpoints, reversibility, and retained technical depth.',
    minutes: 60,
    areaWeights: { 'system-design': .4, behavioral: .3, production: .2, experimentation: .1 },
    starterLinks: [
      { label: 'Design an enterprise agent platform', href: '/questions/design-enterprise-agent-platform/' },
      { label: 'Study an annotated upper-IC mock', href: '/guides/annotated-upper-ic-agent-platform-mock/' },
      { label: 'Reasoning-model strategy case', href: '/questions/design-reasoning-model-fixed-budget/' },
      { label: 'Annotated reasoning-strategy mock', href: '/guides/annotated-reasoning-strategy-mock/' },
      { label: 'Ecosystem-ranking strategy case', href: '/questions/design-short-form-video-ecosystem/' },
      { label: 'Annotated ecosystem-strategy mock', href: '/guides/annotated-ecosystem-strategy-mock/' },
      { label: 'Calibrate senior through senior-principal scope', href: '/guides/l5-vs-l6-faang-ml/' },
    ],
  },
  {
    id: 'technical-presentation',
    label: 'Technical project presentation',
    description: 'Present one consequential ML project, then defend its decisions, evidence, failures, and impact under interruption.',
    minutes: 45,
    areaWeights: { behavioral: .45, production: .2, 'system-design': .2, experimentation: .15 },
    starterLinks: [
      { label: 'Present a technical ML project', href: '/questions/present-technical-ml-project/' },
      { label: 'Most ambitious project', href: '/questions/most-ambitious-project/' },
    ],
  },
  {
    id: 'behavioral',
    label: 'Behavioral / leadership',
    description: 'Conflict, prioritization, failure, influence, mentoring, and collaboration.',
    minutes: 45,
    areaWeights: { behavioral: .9, experimentation: .1 },
    starterLinks: [
      { label: 'Disagreement with a senior person', href: '/questions/disagreed-with-senior/' },
      { label: 'Decide what to work on', href: '/questions/decide-what-to-work-on/' },
    ],
  },
  {
    id: 'values-mission',
    label: 'Values & mission',
    description: 'Reason through a real ethical tension, explain a changed belief, and show principled disagreement without slogans.',
    minutes: 45,
    areaWeights: { behavioral: .7, experimentation: .15, 'system-design': .15 },
    starterLinks: [
      { label: 'Defend values under pressure', href: '/questions/defend-values-under-ethical-pressure/' },
      { label: 'Quality or safety over speed', href: '/questions/advocated-quality-over-speed/' },
    ],
  },
  {
    id: 'product-experimentation',
    label: 'Product & experimentation',
    description: 'Metrics, experiment design, causal threats, guardrails, and ship decisions.',
    minutes: 45,
    areaWeights: { experimentation: .7, 'system-design': .15, fundamentals: .15 },
    starterLinks: [
      { label: 'A/B test an ML product', href: '/questions/design-ml-ab-test/' },
      { label: 'Debug offline–online disagreement', href: '/questions/debug-offline-online-metric-gap/' },
    ],
  },
  {
    id: 'research-depth',
    label: 'Research depth / paper critique',
    description: 'Hypotheses, ablations, evidence quality, derivations, and research judgment.',
    minutes: 45,
    areaWeights: { 'math-research': .65, fundamentals: .25, experimentation: .1 },
    starterLinks: [
      { label: 'Design an ablation study', href: '/questions/design-ablation-study/' },
      { label: 'Critique an ML paper', href: '/questions/critique-ml-paper/' },
    ],
  },
  {
    id: 'research-work-sample',
    label: 'Research work sample',
    description: 'Turn a black-box observation into hypotheses, discriminating probes, measured evidence, and a short readout.',
    minutes: 90,
    areaWeights: { 'math-research': .5, experimentation: .25, coding: .15, fundamentals: .1 },
    starterLinks: [
      { label: 'Investigate black-box model behavior', href: '/questions/investigate-black-box-model-behavior/' },
      { label: 'Critique an ML paper', href: '/questions/critique-ml-paper/' },
    ],
  },
  {
    id: 'math-oral',
    label: 'Math & statistics oral',
    description: 'Rapid derivations, assumptions, sanity checks, interpretation, and changed-assumption follow-ups.',
    minutes: 45,
    areaWeights: { 'math-research': .8, fundamentals: .2 },
    starterLinks: [
      { label: 'Derive ML math under pressure', href: '/questions/derive-ml-math-under-pressure/' },
      { label: 'Derive logistic regression', href: '/questions/derive-logistic-regression/' },
    ],
  },
  {
    id: 'systems-infrastructure',
    label: 'Systems / infrastructure',
    description: 'Training and serving scale, reliability, bottlenecks, and cost trade-offs.',
    minutes: 45,
    areaWeights: { production: .55, coding: .25, 'system-design': .2 },
    starterLinks: [
      { label: 'Design a multi-team ML platform', href: '/questions/design-multi-team-ml-platform/' },
      { label: 'Train a 100B-parameter model', href: '/questions/train-100b-model/' },
      { label: 'Reduce LLM inference cost 10×', href: '/questions/reduce-llm-inference-cost-10x/' },
    ],
  },
];

export const ROLE_OVERLAYS: Record<RoleId, {
  label: string;
  summary: string;
  defaultRounds: RoundId[];
  allocation: Array<{ label: string; percent: number }>;
  priorityLinks: Array<{ label: string; href: string }>;
  warning: string;
}> = {
  'applied-scientist': {
    label: 'Applied Scientist',
    summary: 'Balance scientific depth with scoping, product judgment, and evidence that you can ship.',
    defaultRounds: ['ml-breadth', 'ml-system-design', 'project-deep-dive', 'behavioral', 'product-experimentation'],
    allocation: [
      { label: 'ML breadth & research depth', percent: 25 },
      { label: 'ML design & experimentation', percent: 30 },
      { label: 'Project deep-dive', percent: 20 },
      { label: 'Behavioral & leadership', percent: 15 },
      { label: 'Coding / implementation', percent: 10 },
    ],
    priorityLinks: [
      { label: 'Five things AS interviews test', href: '/guides/five-things-as-interview-tests/' },
      { label: 'Evaluate an LLM application', href: '/questions/how-would-you-evaluate-an-llm-application/' },
      { label: 'Most ambitious project', href: '/questions/most-ambitious-project/' },
    ],
    warning: 'Do not prep like a pure researcher: shipping evidence, product metrics, and engineering judgment often decide the loop.',
  },
  'ml-engineer': {
    label: 'Machine Learning Engineer',
    summary: 'Prioritize coding and dependable systems while retaining enough ML depth to make sound modeling decisions.',
    defaultRounds: ['coding', 'ml-system-design', 'systems-infrastructure', 'ml-breadth', 'behavioral'],
    allocation: [
      { label: 'ML implementation / coding', percent: 30 },
      { label: 'ML system design', percent: 25 },
      { label: 'Production & infrastructure', percent: 20 },
      { label: 'ML breadth', percent: 15 },
      { label: 'Behavioral & project evidence', percent: 10 },
    ],
    priorityLinks: [
      { label: 'Debug a training loop', href: '/questions/debug-training-loop/' },
      { label: 'Design ML monitoring', href: '/questions/design-ml-monitoring/' },
      { label: 'Design a feature store', href: '/questions/design-feature-store/' },
    ],
    warning: 'Do not substitute ML reading for coding repetitions. Many MLE loops reject strong modelers on software execution.',
  },
  'research-engineer': {
    label: 'Research Engineer',
    summary: 'Combine strong implementation and systems reasoning with enough research depth to critique and operationalize ideas.',
    defaultRounds: ['coding', 'systems-infrastructure', 'research-depth', 'ml-breadth', 'project-deep-dive'],
    allocation: [
      { label: 'ML implementation / coding', percent: 30 },
      { label: 'Training & inference systems', percent: 25 },
      { label: 'Research depth', percent: 20 },
      { label: 'ML breadth', percent: 15 },
      { label: 'Project & behavioral evidence', percent: 10 },
    ],
    priorityLinks: [
      { label: 'Implement attention', href: '/questions/implement-attention-from-scratch/' },
      { label: 'Train a 100B model', href: '/questions/train-100b-model/' },
      { label: 'Design an ablation study', href: '/questions/design-ablation-study/' },
    ],
    warning: 'Expect title variation. Confirm whether the loop behaves like research, ML systems, or general software engineering.',
  },
  'research-scientist': {
    label: 'Research Scientist',
    summary: 'Prioritize original hypotheses, mathematical depth, research judgment, and defense of a small number of important contributions.',
    defaultRounds: ['coding', 'math-oral', 'research-depth', 'research-work-sample', 'technical-presentation', 'behavioral'],
    allocation: [
      { label: 'Research depth & paper defense', percent: 30 },
      { label: 'Math & ML breadth', percent: 25 },
      { label: 'Research coding & experiments', percent: 20 },
      { label: 'Job talk & project defense', percent: 15 },
      { label: 'Behavioral & collaboration', percent: 10 },
    ],
    priorityLinks: [
      { label: 'Critique a paper', href: '/questions/critique-ml-paper/' },
      { label: 'Design an ablation study', href: '/questions/design-ablation-study/' },
      { label: 'Investigate model behavior', href: '/questions/investigate-black-box-model-behavior/' },
    ],
    warning: 'Publications open the door, but the loop tests whether you can derive, implement, challenge evidence, generate alternatives, and defend your own decisions.',
  },
};

export const LEVEL_OVERLAYS: Record<LevelId, {
  label: string;
  bar: string;
  evidence: string[];
  commonFailure: string;
}> = {
  l4: {
    label: 'L4 / mid-level',
    bar: 'Execute a bounded problem correctly and explain the technical choices.',
    evidence: ['Correct implementation', 'Sound fundamentals', 'Clear debugging procedure', 'Awareness of basic trade-offs'],
    commonFailure: 'Trying to manufacture strategy instead of demonstrating reliable execution.',
  },
  l5: {
    label: 'L5 / senior',
    bar: 'Own an ambiguous project area and make autonomous, defensible decisions.',
    evidence: ['A clear personal decision thread', 'End-to-end shipping evidence', 'Trade-offs and failed attempts', 'Online or operational outcomes'],
    commonFailure: 'Describing team activity without identifying the decisions you personally owned.',
  },
  l6: {
    label: 'L6 / staff',
    bar: 'Choose problems and strategy, influence multiple teams, and remain technically credible.',
    evidence: ['Cross-team adoption or influence', 'A wrong strategic bet and recovery', 'A project you killed or redirected', 'Hands-on depth beneath the strategy'],
    commonFailure: 'Speaking only at the strategic level without concrete technical decisions and failure evidence.',
  },
  l7: {
    label: 'L7 / principal',
    bar: 'Set durable direction across organizations, balance a multi-year technical portfolio, and preserve the ability to change course.',
    evidence: ['Multi-organization technical direction', 'Portfolio choices with explicit opportunity cost', 'Reversible standards and decision checkpoints', 'Other technical leaders carrying the work'],
    commonFailure: 'Presenting a broad architecture without migration cost, stop conditions, retained technical depth, or evidence that the direction outlived personal involvement.',
  },
  l8: {
    label: 'Senior principal / distinguished',
    bar: 'Create coherent technical direction across several principal-owned portfolios while preserving delegated authority, external adaptability, succession, and reversal.',
    evidence: ['A durable doctrine above changing implementations', 'Principal-level leaders carrying distinct technical domains', 'Portfolio changes driven by external or multi-year evidence', 'Standards that survive leadership, vendor, and regulatory change'],
    commonFailure: 'Using company-wide scope as a substitute for decision rights, technical mechanisms, independent principal leadership, or evidence that can reverse the direction.',
  },
};

const sharedFraming: PracticeRubricItem = {
  id: 'framing',
  label: 'Problem framing',
  question: 'Did you clarify the objective, user, constraints, and failure cost before solving?',
  coaching: 'On the next attempt, spend the opening minutes only on objective, user, success metric, data, latency, cost, and failure asymmetry. Restate the scoped problem before proposing a solution.',
};

const sharedCommunication: PracticeRubricItem = {
  id: 'communication',
  label: 'Communication',
  question: 'Was the answer structured, concise, and responsive to new information?',
  coaching: 'Lead with a short outline, signpost transitions, and end with the decision. Ask one clarifying question before defending a challenged assumption.',
};

export const PRACTICE_MODES: Record<PracticeModeId, PracticeModeDefinition> = {
  breadth: {
    id: 'breadth', label: 'ML breadth', minutes: 8,
    instruction: 'Explain the mechanism, why it works, when it fails, and one alternative.',
    rubric: [
      { id: 'mechanism', label: 'Mechanism', question: 'Did you explain why it works rather than recite a recipe?', coaching: 'Write the causal or mathematical mechanism in three sentences, then explain it without jargon.' },
      { id: 'assumptions', label: 'Assumptions', question: 'Did you name the assumptions and where they break?', coaching: 'Add one data, optimization, or statistical assumption and a concrete counterexample.' },
      { id: 'tradeoffs', label: 'Trade-offs', question: 'Did you compare at least one credible alternative?', coaching: 'State the decision criterion and what evidence would make you choose the alternative.' },
      { id: 'followup', label: 'Follow-up depth', question: 'Could you answer one level deeper without notes?', coaching: 'Generate the most likely “why?” follow-up and prepare a mechanism-level answer.' },
      sharedCommunication,
    ],
  },
  coding: {
    id: 'coding', label: 'ML implementation', minutes: 40,
    instruction: 'Clarify the contract, implement a correct baseline, test edge cases, then optimize.',
    rubric: [
      { id: 'contract', label: 'Contract & examples', question: 'Did you clarify inputs, outputs, constraints, and a small example?', coaching: 'Before coding, write one normal example, one edge case, and the expected output.' },
      { id: 'correctness', label: 'Correctness', question: 'Is the implementation complete and correct on edge cases?', coaching: 'Trace the smallest failing input by hand and add a targeted test before changing the code.' },
      { id: 'complexity', label: 'Complexity', question: 'Did you state time, memory, and important bottlenecks?', coaching: 'Name the dominant operation and derive complexity from the loops and stored data.' },
      { id: 'testing', label: 'Testing & debugging', question: 'Did you test systematically and localize failures?', coaching: 'Use normal, empty, duplicate, boundary, and large-input cases; explain the next diagnostic before editing.' },
      sharedCommunication,
    ],
  },
  agentic: {
    id: 'agentic', label: 'Agentic ML implementation', minutes: 60,
    instruction: 'Map the codebase, state the plan, delegate bounded changes, review every diff, and prove the result with tests and measurements.',
    rubric: [
      { id: 'map', label: 'Codebase map', question: 'Did you identify the execution path, invariants, tests, and likely change surface before editing?', coaching: 'Write a five-line codebase map: entry point, data flow, core invariant, test command, and files likely to change.' },
      { id: 'delegation', label: 'Delegation', question: 'Did prompts give the agent bounded context, constraints, and a verifiable outcome?', coaching: 'Delegate one function or failing behavior at a time and state what must remain unchanged.' },
      { id: 'review', label: 'Critical review', question: 'Could you explain and challenge every generated change rather than accepting it wholesale?', coaching: 'Inspect the diff line by line and reject one unnecessary, unsafe, or overly broad change before continuing.' },
      { id: 'verification', label: 'Verification', question: 'Did you run focused tests, add an edge case, and measure the relevant behavior?', coaching: 'Run the narrowest failing test first, then the full suite, then one benchmark or invariant check.' },
      sharedCommunication,
    ],
  },
  'system-design': {
    id: 'system-design', label: 'ML system design', minutes: 40,
    instruction: 'Scope first; then cover data, baseline, model, evaluation, serving, monitoring, and iteration.',
    rubric: [
      sharedFraming,
      { id: 'architecture', label: 'End-to-end architecture', question: 'Did data, training, serving, feedback, and ownership form one coherent system?', coaching: 'Draw the data path from user action to labels, training, serving, logging, and retraining.' },
      { id: 'evaluation', label: 'Evaluation & decision', question: 'Did you connect offline metrics, online outcomes, guardrails, and launch criteria?', coaching: 'Name one primary metric, two guardrails, key slices, and the condition that blocks launch.' },
      { id: 'operations', label: 'Operations & failure', question: 'Did you cover latency, cost, monitoring, rollback, and delayed labels?', coaching: 'Choose the two most expensive failures and specify detection, ownership, and fallback behavior.' },
      sharedCommunication,
    ],
  },
  project: {
    id: 'project', label: 'Project deep-dive', minutes: 20,
    instruction: 'Give a two-minute overview, then pressure-test decisions, failures, evidence, and ownership.',
    rubric: [
      { id: 'scope', label: 'Scope & stakes', question: 'Was the problem’s scale and importance clear for the target level?', coaching: 'State who was affected, the time horizon, the cost of failure, and why existing approaches were insufficient.' },
      { id: 'ownership', label: 'Personal ownership', question: 'Could you distinguish your decisions from the team’s work?', coaching: 'Replace vague “we” clauses with bounded “I owned…” and explicitly credit what others owned.' },
      { id: 'decision', label: 'Decision quality', question: 'Did you explain alternatives, evidence, and why you chose this path?', coaching: 'Pick the most consequential decision and reconstruct the options and evidence available at that time.' },
      { id: 'failure', label: 'Failure & recovery', question: 'Did you discuss a specific failure and how your model changed?', coaching: 'Name the wrong assumption, the signal that disproved it, and the action you drove next.' },
      { id: 'impact', label: 'Impact & reflection', question: 'Was the outcome measurable, attributable, and honestly bounded?', coaching: 'Separate output from outcome, quantify direction or scale, and state what you would change now.' },
    ],
  },
  presentation: {
    id: 'presentation', label: 'Technical presentation', minutes: 45,
    instruction: 'Deliver a 30-minute decision narrative, then defend assumptions, alternatives, failures, ownership, and impact for 15 minutes.',
    rubric: [
      { id: 'thesis', label: 'Opening thesis', question: 'Did the first two minutes establish the problem, your claim, your role, and why the outcome mattered?', coaching: 'Rewrite the opening as four sentences: problem, stakes, your ownership, and result.' },
      { id: 'decisions', label: 'Decision spine', question: 'Did the presentation center on consequential decisions rather than chronology?', coaching: 'Choose three decisions and put rejected alternatives and evidence beside each one.' },
      { id: 'technical-depth', label: 'Technical depth', question: 'Could you descend from architecture to one implementation detail without losing the audience?', coaching: 'Prepare one architecture view, one bottleneck calculation, and one failure trace.' },
      { id: 'defense', label: 'Defense under questions', question: 'Did you answer challenges directly, update when warranted, and preserve a clear position?', coaching: 'Practice a two-sentence direct answer before adding context or caveats.' },
      { id: 'impact', label: 'Impact & attribution', question: 'Were outcomes measured and your contribution separated from team output?', coaching: 'Label each outcome as observed, estimated, or influenced, then state exactly what you owned.' },
    ],
  },
  behavioral: {
    id: 'behavioral', label: 'Behavioral / leadership', minutes: 3,
    instruction: 'Answer the exact question in 90–120 seconds, then reserve detail for follow-ups.',
    rubric: [
      { id: 'answer', label: 'Question fit', question: 'Did the story directly answer the prompt?', coaching: 'Repeat the question in your opening sentence and explain why this story is the relevant example.' },
      { id: 'ownership', label: 'Ownership', question: 'Were your decisions and actions distinct from the team’s?', coaching: 'Use precise “I” verbs for your contribution and name what collaborators owned.' },
      { id: 'stakes', label: 'Stakes & conflict', question: 'Was there a real decision, tension, or consequence?', coaching: 'Cut setup that does not establish the decision, disagreement, risk, or cost.' },
      { id: 'evidence', label: 'Evidence & result', question: 'Did the outcome include concrete evidence without exaggeration?', coaching: 'Use defensible metrics, scale, or observed behavior and distinguish correlation from your contribution.' },
      { id: 'reflection', label: 'Reflection', question: 'Did you show how the experience changed your judgment?', coaching: 'State one principle you now use and one thing you would do differently.' },
    ],
  },
  values: {
    id: 'values', label: 'Values & mission', minutes: 12,
    instruction: 'Take a position on a real tension, ground it in experience, expose uncertainty, and respond honestly to principled pushback.',
    rubric: [
      { id: 'tension', label: 'Real tension', question: 'Did the answer identify competing values and a real cost on both sides?', coaching: 'Name what a reasonable person on the other side is protecting and what your choice sacrifices.' },
      { id: 'evidence', label: 'Grounded evidence', question: 'Did you use an experience or decision rather than mission slogans?', coaching: 'Replace one abstract principle with a concrete decision, consequence, and lesson.' },
      { id: 'independence', label: 'Independent judgment', question: 'Did you disagree or qualify a premise where the evidence required it?', coaching: 'State one company or industry position you would challenge and the evidence that would change your mind.' },
      { id: 'update', label: 'Ability to update', question: 'Could you explain a belief that changed and what caused the update?', coaching: 'Name the old belief, disconfirming evidence, and the behavior you changed afterward.' },
      sharedCommunication,
    ],
  },
  experimentation: {
    id: 'experimentation', label: 'Product & experimentation', minutes: 20,
    instruction: 'Define the decision first, then design the measurement and enumerate validity threats.',
    rubric: [
      sharedFraming,
      { id: 'metrics', label: 'Metrics', question: 'Did you choose a primary metric, guardrails, and diagnostic slices?', coaching: 'Tie one primary metric to the product decision; add guardrails for user harm, reliability, and cost.' },
      { id: 'design', label: 'Experiment design', question: 'Did you specify unit, randomization, duration, power, and exposure?', coaching: 'Name the randomization unit and identify interference, novelty, carryover, and sample-ratio mismatch risks.' },
      { id: 'decision', label: 'Decision rule', question: 'Did you state what result leads to ship, iterate, or stop?', coaching: 'Write the minimum worthwhile effect and the practical, not merely statistical, decision threshold.' },
      sharedCommunication,
    ],
  },
  research: {
    id: 'research', label: 'Research depth', minutes: 25,
    instruction: 'State the claim, test its evidence, design discriminating experiments, and identify alternatives.',
    rubric: [
      { id: 'claim', label: 'Claim & hypothesis', question: 'Was the falsifiable claim separated from motivation and implementation?', coaching: 'Rewrite the central claim so one result could prove it wrong.' },
      { id: 'evidence', label: 'Evidence quality', question: 'Did you assess baselines, variance, leakage, and alternative explanations?', coaching: 'Ask whether the strongest baseline, repeated seeds, and matched compute would preserve the result.' },
      { id: 'ablation', label: 'Ablations', question: 'Did proposed experiments isolate the claimed mechanism?', coaching: 'Design one ablation where the claim predicts a different result than the strongest alternative explanation.' },
      { id: 'limitations', label: 'Limitations & transfer', question: 'Did you identify boundary conditions and what would generalize?', coaching: 'Name the data, scale, domain, or compute condition most likely to reverse the conclusion.' },
      sharedCommunication,
    ],
  },
  infrastructure: {
    id: 'infrastructure', label: 'Systems / infrastructure', minutes: 35,
    instruction: 'Quantify the workload, identify the bottleneck, choose an architecture, and design failure recovery.',
    rubric: [
      { id: 'workload', label: 'Workload model', question: 'Did you quantify scale, throughput, latency, memory, and cost?', coaching: 'Estimate orders of magnitude before selecting technologies; state the dominant resource.' },
      { id: 'bottleneck', label: 'Bottleneck reasoning', question: 'Did you distinguish compute, memory, network, storage, and coordination limits?', coaching: 'Use a simple roofline or critical-path argument to identify what actually limits throughput.' },
      { id: 'architecture', label: 'Architecture & trade-offs', question: 'Did components and parallelism choices follow from the workload?', coaching: 'Compare two architectures using utilization, operational complexity, and failure blast radius.' },
      { id: 'reliability', label: 'Reliability', question: 'Did you cover retries, checkpointing, degradation, observability, and recovery?', coaching: 'Name the highest-probability and highest-cost failures, then define detection and recovery.' },
      sharedCommunication,
    ],
  },
  math: {
    id: 'math', label: 'Math / derivation', minutes: 15,
    instruction: 'State assumptions, derive cleanly, check dimensions or limiting cases, then interpret the result.',
    rubric: [
      { id: 'setup', label: 'Setup & assumptions', question: 'Did you define symbols, dimensions, and assumptions before deriving?', coaching: 'Write the object, its shape, and the assumption used by each major step.' },
      { id: 'derivation', label: 'Derivation', question: 'Were steps justified without skipping the key identity?', coaching: 'Find the first non-obvious equality and explain the identity or theorem that permits it.' },
      { id: 'sanity', label: 'Sanity checks', question: 'Did you check dimensions, signs, boundaries, or a simple case?', coaching: 'Test the result on a one-dimensional or symmetric special case.' },
      { id: 'interpretation', label: 'Interpretation', question: 'Did you connect the expression to model behavior or a decision?', coaching: 'Explain what increases, decreases, or breaks when one term changes.' },
      sharedCommunication,
    ],
  },
};

const MODE_BY_SUBCATEGORY: Record<string, PracticeModeId> = {
  'ML Fundamentals': 'breadth',
  'Deep Learning Production': 'infrastructure',
  'LLM Systems': 'breadth',
  'Recsys & Search': 'breadth',
  'ML System Design': 'system-design',
  'Product & Experimentation': 'experimentation',
  'Behavioral': 'behavioral',
  'Math & Research': 'research',
  'ML Implementation': 'coding',
};

const QUESTION_MODE_OVERRIDES: Record<string, PracticeModeId> = {
  'most-ambitious-project': 'project',
  'scope-ambiguous-problem': 'project',
  'implement-attention-from-scratch': 'coding',
  'build-llm-coding-assistant': 'system-design',
  'rag-for-legal-docs': 'system-design',
  'how-would-you-evaluate-an-llm-application': 'experimentation',
  'evaluate-an-agent': 'experimentation',
  'evals-for-coding-assistant': 'experimentation',
  'ab-test-chatbot': 'experimentation',
  'train-100b-model': 'infrastructure',
  'reduce-llm-inference-cost-10x': 'infrastructure',
  'derive-logistic-regression': 'math',
  'softmax-cross-entropy-pairing': 'math',
  'reparameterization-trick': 'math',
  'design-ablation-study': 'research',
  'critique-ml-paper': 'research',
  'agentic-ml-codebase-interview': 'agentic',
  'present-technical-ml-project': 'presentation',
  'debug-frontier-llm-training-run': 'coding',
  'design-production-llm-inference-service': 'system-design',
  'design-multi-team-ml-platform': 'system-design',
  'design-enterprise-agent-platform': 'system-design',
  'design-reasoning-model-fixed-budget': 'system-design',
  'design-real-time-multimodal-assistant': 'system-design',
  'design-short-form-video-ecosystem': 'system-design',
  'design-foundation-model-data-platform': 'system-design',
  'design-ai-coding-product': 'system-design',
  'design-agent-safety-control-plane': 'system-design',
  'investigate-black-box-model-behavior': 'research',
  'defend-values-under-ethical-pressure': 'values',
  'optimize-accelerator-workload': 'infrastructure',
  'design-fault-tolerant-distributed-training': 'infrastructure',
  'design-post-training-data-and-rl-environment': 'research',
  'design-llm-red-team-program': 'system-design',
  'implement-transformer-decoder': 'coding',
  'implement-kv-cache-decode': 'coding',
  'implement-beam-search': 'coding',
  'implement-lora-adapter': 'coding',
  'implement-reverse-mode-autograd': 'coding',
  'derive-ml-math-under-pressure': 'math',
};

export function getPracticeMode(slug: string, subcategory?: string): PracticeModeDefinition {
  const modeId = QUESTION_MODE_OVERRIDES[slug] ?? MODE_BY_SUBCATEGORY[subcategory ?? ''] ?? 'breadth';
  return PRACTICE_MODES[modeId];
}
