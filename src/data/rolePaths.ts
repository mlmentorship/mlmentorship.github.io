export interface RolePathStep {
  title: string;
  href: string;
  detail: string;
}

export interface RolePath {
  id: 'applied-scientist' | 'research-scientist' | 'ml-engineer' | 'research-engineer';
  title: string;
  shortTitle: string;
  rule: string;
  firstPracticeHref: string;
  sequence: RolePathStep[];
}

export interface LevelPath {
  id: 'staff-principal';
  title: string;
  rule: string;
  firstPracticeHref: string;
  sequence: RolePathStep[];
}

export const ROLE_PATHS: RolePath[] = [
  {
    id: 'applied-scientist',
    title: 'Applied Scientist',
    shortTitle: 'AS',
    rule: 'Preserve scientific depth, but prove scoping, experimentation, shipping, and influence.',
    firstPracticeHref: '/questions/scope-ambiguous-problem/?practice=1',
    sequence: [
      { title: 'Calibrate role and level', href: '/guides/as-vs-mle-vs-re/', detail: 'Read the role taxonomy and senior-through-principal calibration; choose the target rubric.' },
      { title: 'Scope before solving', href: '/questions/scope-ambiguous-problem/', detail: 'Practice objective, user, data, constraints, and the smallest discriminating experiment.' },
      { title: 'Prove ML breadth', href: '/questions/how-to-choose-loss-function/', detail: 'Use distributions, costs, mechanisms, and alternatives rather than memorized recipes.' },
      { title: 'Design end to end', href: '/questions/design-fraud-detection/', detail: 'Connect data, baseline, model, evaluation, serving, monitoring, and iteration.' },
      { title: 'Make a product decision', href: '/questions/design-ml-ab-test/', detail: 'Define the causal estimand, metrics, validity checks, uncertainty, and ship rule.' },
      { title: 'Debug evidence disagreement', href: '/questions/debug-offline-online-metric-gap/', detail: 'Separate objective mismatch, point-in-time leakage, serving skew, and feedback effects.' },
      { title: 'Defend project ownership', href: '/questions/most-ambitious-project/', detail: 'Prepare a short opening and a 20-minute technical deep dive.' },
      { title: 'Show judgment under pressure', href: '/questions/advocated-quality-over-speed/', detail: 'Balance legitimate delivery pressure with measurable user risk.' },
      { title: 'Simulate the loop', href: '/prep/simulations/#applied-scientist', detail: 'Run the AS packet and reduce results to three bounded repairs.' },
    ],
  },
  {
    id: 'research-scientist',
    title: 'Research Scientist',
    shortTitle: 'RS',
    rule: 'Prove mathematical depth, executable experiments, research taste, and ownership of the central claims.',
    firstPracticeHref: '/questions/critique-ml-paper/?practice=1',
    sequence: [
      { title: 'Confirm the research loop', href: '/guides/as-vs-mle-vs-re/', detail: 'Separate research discussion, coding, math, brainstorm, paper defense, and job-talk expectations.' },
      { title: 'Defend a research claim', href: '/questions/critique-ml-paper/', detail: 'Separate the central claim from evidence, assumptions, limitations, and the strongest alternative explanation.' },
      { title: 'Derive under pressure', href: '/questions/derive-ml-math-under-pressure/', detail: 'Use moments and distributions, state assumptions, check dimensions, and handle one changed condition.' },
      { title: 'Keep experiments executable', href: '/questions/implement-attention-from-scratch/', detail: 'Implement a core mechanism with correct tensor contracts, tests, and complexity.' },
      { title: 'Design discriminating evidence', href: '/questions/design-ablation-study/', detail: 'Control compute and tuning, preserve paired comparisons, and report resampled uncertainty.' },
      { title: 'Investigate unknown behavior', href: '/questions/investigate-black-box-model-behavior/', detail: 'Generate competing hypotheses and the smallest probes that separate them.' },
      { title: 'Prepare the job talk', href: '/prep/presentation/', detail: 'Organize the talk around claims, decisions, failed paths, evidence, and future work.' },
      { title: 'Show research collaboration', href: '/questions/disagreed-with-senior/', detail: 'Use a real disagreement and show how evidence changed the decision or your own view.' },
      { title: 'Simulate the loop', href: '/prep/simulations/#research-scientist', detail: 'Run the RS packet with unfamiliar prompts and reduce results to three bounded repairs.' },
    ],
  },
  {
    id: 'ml-engineer',
    title: 'Machine Learning Engineer',
    shortTitle: 'MLE',
    rule: 'Do not substitute ML reading for software execution. Code, debug, and design reliable systems repeatedly.',
    firstPracticeHref: '/questions/implement-streaming-classification-metrics/?practice=1',
    sequence: [
      { title: 'Confirm the loop', href: '/guides/as-vs-mle-vs-re/', detail: 'Ask whether implementation means general algorithms, ML primitives, debugging, or all three.' },
      { title: 'Establish an ML implementation baseline', href: '/questions/implement-streaming-classification-metrics/', detail: 'Build mergeable model-evaluation logic with tests, edge cases, and explicit metric semantics.' },
      { title: 'Add bounded-memory implementation', href: '/questions/implement-batched-top-k/', detail: 'Reason about vectorization, memory, and exact-versus-ANN trade-offs.' },
      { title: 'Debug systematically', href: '/questions/debug-training-loop/', detail: 'State the diagnostic order before editing code.' },
      { title: 'Design reliable serving', href: '/questions/design-ml-monitoring/', detail: 'Cover point-in-time data, thresholds, model outcomes, alerts, ownership, and rollback.' },
      { title: 'Make cost explicit', href: '/questions/design-ml-system-fixed-budget/', detail: 'Quantify workload and quality, cost, and latency frontiers.' },
      { title: 'Debug offline and online disagreement', href: '/questions/debug-offline-online-metric-gap/', detail: 'Separate leakage, experiment integrity, serving skew, proxy mismatch, and feedback effects.' },
      { title: 'Prepare failure and recovery stories', href: '/prep/story-bank/', detail: 'Use failure, incident recovery, trade-off, and cross-team stories.' },
      { title: 'Simulate the loop', href: '/prep/simulations/#ml-engineer', detail: 'Run ML implementation, design, production, breadth, and project rounds.' },
    ],
  },
  {
    id: 'research-engineer',
    title: 'Research Engineer',
    shortTitle: 'RE',
    rule: 'Balance implementation and systems depth with scientific skepticism. Confirm whether the title is research-heavy or software-heavy.',
    firstPracticeHref: '/questions/implement-attention-from-scratch/?practice=1',
    sequence: [
      { title: 'Confirm role mix', href: '/guides/as-vs-mle-vs-re/', detail: 'Map research, modeling, engineering, and product expectations.' },
      { title: 'Implement a core primitive', href: '/questions/implement-attention-from-scratch/', detail: 'Build the mechanism correctly and explain dimensions and complexity.' },
      { title: 'Add systems implementation', href: '/questions/implement-batched-top-k/', detail: 'Handle memory and performance without hiding behind a library.' },
      { title: 'Model the workload', href: '/questions/plan-70b-training-run/', detail: 'Calculate parameters, memory, FLOPs, batch, communication, time, and a measured parallel layout.' },
      { title: 'Reason about inference', href: '/questions/reduce-llm-inference-cost-10x/', detail: 'Use measurement and cost models rather than a list of optimizations.' },
      { title: 'Design discriminating evidence', href: '/questions/design-ablation-study/', detail: 'Control compute and tuning, use paired resampling, and test alternative explanations.' },
      { title: 'Critique research fairly', href: '/questions/critique-ml-paper/', detail: 'Prioritize the threat that most changes the central claim.' },
      { title: 'Prepare translation stories', href: '/prep/story-bank/', detail: 'Show failed experiments, implementation decisions, and research-to-production impact.' },
      { title: 'Simulate the loop', href: '/prep/simulations/#research-engineer', detail: 'Run ML implementation, systems, research depth, breadth, and project rounds.' },
    ],
  },
];

export const STAFF_PRINCIPAL_PATH: LevelPath = {
  id: 'staff-principal',
  title: 'Staff through senior-principal ML',
  rule: 'Keep the role-specific technical bar, then prove problem selection, cross-team direction, portfolio judgment, delegated technical authority, succession, and reversible strategy.',
  firstPracticeHref: '/questions/design-multi-team-ml-platform/?practice=1',
  sequence: [
    { title: 'Calibrate scope with evidence', href: '/guides/l5-vs-l6-faang-ml/', detail: 'Map stories by problem ownership, technical depth, influence, portfolio scope, delegated authority, durability, and succession.' },
    { title: 'Baseline a staff architecture case', href: '/questions/design-multi-team-ml-platform/', detail: 'Attempt the case closed-book before reading. Preserve contracts, migration, ownership, adoption, and reversibility.' },
    { title: 'Design an enterprise agent platform', href: '/questions/design-enterprise-agent-platform/', detail: 'Defend delegated authority, tool effects, durable execution, memory, evaluation, regional policy, and provider portability.' },
    { title: 'Study an annotated upper-IC mock', href: '/guides/annotated-upper-ic-agent-platform-mock/', detail: 'Compare each challenged turn, weak alternative, score change, and retry drill before repeating with new constraints.' },
    { title: 'Defend problem selection', href: '/questions/decide-what-to-work-on/', detail: 'Compare credible investments and explain the counterfactual cost of the work you chose.' },
    { title: 'Reframe an ambiguous mandate', href: '/questions/scope-ambiguous-problem/', detail: 'Challenge the requested solution only when evidence supports a better problem definition.' },
    { title: 'Quantify architecture trade-offs', href: '/questions/design-ml-system-fixed-budget/', detail: 'Turn quality, latency, reliability, staffing, and cost into an explicit decision frontier.' },
    { title: 'Defend high-scope ownership', href: '/questions/most-ambitious-project/', detail: 'Partition your decisions from team execution and connect influence to measured operating outcomes.' },
    { title: 'Expose a wrong bet and recovery', href: '/questions/killed-ml-project/', detail: 'Show the disconfirming signal, cost of delay, stakeholder impact, and the direction you changed.' },
    { title: 'Practice influence under conflict', href: '/questions/disagreed-with-senior/', detail: 'Explain incentives, evidence, and how the decision improved, including cases where your view changed.' },
    { title: 'Simulate the level bar', href: '/prep/simulations/#senior-principal', detail: 'Run architecture, technical strategy, project, delegated-leadership, and retained-domain-depth rounds with an experienced observer.' },
  ],
};

export function getRolePath(id: string): RolePath | undefined {
  return ROLE_PATHS.find((role) => role.id === id);
}
