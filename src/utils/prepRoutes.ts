import { getPracticeMode, INTERVIEW_ROUNDS, type RoundId } from '../data/prepCurriculum';
import type { PrepPlanState } from './prepPlan';
import type { PracticeProgressRecord } from './prepProgress';
import { getSubcategoryMap } from './subcategories';

type QuestionPair = [string, string];

const roundQuestions: Record<RoundId, QuestionPair> = {
  'ml-breadth': ['bias-variance-tradeoff', 'how-to-choose-loss-function'],
  coding: ['debug-training-loop', 'implement-batched-top-k'],
  'agentic-codebase': ['agentic-ml-codebase-interview', 'debug-frontier-llm-training-run'],
  'ml-system-design': ['design-fraud-detection', 'design-ml-system-fixed-budget'],
  'project-deep-dive': ['most-ambitious-project', 'scope-ambiguous-problem'],
  'technical-strategy': ['design-enterprise-agent-platform', 'design-reasoning-model-fixed-budget'],
  'technical-presentation': ['present-technical-ml-project', 'most-ambitious-project'],
  behavioral: ['disagreed-with-senior', 'decide-what-to-work-on'],
  'values-mission': ['defend-values-under-ethical-pressure', 'advocated-quality-over-speed'],
  'product-experimentation': ['design-ml-ab-test', 'debug-offline-online-metric-gap'],
  'research-depth': ['critique-ml-paper', 'design-ablation-study'],
  'research-work-sample': ['investigate-black-box-model-behavior', 'design-ablation-study'],
  'math-oral': ['derive-ml-math-under-pressure', 'derive-logistic-regression'],
  'systems-infrastructure': ['train-100b-model', 'reduce-llm-inference-cost-10x'],
};

const designDomains: Record<string, QuestionPair> = {
  llm: ['design-production-llm-inference-service', 'design-ai-coding-product'],
  recsys: ['design-youtube-recommender', 'design-short-form-video-ecosystem'],
  platform: ['design-multi-team-ml-platform', 'design-foundation-model-data-platform'],
  research: ['design-reasoning-model-fixed-budget', 'design-ml-system-fixed-budget'],
  'post-training': ['design-reasoning-model-fixed-budget', 'design-production-llm-inference-service'],
  alignment: ['design-agent-safety-control-plane', 'design-llm-red-team-program'],
  multimodal: ['design-real-time-multimodal-assistant', 'design-production-llm-inference-service'],
  product: ['design-fraud-detection', 'design-ai-coding-product'],
};

const repairConcepts: Record<string, { slug: string; label: string }> = {
  'bias-variance-tradeoff': { slug: 'bias-variance-of-estimators', label: 'Bias and variance of estimators' },
  'how-to-choose-loss-function': { slug: 'cross-entropy-softmax', label: 'Cross-entropy and softmax' },
  'design-fraud-detection': { slug: 'decision-thresholds-asymmetric-costs-abstention', label: 'Decision thresholds and asymmetric costs' },
  'design-ml-monitoring': { slug: 'delayed-labels-selective-labels-feedback-loops', label: 'Delayed labels and feedback loops' },
  'design-ml-system-fixed-budget': { slug: 'transformer-compute-memory-accounting', label: 'Compute and memory accounting' },
  'design-youtube-recommender': { slug: 'two-tower-retrieval', label: 'Two-tower retrieval' },
  'design-short-form-video-ecosystem': { slug: 'position-bias-counterfactual-learning-to-rank', label: 'Position bias in ranking' },
  'design-production-llm-inference-service': { slug: 'prefill-vs-decode', label: 'Prefill versus decode' },
  'design-reasoning-model-fixed-budget': { slug: 'test-time-compute-search-verifiers', label: 'Test-time compute and verifiers' },
  'design-foundation-model-data-platform': { slug: 'foundation-model-data-curation', label: 'Foundation-model data curation' },
  'design-agent-safety-control-plane': { slug: 'scalable-oversight-and-ai-control', label: 'Oversight and AI control' },
  'design-real-time-multimodal-assistant': { slug: 'multimodal-foundation-models', label: 'Multimodal foundation models' },
  'implement-attention-from-scratch': { slug: 'attention-mechanism', label: 'Attention as weighted retrieval' },
  'implement-kv-cache-decode': { slug: 'kv-cache', label: 'KV-cache memory and reuse' },
  'implement-batched-top-k': { slug: 'approximate-nearest-neighbors', label: 'Nearest-neighbor retrieval' },
  'implement-streaming-classification-metrics': { slug: 'confusion-matrix-and-classification-metrics', label: 'Confusion matrix and metrics' },
  'debug-training-loop': { slug: 'neural-network-training-recipe', label: 'Neural-network training recipe' },
  'design-ml-ab-test': { slug: 'hypothesis-testing-confidence-intervals', label: 'Hypothesis tests and confidence intervals' },
  'debug-offline-online-metric-gap': { slug: 'data-leakage-point-in-time-correctness', label: 'Leakage and point-in-time correctness' },
  'critique-ml-paper': { slug: 'evaluation-validity-benchmark-contamination', label: 'Evaluation validity and contamination' },
  'design-ablation-study': { slug: 'reproducibility-fair-model-comparison', label: 'Reproducibility and fair comparisons' },
  'derive-logistic-regression': { slug: 'maximum-likelihood-estimation', label: 'Maximum likelihood estimation' },
  'train-100b-model': { slug: 'fsdp-and-zero', label: 'FSDP and ZeRO memory layout' },
  'reduce-llm-inference-cost-10x': { slug: 'continuous-batching', label: 'Continuous batching' },
};

export function questionPair(plan: PrepPlanState, roundId: RoundId): { slugs: QuestionPair; reason: string } {
  const base = { slugs: roundQuestions[roundId], reason: `Baseline and transfer for your ${INTERVIEW_ROUNDS.find(round => round.id === roundId)?.label} round.` };
  if (roundId === 'ml-system-design') {
    if (designDomains[plan.domain]) return { slugs: designDomains[plan.domain], reason: `${plan.domain.replaceAll('-', ' ')} domain: practice the workload and constraints you expect.` };
    if (['l6', 'l7', 'l8'].includes(plan.level)) return { slugs: ['design-multi-team-ml-platform', 'design-enterprise-agent-platform'], reason: `${plan.level.toUpperCase()} scope: defend cross-team ownership, migration, and failure recovery.` };
    if (plan.role === 'ml-engineer') return { slugs: ['design-ml-monitoring', 'design-ml-system-fixed-budget'], reason: 'MLE: start with production failures, then defend resource trade-offs.' };
  }
  if (roundId === 'coding') {
    if (['llm', 'post-training', 'alignment', 'multimodal'].includes(plan.domain) || ['research-engineer', 'research-scientist'].includes(plan.role)) return { slugs: ['implement-attention-from-scratch', 'implement-kv-cache-decode'], reason: 'Model implementation: prove tensor correctness before reasoning about decoding memory.' };
    if (plan.domain === 'recsys') return { slugs: ['implement-batched-top-k', 'implement-streaming-classification-metrics'], reason: 'Retrieval: bounded memory first, then evaluation across batches.' };
    if (plan.role === 'ml-engineer') return { slugs: ['implement-streaming-classification-metrics', 'debug-training-loop'], reason: 'MLE: test aggregation invariants, then diagnose a failing training loop.' };
  }
  if (roundId === 'research-depth' && ['post-training', 'alignment'].includes(plan.domain)) return { slugs: ['design-post-training-data-and-rl-environment', 'design-ablation-study'], reason: 'Post-training research: connect data and reward choices to discriminating experiments.' };
  return base;
}

export type RouteStatus = 'new' | 'due' | 'scheduled' | 'mixed' | 'mastered';

export function practiceStatus(record: PracticeProgressRecord | undefined, today: string): RouteStatus {
  if (!record) return 'new';
  if (record.successfulAttempts >= 2 && record.weakDimensions.length === 0 && record.score === 'Confident') {
    if (record.mixedVerifiedOn && record.mixedVerifiedOn > (record.lastSuccessfulOn ?? record.lastAttemptOn)) return 'mastered';
    return 'mixed';
  }
  return record.dueOn && record.dueOn > today ? 'scheduled' : 'due';
}

export function buildStudyRoutes(plan: PrepPlanState, records: PracticeProgressRecord[], today: string) {
  return INTERVIEW_ROUNDS.filter(round => plan.selectedRounds.includes(round.id)).map(round => {
    const selected = questionPair(plan, round.id);
    const evidence = Object.entries(round.areaWeights).reduce((sum, [area, weight]) => sum + (plan.areaRatings[area] ?? 0) * weight, 0);
    const steps = selected.slugs.map((slug, index) => {
      const record = records.find(item => item.slug === slug);
      const practice = getPracticeMode(slug, getSubcategoryMap('questions')?.map[slug]);
      const label = INTERVIEW_ROUNDS.flatMap(item => item.starterLinks).find(link => link.href === `/questions/${slug}/`)?.label
        ?? slug.replaceAll('-', ' ').replace(/^./, letter => letter.toUpperCase());
      const repair = repairConcepts[slug] ?? { slug, label: 'Worked visual for this prompt' };
      return { slug, label, stage: index === 0 ? 'Diagnostic' : 'Transfer', href: `/questions/${slug}/?practice=1`, visualHref: `/review/#${repair.slug}`, repairLabel: repair.label, practice, record, status: practiceStatus(record, today) };
    });
    const hasWeakAttempt = steps.some(step => step.record && (step.record.score === 'Weak' || step.record.weakDimensions.length > 0));
    return { id: round.id, label: round.label, reason: selected.reason, evidence, priority: (5 - evidence) + (hasWeakAttempt ? 5 : 0), steps, sessionMinutes: steps[0].practice.minutes + 15 };
  }).sort((left, right) => right.priority - left.priority);
}

export function studyBudget(plan: PrepPlanState, routes: ReturnType<typeof buildStudyRoutes>) {
  const weeklyMinutes = Math.max(0, Math.floor(plan.weeklyHours * 60));
  const repairMinutes = Math.floor(weeklyMinutes / 2);
  let remaining = weeklyMinutes - repairMinutes;
  const sessions = routes.filter(route => route.steps.some(step => step.status === 'new' && (step.stage === 'Diagnostic' || Boolean(route.steps[0].record?.successfulAttempts)))).flatMap(route => {
    if (route.sessionMinutes > remaining) return [];
    remaining -= route.sessionMinutes;
    return [{ roundId: route.id, minutes: route.sessionMinutes }];
  });
  const baselineMinutes = routes.filter(route => !route.steps[0].record).reduce((sum, route) => sum + route.sessionMinutes, 0);
  return { weeklyMinutes, repairMinutes, sessions, unassignedMinutes: remaining, baselineMinutes, availableMinutes: weeklyMinutes * plan.availableWeeks };
}