import { getPracticeMode, INTERVIEW_ROUNDS, type RoundId } from '../data/prepCurriculum';
import type { PrepPlanState } from './prepPlan';
import type { PracticeProgressRecord } from './prepProgress';
import { getSubcategoryMap } from './subcategories';

type QuestionPair = [string, string];
type QuestionSelection = { slugs: string[]; reason: string };

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
  'advocated-quality-over-speed': { slug: 'advocated-quality-over-speed', label: 'Quality or safety under pressure' },
  'agentic-ml-codebase-interview': { slug: 'agentic-ml-codebase-interview', label: 'Agentic codebase method' },
  'debug-frontier-llm-training-run': { slug: 'debug-frontier-llm-training-run', label: 'Frontier training-run diagnosis' },
  'decide-what-to-work-on': { slug: 'decide-what-to-work-on', label: 'Prioritization and decision evidence' },
  'defend-values-under-ethical-pressure': { slug: 'defend-values-under-ethical-pressure', label: 'Values under pressure' },
  'derive-ml-math-under-pressure': { slug: 'derive-ml-math-under-pressure', label: 'ML derivation under pressure' },
  'design-ai-coding-product': { slug: 'design-ai-coding-product', label: 'AI coding product design' },
  'design-enterprise-agent-platform': { slug: 'design-enterprise-agent-platform', label: 'Enterprise agent platform design' },
  'design-llm-red-team-program': { slug: 'design-llm-red-team-program', label: 'LLM red-team program design' },
  'design-multi-team-ml-platform': { slug: 'design-multi-team-ml-platform', label: 'Multi-team ML platform design' },
  'design-post-training-data-and-rl-environment': { slug: 'design-post-training-data-and-rl-environment', label: 'Post-training data and RL environment' },
  'disagreed-with-senior': { slug: 'disagreed-with-senior', label: 'Technical disagreement and updating' },
  'investigate-black-box-model-behavior': { slug: 'investigate-black-box-model-behavior', label: 'Black-box investigation method' },
  'most-ambitious-project': { slug: 'most-ambitious-project', label: 'Project ownership and evidence' },
  'present-technical-ml-project': { slug: 'present-technical-ml-project', label: 'Technical project presentation' },
  'scope-ambiguous-problem': { slug: 'scope-ambiguous-problem', label: 'Scoping an ambiguous problem' },
  'killed-ml-project': { slug: 'killed-ml-project', label: 'Learning from a stopped project' },
  'debug-model-not-learning': { slug: 'neural-network-training-recipe', label: 'Neural-network training recipe' },
  'design-feature-store': { slug: 'ml-data-lineage-versioning', label: 'Feature and data lineage' },
  'how-would-you-evaluate-an-llm-application': { slug: 'llm-as-judge', label: 'LLM evaluation validity' },
  'evaluate-an-agent': { slug: 'evaluation-validity-benchmark-contamination', label: 'Agent evaluation validity' },
  'ab-test-chatbot': { slug: 'hypothesis-testing-confidence-intervals', label: 'Experiment uncertainty' },
  'choose-ml-product-metrics': { slug: 'decision-thresholds-asymmetric-costs-abstention', label: 'Metrics, costs, and thresholds' },
  'evaluate-search-ranker': { slug: 'position-bias-counterfactual-learning-to-rank', label: 'Ranking evaluation and position bias' },
  'design-fault-tolerant-distributed-training': { slug: 'fault-tolerant-collectives', label: 'Fault-tolerant collectives' },
  'optimize-accelerator-workload': { slug: 'profiling-distributed-ml-workloads', label: 'Profiling distributed workloads' },
  'plan-70b-training-run': { slug: 'transformer-compute-memory-accounting', label: 'Training compute and memory accounting' },
};

export function questionPair(plan: PrepPlanState, roundId: RoundId): { slugs: QuestionPair; reason: string } {
  const selected = questionSelection(plan, roundId);
  return { slugs: [selected.slugs[0], selected.slugs[1]], reason: selected.reason };
}

function questionSelection(plan: PrepPlanState, roundId: RoundId): QuestionSelection {
  const base = { slugs: roundQuestions[roundId], reason: `Baseline and transfer for your ${INTERVIEW_ROUNDS.find(round => round.id === roundId)?.label} round.` };
  const upperIc = ['l6', 'l7', 'l8'].includes(plan.level);
  const levelScope = plan.level === 'l4'
    ? 'L4 scope: establish a correct baseline and explain the main trade-offs'
    : plan.level === 'l5'
      ? 'L5 scope: make an autonomous, evidence-backed decision and handle follow-up'
      : plan.level === 'l6'
        ? 'Staff scope: define interfaces, ownership, adoption, and recovery across teams'
        : plan.level === 'l7'
          ? 'Principal scope: set direction, sequence investment, delegate authority, and show reversal evidence'
          : 'Senior-principal scope: defend enterprise boundaries, succession, portability, and durable technical doctrine';
  if (roundId === 'ml-system-design') {
    const domainPools: Record<string, string[]> = {
      llm: upperIc ? ['design-production-llm-inference-service', 'design-ai-coding-product', 'design-real-time-multimodal-assistant'] : ['design-production-llm-inference-service', 'design-ai-coding-product', 'design-reasoning-model-fixed-budget'],
      recsys: upperIc ? ['design-short-form-video-ecosystem', 'design-youtube-recommender', 'design-enterprise-agent-platform'] : ['design-youtube-recommender', 'design-short-form-video-ecosystem', 'design-fraud-detection'],
      platform: ['design-multi-team-ml-platform', 'design-foundation-model-data-platform', 'design-enterprise-agent-platform'],
      research: upperIc ? ['design-reasoning-model-fixed-budget', 'design-ml-system-fixed-budget', 'design-post-training-data-and-rl-environment'] : ['design-reasoning-model-fixed-budget', 'design-ml-system-fixed-budget', 'design-post-training-data-and-rl-environment'],
      'post-training': ['design-post-training-data-and-rl-environment', 'design-reasoning-model-fixed-budget', 'design-production-llm-inference-service'],
      alignment: ['design-agent-safety-control-plane', 'design-llm-red-team-program', 'design-enterprise-agent-platform'],
      multimodal: ['design-real-time-multimodal-assistant', 'design-production-llm-inference-service', 'design-foundation-model-data-platform'],
      product: ['design-fraud-detection', 'design-ai-coding-product', 'design-short-form-video-ecosystem'],
    };
    if (domainPools[plan.domain]) return { slugs: domainPools[plan.domain], reason: `${levelScope}. ${plan.domain.replaceAll('-', ' ')} domain: practice the workload, constraints, and failure costs you expect.` };
    if (upperIc) return { slugs: ['design-multi-team-ml-platform', 'design-enterprise-agent-platform', 'design-short-form-video-ecosystem'], reason: `${levelScope}.` };
    if (plan.role === 'ml-engineer') return { slugs: ['design-ml-monitoring', 'design-ml-system-fixed-budget', 'design-feature-store'], reason: `${levelScope}. MLE: start with production failures, then defend resource trade-offs.` };
  }
  if (roundId === 'coding') {
    if (['llm', 'post-training', 'alignment', 'multimodal'].includes(plan.domain) || ['research-engineer', 'research-scientist'].includes(plan.role)) return { slugs: ['implement-attention-from-scratch', 'implement-kv-cache-decode'], reason: 'Model implementation: prove tensor correctness before reasoning about decoding memory.' };
    if (plan.domain === 'recsys') return { slugs: ['implement-batched-top-k', 'implement-streaming-classification-metrics'], reason: 'Retrieval: bounded memory first, then evaluation across batches.' };
    if (plan.role === 'ml-engineer') return { slugs: ['implement-streaming-classification-metrics', 'debug-training-loop'], reason: 'MLE: test aggregation invariants, then diagnose a failing training loop.' };
  }
  if (roundId === 'technical-strategy') {
    const strategyPools: Record<string, string[]> = {
      llm: ['design-reasoning-model-fixed-budget', 'design-enterprise-agent-platform', 'design-ai-coding-product', 'design-short-form-video-ecosystem'],
      recsys: ['design-short-form-video-ecosystem', 'design-enterprise-agent-platform', 'design-youtube-recommender', 'design-multi-team-ml-platform'],
      platform: ['design-foundation-model-data-platform', 'design-enterprise-agent-platform', 'design-multi-team-ml-platform', 'design-fault-tolerant-distributed-training'],
      research: ['design-reasoning-model-fixed-budget', 'design-post-training-data-and-rl-environment', 'design-enterprise-agent-platform', 'design-ai-coding-product'],
      'post-training': ['design-post-training-data-and-rl-environment', 'design-reasoning-model-fixed-budget', 'design-enterprise-agent-platform', 'design-ai-coding-product'],
      alignment: ['design-agent-safety-control-plane', 'design-llm-red-team-program', 'design-enterprise-agent-platform'],
      multimodal: ['design-real-time-multimodal-assistant', 'design-enterprise-agent-platform', 'design-foundation-model-data-platform'],
      product: ['design-short-form-video-ecosystem', 'design-ai-coding-product', 'design-enterprise-agent-platform'],
    };
    return { slugs: strategyPools[plan.domain] ?? ['design-enterprise-agent-platform', 'design-short-form-video-ecosystem', 'design-reasoning-model-fixed-budget', 'design-multi-team-ml-platform'], reason: `${levelScope}. Make the investment order, decision rights, migration cost, and reversal evidence explicit.` };
  }
  if (roundId === 'project-deep-dive') return { slugs: upperIc ? ['most-ambitious-project', 'killed-ml-project', 'scope-ambiguous-problem'] : ['most-ambitious-project', 'scope-ambiguous-problem', 'killed-ml-project'], reason: `${levelScope}. Project evidence must make personal decisions, alternatives, failure, and measured impact concrete.` };
  if (roundId === 'technical-presentation') return { slugs: ['present-technical-ml-project', 'killed-ml-project', 'scope-ambiguous-problem', 'most-ambitious-project'], reason: `${levelScope}. Communicate one project clearly, then defend its evidence, failures, and changed assumptions.` };
  if (roundId === 'product-experimentation') {
    const productPools: Record<string, string[]> = {
      llm: ['how-would-you-evaluate-an-llm-application', 'evaluate-an-agent', 'ab-test-chatbot'],
      recsys: ['evaluate-search-ranker', 'choose-ml-product-metrics', 'debug-offline-online-metric-gap'],
      platform: ['design-ml-ab-test', 'debug-offline-online-metric-gap', 'choose-ml-product-metrics'],
      research: ['design-ml-ab-test', 'choose-ml-product-metrics', 'debug-offline-online-metric-gap'],
      'post-training': ['evaluate-an-agent', 'ab-test-chatbot', 'design-ml-ab-test'],
      alignment: ['evaluate-an-agent', 'ab-test-chatbot', 'choose-ml-product-metrics'],
      multimodal: ['how-would-you-evaluate-an-llm-application', 'ab-test-chatbot', 'choose-ml-product-metrics'],
      product: ['choose-ml-product-metrics', 'design-ml-ab-test', 'debug-offline-online-metric-gap'],
    };
    return { slugs: productPools[plan.domain] ?? ['design-ml-ab-test', 'choose-ml-product-metrics', 'debug-offline-online-metric-gap'], reason: `${plan.domain.replaceAll('-', ' ')} experimentation: connect the product decision to metrics, validity threats, guardrails, and the ship rule.` };
  }
  if (roundId === 'systems-infrastructure') {
    const systemsPools: Record<string, string[]> = {
      llm: ['reduce-llm-inference-cost-10x', 'design-production-llm-inference-service', 'train-100b-model'],
      platform: ['design-fault-tolerant-distributed-training', 'plan-70b-training-run', 'optimize-accelerator-workload'],
      research: ['plan-70b-training-run', 'train-100b-model', 'design-fault-tolerant-distributed-training'],
      'post-training': ['plan-70b-training-run', 'design-fault-tolerant-distributed-training', 'reduce-llm-inference-cost-10x'],
      alignment: ['design-fault-tolerant-distributed-training', 'reduce-llm-inference-cost-10x', 'train-100b-model'],
      multimodal: ['design-production-llm-inference-service', 'reduce-llm-inference-cost-10x', 'train-100b-model'],
    };
    const slugs = systemsPools[plan.domain] ?? (['ml-engineer', 'research-engineer'].includes(plan.role)
      ? ['design-fault-tolerant-distributed-training', 'train-100b-model', 'optimize-accelerator-workload']
      : ['train-100b-model', 'reduce-llm-inference-cost-10x', 'design-fault-tolerant-distributed-training']);
    return { slugs, reason: `${plan.domain.replaceAll('-', ' ')} systems: quantify the workload, identify the bottleneck, choose the architecture, and define recovery.` };
  }
  if (roundId === 'research-depth' && ['post-training', 'alignment'].includes(plan.domain)) return { slugs: ['design-post-training-data-and-rl-environment', 'design-ablation-study'], reason: 'Post-training research: connect data and reward choices to discriminating experiments.' };
  if (roundId === 'research-work-sample') return { slugs: ['investigate-black-box-model-behavior', 'debug-frontier-llm-training-run', 'design-post-training-data-and-rl-environment', 'debug-model-not-learning'], reason: `${plan.domain.replaceAll('-', ' ')} research work sample: form competing hypotheses, choose discriminating probes, and report measured evidence.` };
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
  const routeInputs = INTERVIEW_ROUNDS.filter(round => plan.selectedRounds.includes(round.id)).map(round => {
    const selected = questionSelection(plan, round.id);
    const evidence = Object.entries(round.areaWeights).reduce((sum, [area, weight]) => sum + (plan.areaRatings[area] ?? 0) * weight, 0);
    const hasWeakAttempt = selected.slugs.some(slug => {
      const record = records.find(item => item.slug === slug);
      return record && (record.score === 'Weak' || record.weakDimensions.length > 0);
    });
    return { round, selected, evidence, priority: (5 - evidence) + (hasWeakAttempt ? 5 : 0) };
  });
  const usedSlugs = new Set<string>();
  return routeInputs.map(({ round, selected, evidence, priority }) => {
    const fresh = [...new Set(selected.slugs)].filter(slug => !usedSlugs.has(slug));
    const slugs = [...new Set(fresh.length >= 2 ? fresh : [...fresh, ...roundQuestions[round.id].filter(slug => !usedSlugs.has(slug))])].slice(0, 2);
    slugs.forEach(slug => usedSlugs.add(slug));
    const steps = slugs.map((slug, index) => {
      const record = records.find(item => item.slug === slug);
      const practice = getPracticeMode(slug, getSubcategoryMap('questions')?.map[slug]);
      const label = INTERVIEW_ROUNDS.flatMap(item => item.starterLinks).find(link => link.href === `/questions/${slug}/`)?.label
        ?? slug.replaceAll('-', ' ').replace(/^./, letter => letter.toUpperCase());
      const repair = repairConcepts[slug] ?? { slug, label: 'Worked visual for this prompt' };
      return { slug, label, stage: index === 0 ? 'Diagnostic' : 'Transfer', href: `/questions/${slug}/?practice=1`, visualHref: `/review/#${repair.slug}`, repairLabel: repair.label, practice, record, status: practiceStatus(record, today) };
    });
    return { id: round.id, label: round.label, reason: selected.reason, evidence, priority, steps, sessionMinutes: steps[0].practice.minutes + 15 };
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