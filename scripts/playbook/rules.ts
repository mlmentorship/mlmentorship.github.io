import type {
  CatalogResource,
  DomainTrack,
  InterviewRound,
  Level,
  PlanArea,
  Role,
  TaskType,
} from './types';

export const ENGINE_VERSION = '1.0.0';

export const ROLE_LABELS: Record<Role, string> = {
  'applied-scientist': 'Applied Scientist',
  'ml-engineer': 'Machine Learning Engineer',
  'research-engineer': 'Research Engineer',
};

export const LEVEL_LABELS: Record<Level, string> = {
  l4: 'L4 / mid-level',
  l5: 'L5 / senior',
  l6: 'L6 / staff',
};

export const AREA_LABELS: Record<PlanArea, string> = {
  fundamentals: 'ML fundamentals',
  production: 'Production & debugging',
  'llm-systems': 'LLM systems',
  'recsys-search': 'Retrieval, search & recommendations',
  'system-design': 'ML system design',
  behavioral: 'Behavioral & leadership',
  'math-research': 'Math & research depth',
  coding: 'Coding & implementation',
};

export const ROUND_LABELS: Record<InterviewRound, string> = {
  'ml-fundamentals': 'ML fundamentals',
  'production-debugging': 'Production & debugging',
  'llm-systems': 'LLM systems',
  'recsys-search': 'Retrieval / search / recommendations',
  'ml-system-design': 'ML system design',
  behavioral: 'Behavioral & leadership',
  'math-research': 'Math & research depth',
  coding: 'Coding & implementation',
};

export const ROUND_AREA: Record<InterviewRound, PlanArea> = {
  'ml-fundamentals': 'fundamentals',
  'production-debugging': 'production',
  'llm-systems': 'llm-systems',
  'recsys-search': 'recsys-search',
  'ml-system-design': 'system-design',
  behavioral: 'behavioral',
  'math-research': 'math-research',
  coding: 'coding',
};

export const ROLE_WEIGHTS: Record<Role, Record<PlanArea, number>> = {
  'applied-scientist': {
    fundamentals: 1.15,
    production: 0.85,
    'llm-systems': 1.1,
    'recsys-search': 1,
    'system-design': 1.35,
    behavioral: 1.2,
    'math-research': 1.15,
    coding: 0.65,
  },
  'ml-engineer': {
    fundamentals: 1,
    production: 1.35,
    'llm-systems': 1.25,
    'recsys-search': 1.05,
    'system-design': 1.25,
    behavioral: 0.9,
    'math-research': 0.7,
    coding: 1.25,
  },
  'research-engineer': {
    fundamentals: 1.2,
    production: 1.25,
    'llm-systems': 1.15,
    'recsys-search': 0.75,
    'system-design': 0.9,
    behavioral: 0.85,
    'math-research': 1.35,
    coding: 1.25,
  },
};

export const SUBCATEGORY_AREA: Record<string, PlanArea> = {
  'ML Fundamentals': 'fundamentals',
  'Deep Learning Production': 'production',
  'LLM Systems': 'llm-systems',
  'Recsys & Search': 'recsys-search',
  'ML System Design': 'system-design',
  Behavioral: 'behavioral',
  Math: 'math-research',
  Coding: 'coding',
  'Linear Algebra & Math': 'math-research',
  'Probability & Statistics': 'math-research',
  'Classical ML': 'fundamentals',
  'Deep Learning Foundations': 'fundamentals',
  'Generative Models': 'math-research',
  'Probabilistic Models': 'math-research',
  'Reinforcement Learning': 'math-research',
  'Computer Vision': 'fundamentals',
  'NLP & Speech': 'fundamentals',
  'Retrieval & Recommenders': 'recsys-search',
  'LLM Internals': 'llm-systems',
  'Training Fundamentals': 'production',
  'Systems & Infrastructure': 'production',
  'ML Systems & Evaluation': 'system-design',
  Guides: 'system-design',
};

export const AREA_ROLES: Record<PlanArea, Role[]> = {
  fundamentals: ['applied-scientist', 'ml-engineer', 'research-engineer'],
  production: ['ml-engineer', 'research-engineer', 'applied-scientist'],
  'llm-systems': ['ml-engineer', 'applied-scientist', 'research-engineer'],
  'recsys-search': ['applied-scientist', 'ml-engineer', 'research-engineer'],
  'system-design': ['applied-scientist', 'ml-engineer', 'research-engineer'],
  behavioral: ['applied-scientist', 'ml-engineer', 'research-engineer'],
  'math-research': ['research-engineer', 'applied-scientist', 'ml-engineer'],
  coding: ['research-engineer', 'ml-engineer', 'applied-scientist'],
};

export const AREA_ROUNDS: Record<PlanArea, InterviewRound[]> = {
  fundamentals: ['ml-fundamentals'],
  production: ['production-debugging'],
  'llm-systems': ['llm-systems'],
  'recsys-search': ['recsys-search'],
  'system-design': ['ml-system-design'],
  behavioral: ['behavioral'],
  'math-research': ['math-research'],
  coding: ['coding'],
};

interface ResourceOverride {
  taskType?: Exclude<TaskType, 'review' | 'simulation'>;
  minutes?: number;
  areas?: PlanArea[];
  roles?: Role[];
  levels?: Level[];
  rounds?: InterviewRound[];
  domains?: DomainTrack[];
  priority?: number;
  prerequisites?: string[];
}

const ALL_ROLES: Role[] = ['applied-scientist', 'ml-engineer', 'research-engineer'];
const ALL_LEVELS: Level[] = ['l4', 'l5', 'l6'];

export const RESOURCE_OVERRIDES: Record<string, ResourceOverride> = {
  'as-vs-mle-vs-re': { taskType: 'read', minutes: 20, roles: ALL_ROLES, levels: ALL_LEVELS, priority: 100 },
  'five-things-as-interview-tests': { taskType: 'read', minutes: 25, roles: ALL_ROLES, levels: ['l5', 'l6'], areas: ['behavioral', 'system-design'], priority: 96 },
  'l5-vs-l6-faang-ml': { taskType: 'read', minutes: 25, roles: ALL_ROLES, levels: ['l5', 'l6'], areas: ['behavioral'], priority: 98 },
  'most-ambitious-project': { taskType: 'story', minutes: 45, roles: ALL_ROLES, levels: ALL_LEVELS, areas: ['behavioral'], rounds: ['behavioral'], priority: 100 },
  'scope-ambiguous-problem': { taskType: 'story', minutes: 40, roles: ALL_ROLES, levels: ['l5', 'l6'], areas: ['behavioral', 'system-design'], rounds: ['behavioral', 'ml-system-design'], priority: 95 },
  'disagreed-with-senior': { taskType: 'story', minutes: 40, roles: ALL_ROLES, levels: ['l5', 'l6'], areas: ['behavioral'], rounds: ['behavioral'], priority: 88 },
  'decide-what-to-work-on': { taskType: 'story', minutes: 40, roles: ALL_ROLES, levels: ['l5', 'l6'], areas: ['behavioral'], rounds: ['behavioral'], priority: 86 },

  'bias-variance-tradeoff': { taskType: 'practice', minutes: 35, roles: ALL_ROLES, levels: ALL_LEVELS, areas: ['fundamentals'], rounds: ['ml-fundamentals'], priority: 90 },
  'explain-backprop': { taskType: 'practice', minutes: 35, roles: ALL_ROLES, levels: ALL_LEVELS, areas: ['fundamentals', 'math-research'], rounds: ['ml-fundamentals', 'math-research'], priority: 90 },
  'how-to-choose-learning-rate': { taskType: 'practice', minutes: 35, roles: ['ml-engineer', 'research-engineer', 'applied-scientist'], levels: ALL_LEVELS, areas: ['production'], rounds: ['production-debugging'], priority: 84 },
  'debug-model-not-learning': { taskType: 'practice', minutes: 40, roles: ALL_ROLES, levels: ALL_LEVELS, areas: ['production'], rounds: ['production-debugging'], priority: 94 },
  'debug-training-loop': { taskType: 'practice', minutes: 60, roles: ['ml-engineer', 'research-engineer'], levels: ALL_LEVELS, areas: ['coding', 'production'], rounds: ['coding', 'production-debugging'], priority: 93 },
  'implement-knn': { taskType: 'practice', minutes: 50, roles: ['ml-engineer', 'research-engineer'], levels: ['l4', 'l5'], areas: ['coding'], rounds: ['coding'], priority: 78 },
  'implement-attention-from-scratch': { taskType: 'practice', minutes: 60, roles: ['research-engineer', 'ml-engineer'], levels: ['l5', 'l6'], areas: ['coding', 'llm-systems'], rounds: ['coding', 'llm-systems'], domains: ['llm'], priority: 94, prerequisites: ['attention-mechanism', 'transformer-architecture'] },

  'how-would-you-evaluate-an-llm-application': { taskType: 'design', minutes: 50, roles: ALL_ROLES, levels: ['l5', 'l6'], areas: ['llm-systems', 'system-design'], rounds: ['llm-systems', 'ml-system-design'], domains: ['llm'], priority: 100, prerequisites: ['llm-evals-the-hardest-part'] },
  'evaluate-an-agent': { taskType: 'design', minutes: 50, roles: ALL_ROLES, levels: ['l5', 'l6'], areas: ['llm-systems', 'system-design'], rounds: ['llm-systems', 'ml-system-design'], domains: ['llm'], priority: 92 },
  'fine-tune-vs-prompt-vs-rag': { taskType: 'design', minutes: 45, roles: ALL_ROLES, levels: ['l5', 'l6'], areas: ['llm-systems'], rounds: ['llm-systems'], domains: ['llm'], priority: 95, prerequisites: ['rag-overview'] },
  'handle-hallucinations-in-production': { taskType: 'design', minutes: 45, roles: ['applied-scientist', 'ml-engineer'], levels: ['l5', 'l6'], areas: ['llm-systems', 'production'], rounds: ['llm-systems', 'production-debugging'], domains: ['llm'], priority: 90 },
  'reduce-llm-inference-cost-10x': { taskType: 'design', minutes: 50, roles: ['ml-engineer', 'research-engineer'], levels: ['l5', 'l6'], areas: ['llm-systems', 'production'], rounds: ['llm-systems', 'production-debugging'], domains: ['llm'], priority: 98, prerequisites: ['kv-cache', 'continuous-batching'] },
  'build-llm-coding-assistant': { taskType: 'design', minutes: 55, roles: ['ml-engineer', 'applied-scientist'], levels: ['l5', 'l6'], areas: ['llm-systems', 'system-design'], rounds: ['llm-systems', 'ml-system-design'], domains: ['llm'], priority: 87 },
  'llm-evals-the-hardest-part': { taskType: 'read', minutes: 30, roles: ALL_ROLES, levels: ['l5', 'l6'], areas: ['llm-systems', 'system-design'], domains: ['llm'], priority: 94 },
  'designing-rag-that-works': { taskType: 'read', minutes: 30, roles: ALL_ROLES, levels: ['l5', 'l6'], areas: ['llm-systems', 'system-design'], domains: ['llm'], priority: 90, prerequisites: ['rag-overview'] },
  'llm-inference-cost': { taskType: 'read', minutes: 30, roles: ['ml-engineer', 'research-engineer'], levels: ['l5', 'l6'], areas: ['llm-systems', 'production'], domains: ['llm'], priority: 88 },

  'design-youtube-recommender': { taskType: 'design', minutes: 55, roles: ['applied-scientist', 'ml-engineer'], levels: ['l5', 'l6'], areas: ['recsys-search', 'system-design'], rounds: ['recsys-search', 'ml-system-design'], domains: ['recsys-search'], priority: 96, prerequisites: ['two-tower-retrieval'] },
  'evaluate-search-ranker': { taskType: 'design', minutes: 45, roles: ['applied-scientist', 'ml-engineer'], levels: ['l5', 'l6'], areas: ['recsys-search', 'system-design'], rounds: ['recsys-search', 'ml-system-design'], domains: ['recsys-search'], priority: 90 },
  'two-tower-vs-cross-encoder': { taskType: 'design', minutes: 40, roles: ['applied-scientist', 'ml-engineer'], levels: ['l5', 'l6'], areas: ['recsys-search'], rounds: ['recsys-search'], domains: ['recsys-search'], priority: 88, prerequisites: ['two-tower-retrieval'] },
  'personalized-search-ranking': { taskType: 'design', minutes: 55, roles: ['applied-scientist', 'ml-engineer'], levels: ['l5', 'l6'], areas: ['recsys-search', 'system-design'], rounds: ['recsys-search', 'ml-system-design'], domains: ['recsys-search'], priority: 93 },

  'design-fraud-detection': { taskType: 'design', minutes: 55, roles: ['applied-scientist', 'ml-engineer'], levels: ['l5', 'l6'], areas: ['system-design'], rounds: ['ml-system-design'], priority: 94 },
  'design-feature-store': { taskType: 'design', minutes: 50, roles: ['ml-engineer', 'applied-scientist'], levels: ['l5', 'l6'], areas: ['system-design', 'production'], rounds: ['ml-system-design'], priority: 90 },
  'design-ml-monitoring': { taskType: 'design', minutes: 50, roles: ['ml-engineer', 'applied-scientist'], levels: ['l5', 'l6'], areas: ['system-design', 'production'], rounds: ['ml-system-design', 'production-debugging'], priority: 94 },

  'train-100b-model': { taskType: 'design', minutes: 55, roles: ['ml-engineer', 'research-engineer'], levels: ['l5', 'l6'], areas: ['production', 'system-design'], rounds: ['production-debugging', 'ml-system-design'], domains: ['deep-learning'], priority: 90, prerequisites: ['gpu-memory-hierarchy', 'fsdp-and-zero'] },
  'mixed-precision-deep': { taskType: 'practice', minutes: 40, roles: ['ml-engineer', 'research-engineer'], levels: ['l5', 'l6'], areas: ['production'], rounds: ['production-debugging'], domains: ['deep-learning'], priority: 84, prerequisites: ['mixed-precision-training'] },
  'gpu-memory-hierarchy': { taskType: 'read', minutes: 25, roles: ['ml-engineer', 'research-engineer'], levels: ['l5', 'l6'], areas: ['production'], domains: ['deep-learning', 'llm'], priority: 86 },
  'fsdp-and-zero': { taskType: 'read', minutes: 25, roles: ['ml-engineer', 'research-engineer'], levels: ['l5', 'l6'], areas: ['production'], domains: ['deep-learning'], priority: 84 },
  'kv-cache': { taskType: 'read', minutes: 20, roles: ['ml-engineer', 'research-engineer'], levels: ['l5', 'l6'], areas: ['llm-systems', 'production'], domains: ['llm'], priority: 86 },
  'continuous-batching': { taskType: 'read', minutes: 20, roles: ['ml-engineer', 'research-engineer'], levels: ['l5', 'l6'], areas: ['llm-systems', 'production'], domains: ['llm'], priority: 82 },
  'flashattention': { taskType: 'read', minutes: 25, roles: ['ml-engineer', 'research-engineer'], levels: ['l5', 'l6'], areas: ['llm-systems', 'production'], domains: ['llm'], priority: 82 },

  'derive-logistic-regression': { taskType: 'derive', minutes: 45, roles: ['applied-scientist', 'research-engineer'], levels: ALL_LEVELS, areas: ['math-research'], rounds: ['math-research'], priority: 84 },
  'reparameterization-trick': { taskType: 'derive', minutes: 40, roles: ['research-engineer', 'applied-scientist'], levels: ['l5', 'l6'], areas: ['math-research'], rounds: ['math-research'], priority: 82 },
  'expectation-maximization': { taskType: 'derive', minutes: 40, roles: ['research-engineer', 'applied-scientist'], levels: ['l5', 'l6'], areas: ['math-research'], rounds: ['math-research'], priority: 80 },
};

export function applyResourceOverride(resource: CatalogResource): CatalogResource {
  const override = RESOURCE_OVERRIDES[resource.slug];
  if (!override) return resource;
  return {
    ...resource,
    taskType: override.taskType ?? resource.taskType,
    estimatedMinutes: override.minutes ?? resource.estimatedMinutes,
    areas: override.areas ?? resource.areas,
    roles: override.roles ?? resource.roles,
    levels: override.levels ?? resource.levels,
    rounds: override.rounds ?? resource.rounds,
    domainTracks: override.domains ?? resource.domainTracks,
    priority: override.priority ?? resource.priority,
    prerequisites: override.prerequisites ?? resource.prerequisites,
  };
}
