export const ROLES = ['applied-scientist', 'ml-engineer', 'research-engineer'] as const;
export const LEVELS = ['l4', 'l5', 'l6'] as const;
export const ROUNDS = [
  'ml-fundamentals',
  'production-debugging',
  'llm-systems',
  'recsys-search',
  'ml-system-design',
  'behavioral',
  'math-research',
  'coding',
] as const;
export const DOMAIN_TRACKS = ['llm', 'recsys-search', 'classical-ml', 'deep-learning', 'nlp-speech', 'computer-vision'] as const;
export const PLAN_AREAS = [
  'fundamentals',
  'production',
  'llm-systems',
  'recsys-search',
  'system-design',
  'behavioral',
  'math-research',
  'coding',
] as const;

export type Role = (typeof ROLES)[number];
export type Level = (typeof LEVELS)[number];
export type InterviewRound = (typeof ROUNDS)[number];
export type DomainTrack = (typeof DOMAIN_TRACKS)[number];
export type PlanArea = (typeof PLAN_AREAS)[number];
export type ResourceCategory = 'questions' | 'guides' | 'concepts';
export type TaskType = 'read' | 'practice' | 'design' | 'story' | 'derive' | 'review' | 'simulation';

export interface PlaybookIntake {
  version: 1;
  candidateName: string;
  role: Role;
  targetLevel: Level;
  startDate: string;
  weeks: number;
  hoursPerWeek: number;
  rounds: InterviewRound[];
  domainTracks: DomainTrack[];
  selfRatings: Record<PlanArea, number>;
  interviewDate?: string;
  experienceSummary?: string;
  constraints?: string[];
  priorities?: string[];
}

export interface CatalogResource {
  slug: string;
  title: string;
  description: string;
  category: ResourceCategory;
  subcategory: string;
  route: string;
  absoluteUrl: string;
  tags: string[];
  wordCount: number;
  readingMinutes: number;
  taskType: Exclude<TaskType, 'review' | 'simulation'>;
  estimatedMinutes: number;
  areas: PlanArea[];
  roles: Role[];
  levels: Level[];
  rounds: InterviewRound[];
  domainTracks: DomainTrack[];
  priority: number;
  prerequisites: string[];
}

export interface ReadinessArea {
  area: PlanArea;
  label: string;
  rating: number;
  weight: number;
  urgency: number;
  priority: 'critical' | 'high' | 'maintain';
  rationale: string;
}

export interface PlanTask {
  id: string;
  week: number;
  day: number;
  sequence: number;
  type: TaskType;
  title: string;
  area: PlanArea;
  minutes: number;
  route?: string;
  absoluteUrl?: string;
  resourceSlug?: string;
  why: string;
  instructions: string[];
  reviewOf?: string;
}

export interface PlanWeek {
  week: number;
  theme: string;
  objective: string;
  plannedMinutes: number;
  budgetMinutes: number;
  tasks: PlanTask[];
  exitCriteria: string[];
}

export interface PersonalizedPlaybook {
  schemaVersion: 1;
  engineVersion: string;
  planId: string;
  generatedFor: string;
  generatedOn: string;
  intake: PlaybookIntake;
  profile: {
    roleLabel: string;
    levelLabel: string;
    horizonLabel: string;
    totalBudgetHours: number;
    interviewDate?: string;
  };
  executiveSummary: {
    headline: string;
    strategy: string;
    topPriorities: string[];
    risks: string[];
    operatingRules: string[];
  };
  readiness: ReadinessArea[];
  weeks: PlanWeek[];
  practiceProtocol: {
    before: string[];
    during: string[];
    after: string[];
    scoringRubric: Array<{ dimension: string; strongSignal: string }>;
  };
  storyBank: Array<{ prompt: string; evidenceToPrepare: string[] }>;
  finalWeekChecklist: string[];
  resourceAppendix: CatalogResource[];
  totals: {
    scheduledMinutes: number;
    scheduledHours: number;
    uniqueResources: number;
    taskCounts: Record<TaskType, number>;
  };
  disclaimer: string;
}
