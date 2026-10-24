import { isPracticeProgressRecord, PREP_PROGRESS_KEY, type PracticeProgressRecord } from './prepProgress';
import { ACTIVITY_PROGRESS_KEY, parseActivityProgress } from './activityProgress';
import { buildPrepTracks, buildStudyRoutes } from './prepRoutes';
import { INTERVIEW_ROUNDS, PREP_DOMAINS, ROLE_ORDER } from '../data/prepCurriculum';

export const PREP_PLAN_KEY = 'mlm:prep-plan:v1';

export interface PrepPlanGap {
  areaId: string;
  label: string;
  description: string;
  link: string;
  starterLinks: Array<{ label: string; href: string }>;
  rating: number;
}

export interface PrepPlanState {
  version: 1;
  createdOn: string;
  updatedOn: string;
  role: string;
  roleLabel: string;
  level: string;
  domain: string;
  weeklyHours: number;
  availableWeeks: number;
  horizon: 2 | 4 | 8;
  estimatedHours: { low: number; high: number };
  selectedRounds: string[];
  areaRatings: Record<string, number>;
  topGaps: PrepPlanGap[];
  externalRounds: Array<{ label: string; status: 'missing' | 'workable' }>;
}

export interface PrepPlanTask {
  id: string;
  kind: 'retry' | 'new-attempt' | 'role-path' | 'simulation' | 'mixed';
  title: string;
  detail: string;
  href: string;
  dueOn?: string;
}

function isPlan(value: unknown): value is PrepPlanState {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const plan = value as Partial<PrepPlanState>;
  return plan.version === 1
    && typeof plan.createdOn === 'string'
    && typeof plan.updatedOn === 'string'
    && ROLE_ORDER.some(role => role === plan.role)
    && typeof plan.roleLabel === 'string'
    && ['l4', 'l5', 'l6', 'l7', 'l8'].includes(plan.level ?? '')
    && PREP_DOMAINS.some(domain => domain.id === plan.domain)
    && typeof plan.weeklyHours === 'number' && Number.isFinite(plan.weeklyHours) && plan.weeklyHours > 0
    && typeof plan.availableWeeks === 'number' && Number.isSafeInteger(plan.availableWeeks) && plan.availableWeeks > 0
    && [2, 4, 8].includes(plan.horizon ?? 0)
    && plan.estimatedHours !== null && typeof plan.estimatedHours === 'object'
    && Number.isFinite(plan.estimatedHours.low) && Number.isFinite(plan.estimatedHours.high)
    && plan.estimatedHours.low >= 0 && plan.estimatedHours.high >= plan.estimatedHours.low
    && Array.isArray(plan.selectedRounds)
    && plan.selectedRounds.every(round => INTERVIEW_ROUNDS.some(known => known.id === round))
    && plan.areaRatings !== null
    && typeof plan.areaRatings === 'object'
    && !Array.isArray(plan.areaRatings)
    && Object.values(plan.areaRatings).every(rating => typeof rating === 'number' && Number.isFinite(rating))
    && Array.isArray(plan.topGaps)
    && plan.topGaps.every(gap => gap !== null && typeof gap === 'object'
      && ['areaId', 'label', 'description', 'link'].every(key => typeof gap[key as keyof PrepPlanGap] === 'string')
      && Number.isFinite(gap.rating)
      && Array.isArray(gap.starterLinks)
      && gap.starterLinks.every(link => link !== null && typeof link === 'object' && typeof link.label === 'string' && typeof link.href === 'string'))
    && Array.isArray(plan.externalRounds)
    && plan.externalRounds.every(round => round !== null && typeof round === 'object' && typeof round.label === 'string' && ['missing', 'workable'].includes(round.status));
}

export function loadPrepPlan(): PrepPlanState | null {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(PREP_PLAN_KEY) ?? 'null');
    if (value && typeof value === 'object' && (value as Partial<PrepPlanState>).version === 1 && !(value as Partial<PrepPlanState>).areaRatings) {
      (value as Partial<PrepPlanState>).areaRatings = {};
    }
    return isPlan(value) ? value : null;
  } catch {
    return null;
  }
}

export function savePrepPlan(plan: PrepPlanState): void {
  localStorage.setItem(PREP_PLAN_KEY, JSON.stringify(plan));
}

export function clearPrepPlan(): void {
  localStorage.removeItem(PREP_PLAN_KEY);
}

export function currentPlanWeek(plan: PrepPlanState, now = new Date()): number {
  const start = new Date(`${plan.createdOn}T00:00:00`);
  if (Number.isNaN(start.getTime())) return 1;
  const elapsedDays = Math.max(0, Math.floor((now.getTime() - start.getTime()) / 86_400_000));
  return Math.min(plan.availableWeeks, Math.floor(elapsedDays / 7) + 1);
}

export function nextPrepTasks(plan: PrepPlanState, records: PracticeProgressRecord[], today: string, completed: Record<string, string> = {}): PrepPlanTask[] {
  const tasks: PrepPlanTask[] = [];
  const routes = buildStudyRoutes(plan, records, today);
  const seen = new Set<string>();
  const candidates = routes.flatMap(route => route.steps.map(step => ({ route, step })));
  candidates.sort((left, right) => {
    const rank = (status: string) => status === 'due' ? 0 : status === 'mixed' ? 1 : 2;
    return rank(left.step.status) - rank(right.step.status)
      || (left.step.status === 'due' && right.step.status === 'due' ? (left.step.record?.dueOn ?? '').localeCompare(right.step.record?.dueOn ?? '') : 0)
      || (left.step.stage === 'Diagnostic' ? 0 : 1) - (right.step.stage === 'Diagnostic' ? 0 : 1);
  });
  for (const { route, step } of candidates) {
    if (seen.has(step.slug) || step.status === 'scheduled' || step.status === 'mastered') continue;
    if (step.status === 'mixed' && step.record && step.record.lastAttemptOn >= today) continue;
    if (step.stage === 'Transfer' && step.status === 'new' && !route.steps[0].record?.successfulAttempts) continue;
    const kind = step.status === 'due' ? 'retry' : step.status === 'mixed' ? 'mixed' : 'new-attempt';
    const repair = step.record?.weakDimensions.map(dimension => dimension.replaceAll('-', ' ')).join(', ');
    tasks.push({
      id: `${kind}:${step.slug}`, kind,
      title: `${kind === 'retry' ? 'Retry' : kind === 'mixed' ? 'Mixed check' : step.stage}: ${step.label}`,
      detail: `${route.label} · ${step.practice.minutes} min + 15 min review. ${kind === 'mixed' ? 'Use an unfamiliar follow-up without notes; record the result below.' : repair ? `Repair: ${repair}.` : step.practice.instruction}`,
      href: kind === 'mixed' ? '#queue' : step.href,
      dueOn: step.record?.dueOn ?? undefined,
    });
    seen.add(step.slug);
  }
  const urgentQuestionTasks = tasks.filter(task => task.kind === 'retry' || task.kind === 'mixed');
  if (urgentQuestionTasks.length === 0) {
    const evidenceTasks = buildPrepTracks(plan, routes, completed)
      .filter(track => track.status === 'ready')
      .map(track => ({
        id: `track:${track.id}`,
        kind: track.kind === 'simulation' ? 'simulation' as const : 'role-path' as const,
        title: track.title,
        detail: `${track.minutes} min. ${track.detail}`,
        href: track.href,
      }));
    tasks.unshift(...evidenceTasks);
  }
  if (tasks.length === 0 && routes.length > 0 && routes.every(route => route.steps.every(step => step.status === 'mastered'))) {
    tasks.push({ id: 'final-week', kind: 'role-path', title: 'Review final-week logistics', detail: 'The mapped prompts have passed. Confirm your actual round formats and retain only remaining gaps.', href: '/prep/final-week/' });
  }
  return tasks.slice(0, 3);
}

export interface PrepBackup {
  version: 1;
  exportedOn: string;
  plan: PrepPlanState | null;
  records: PracticeProgressRecord[];
  activity?: unknown;
}

export function parsePrepBackup(value: unknown): PrepBackup | null {
  if (!value || typeof value !== 'object') return null;
  const backup = value as Partial<PrepBackup>;
  if (backup.version !== 1 || !Array.isArray(backup.records) || !backup.records.every(isPracticeProgressRecord)) return null;
  const plan = backup.plan && typeof backup.plan === 'object'
    ? { ...backup.plan, areaRatings: backup.plan.areaRatings ?? {} }
    : backup.plan;
  if (plan !== null && plan !== undefined && !isPlan(plan)) return null;
  return {
    version: 1,
    exportedOn: typeof backup.exportedOn === 'string' ? backup.exportedOn : new Date().toISOString(),
    plan: plan ?? null,
    records: backup.records,
    activity: backup.activity,
  };
}

export function restorePrepBackup(value: unknown, storage: Pick<Storage, 'getItem' | 'setItem' | 'removeItem'> = localStorage): void {
  const backup = parsePrepBackup(value);
  if (!backup) throw new TypeError('Invalid workbook backup');
  const replacement: Array<[string, string | null]> = [
    [PREP_PLAN_KEY, backup.plan ? JSON.stringify(backup.plan) : null],
    [PREP_PROGRESS_KEY, JSON.stringify(backup.records)],
    [ACTIVITY_PROGRESS_KEY, JSON.stringify(parseActivityProgress(backup.activity))],
  ];
  const previous = replacement.map(([key]) => [key, storage.getItem(key)] as const);
  let changed = false;
  try {
    for (const [key, stored] of replacement) {
      if (stored === null) storage.removeItem(key);
      else storage.setItem(key, stored);
      changed = true;
    }
  } catch (error) {
    if (changed) {
      for (const [key] of replacement) storage.removeItem(key);
      for (const [key, stored] of previous) if (stored !== null) storage.setItem(key, stored);
    }
    throw error;
  }
}
