import type { PracticeProgressRecord } from './prepProgress';

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
  kind: 'retry' | 'new-attempt' | 'role-path' | 'simulation';
  title: string;
  detail: string;
  href: string;
  dueOn?: string;
}

function isPlan(value: unknown): value is PrepPlanState {
  if (!value || typeof value !== 'object') return false;
  const plan = value as Partial<PrepPlanState>;
  return plan.version === 1
    && typeof plan.createdOn === 'string'
    && typeof plan.updatedOn === 'string'
    && typeof plan.role === 'string'
    && typeof plan.roleLabel === 'string'
    && typeof plan.level === 'string'
    && typeof plan.domain === 'string'
    && typeof plan.weeklyHours === 'number'
    && typeof plan.availableWeeks === 'number'
    && [2, 4, 8].includes(plan.horizon ?? 0)
    && Array.isArray(plan.selectedRounds)
    && plan.areaRatings !== null
    && typeof plan.areaRatings === 'object'
    && Array.isArray(plan.topGaps)
    && Array.isArray(plan.externalRounds);
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
  return Math.min(plan.horizon, Math.floor(elapsedDays / 7) + 1);
}

function slugFromQuestionHref(href: string): string | null {
  const match = href.match(/^\/questions\/([^/?#]+)\/?/);
  return match?.[1] ?? null;
}

export function nextPrepTasks(plan: PrepPlanState, records: PracticeProgressRecord[], today: string): PrepPlanTask[] {
  const tasks: PrepPlanTask[] = [];
  const usedHrefs = new Set<string>();
  const due = records
    .filter((record) => record.dueOn !== null && record.dueOn <= today)
    .sort((left, right) => (left.dueOn ?? '').localeCompare(right.dueOn ?? ''));

  for (const record of due) {
    const href = `/questions/${record.slug}/?practice=1`;
    tasks.push({
      id: `retry:${record.slug}`,
      kind: 'retry',
      title: `Retry: ${record.title}`,
      detail: `${record.score} on the last attempt${record.weakDimensions.length ? ` · repair ${record.weakDimensions[0].replaceAll('-', ' ')}` : ''}`,
      href,
      dueOn: record.dueOn ?? undefined,
    });
    usedHrefs.add(href.replace('?practice=1', ''));
  }

  if (tasks.length < 3 && (plan.level === 'l6' || plan.level === 'l7' || plan.level === 'l8')) {
    tasks.push({
      id: 'level:staff-principal',
      kind: 'role-path',
      title: 'Continue the upper-IC level path',
      detail: 'Practice architecture, problem selection, portfolio judgment, delegated authority, recovery, and succession.',
      href: '/prep/level-paths/staff-principal/',
    });
  }

  for (const gap of plan.topGaps) {
    for (const starter of gap.starterLinks) {
      const canonicalHref = starter.href.replace('?practice=1', '');
      const slug = slugFromQuestionHref(starter.href);
      const record = slug ? records.find((item) => item.slug === slug) : undefined;
      const graduated = record && record.successfulAttempts >= 2 && record.mixedVerifiedOn !== null;
      if (graduated || usedHrefs.has(canonicalHref)) continue;
      tasks.push({
        id: `gap:${gap.areaId}:${starter.href}`,
        kind: 'new-attempt',
        title: starter.label,
        detail: `${gap.label} · current evidence ${gap.rating}/5`,
        href: slug ? `${starter.href}${starter.href.includes('?') ? '&' : '?'}practice=1` : starter.href,
      });
      usedHrefs.add(canonicalHref);
      break;
    }
  }

  if (tasks.length < 3) {
    tasks.push({
      id: `path:${plan.role}`,
      kind: 'role-path',
      title: `Continue the ${plan.roleLabel} path`,
      detail: `Week ${currentPlanWeek(plan)} of ${plan.horizon} · ${plan.weeklyHours} hours per week`,
      href: `/prep/role-paths/${plan.role}/`,
    });
  }

  const attemptCount = records.reduce((total, record) => total + record.attempts, 0);
  if (tasks.length < 3 && attemptCount >= 3) {
    const simulationId = plan.level === 'l8'
      ? 'senior-principal'
      : plan.level === 'l6' || plan.level === 'l7'
        ? 'staff-principal'
        : plan.role;
    tasks.push({
      id: `simulation:${simulationId}`,
      kind: 'simulation',
      title: 'Run an unfamiliar mixed simulation',
      detail: 'Use realistic timing and keep the repair list to three items.',
      href: `/prep/simulations/#${simulationId}`,
    });
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
  if (backup.version !== 1 || !Array.isArray(backup.records)) return null;
  if (backup.plan && !(backup.plan as Partial<PrepPlanState>).areaRatings) {
    (backup.plan as Partial<PrepPlanState>).areaRatings = {};
  }
  if (backup.plan !== null && backup.plan !== undefined && !isPlan(backup.plan)) return null;
  return {
    version: 1,
    exportedOn: typeof backup.exportedOn === 'string' ? backup.exportedOn : new Date().toISOString(),
    plan: backup.plan ?? null,
    records: backup.records,
    activity: backup.activity,
  };
}
