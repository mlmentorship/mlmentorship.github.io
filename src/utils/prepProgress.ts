export const PREP_PROGRESS_KEY = 'mlm:prep-progress:v1';
const PREP_STORAGE_PROBE_KEY = 'mlm:storage-probe';

export type PracticeScore = 'Weak' | 'Review' | 'Confident';

export interface PracticeProgressRecord {
  slug: string;
  title: string;
  mode: string;
  score: PracticeScore;
  weakDimensions: string[];
  dimensionMisses: Record<string, number>;
  attempts: number;
  successfulAttempts: number;
  lastSuccessfulOn: string | null;
  mixedVerifiedOn: string | null;
  lastAttemptOn: string;
  dueOn: string | null;
}

export function isPrepStorageAvailable(): boolean {
  try {
    localStorage.setItem(PREP_STORAGE_PROBE_KEY, '1');
    localStorage.removeItem(PREP_STORAGE_PROBE_KEY);
    return true;
  } catch {
    return false;
  }
}

function isRecord(value: unknown): value is PracticeProgressRecord {
  if (!value || typeof value !== 'object') return false;
  const candidate = value as Partial<PracticeProgressRecord>;
  return typeof candidate.slug === 'string'
    && typeof candidate.title === 'string'
    && typeof candidate.mode === 'string'
    && ['Weak', 'Review', 'Confident'].includes(candidate.score ?? '')
    && Array.isArray(candidate.weakDimensions)
    && typeof candidate.dimensionMisses === 'object'
    && typeof candidate.attempts === 'number'
    && typeof candidate.successfulAttempts === 'number'
    && (candidate.lastSuccessfulOn === null || typeof candidate.lastSuccessfulOn === 'string')
    && (candidate.mixedVerifiedOn === null || typeof candidate.mixedVerifiedOn === 'string')
    && typeof candidate.lastAttemptOn === 'string'
    && (candidate.dueOn === null || typeof candidate.dueOn === 'string');
}

export function loadPrepProgress(): PracticeProgressRecord[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(PREP_PROGRESS_KEY) ?? '[]');
    return Array.isArray(value) ? value.filter(isRecord) : [];
  } catch {
    return [];
  }
}

export function savePracticeProgress(input: {
  slug: string;
  title: string;
  mode: string;
  score: PracticeScore;
  weakDimensions: string[];
}): PracticeProgressRecord {
  const records = loadPrepProgress();
  const existing = records.find((record) => record.slug === input.slug);
  const now = new Date();
  const today = toLocalDate(now);
  const dimensionMisses = { ...(existing?.dimensionMisses ?? {}) };
  input.weakDimensions.forEach((dimension) => { dimensionMisses[dimension] = (dimensionMisses[dimension] ?? 0) + 1; });
  const spacedSuccess = input.score === 'Confident'
    && input.weakDimensions.length === 0
    && existing?.lastSuccessfulOn !== today;
  const successfulAttempts = input.score === 'Confident'
    ? (existing?.successfulAttempts ?? 0) + (spacedSuccess ? 1 : 0)
    : 0;
  const dueDays = input.score === 'Weak' ? 2 : 7;
  const due = input.score === 'Confident' && successfulAttempts >= 2
    ? null
    : new Date(now.getFullYear(), now.getMonth(), now.getDate() + dueDays);
  const record: PracticeProgressRecord = {
    ...input,
    weakDimensions: [...new Set(input.weakDimensions)],
    dimensionMisses,
    attempts: (existing?.attempts ?? 0) + 1,
    successfulAttempts,
    lastSuccessfulOn: spacedSuccess ? today : existing?.lastSuccessfulOn ?? null,
    mixedVerifiedOn: input.score === 'Confident' ? existing?.mixedVerifiedOn ?? null : null,
    lastAttemptOn: today,
    dueOn: due ? toLocalDate(due) : null,
  };
  const updated = [...records.filter((item) => item.slug !== input.slug), record];
  localStorage.setItem(PREP_PROGRESS_KEY, JSON.stringify(updated));
  return record;
}

export function markMixedSessionVerified(slug: string): PracticeProgressRecord | null {
  const records = loadPrepProgress();
  const existing = records.find((record) => record.slug === slug);
  if (!existing || existing.successfulAttempts < 2 || existing.weakDimensions.length > 0) return null;
  const record = { ...existing, mixedVerifiedOn: toLocalDate(new Date()) };
  localStorage.setItem(PREP_PROGRESS_KEY, JSON.stringify([
    ...records.filter((item) => item.slug !== slug),
    record,
  ]));
  return record;
}

export function clearPrepProgress(): void {
  localStorage.removeItem(PREP_PROGRESS_KEY);
}

export function replacePrepProgress(values: unknown[]): number {
  const records = values.filter(isRecord);
  localStorage.setItem(PREP_PROGRESS_KEY, JSON.stringify(records));
  return records.length;
}

export function toLocalDate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

export function compareLocalDates(left: string, right: string): number {
  return left.localeCompare(right);
}
