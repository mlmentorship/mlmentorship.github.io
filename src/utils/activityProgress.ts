export const ACTIVITY_PROGRESS_KEY = 'mlm:activity-progress:v1';

export interface ActivityProgressState {
  version: 1;
  completed: Record<string, string>;
}

export function parseActivityProgress(value: unknown): ActivityProgressState {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return { version: 1, completed: {} };
  const candidate = value as Partial<ActivityProgressState>;
  const completed = candidate.completed && typeof candidate.completed === 'object' && !Array.isArray(candidate.completed) ? candidate.completed : {};
  return { version: 1, completed: Object.fromEntries(Object.entries(completed).filter(([, date]) => typeof date === 'string')) };
}

export function loadActivityProgress(): ActivityProgressState {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(ACTIVITY_PROGRESS_KEY) ?? 'null');
    return value && typeof value === 'object' && (value as Partial<ActivityProgressState>).version === 1 ? parseActivityProgress(value) : { version: 1, completed: {} };
  } catch {
    return { version: 1, completed: {} };
  }
}

export function saveActivityProgress(state: ActivityProgressState): void {
  localStorage.setItem(ACTIVITY_PROGRESS_KEY, JSON.stringify(state));
}

export function setActivityComplete(id: string, complete = true): boolean {
  const state = loadActivityProgress();
  if (complete) state.completed[id] = new Date().toISOString();
  else delete state.completed[id];
  saveActivityProgress(state);
  return complete;
}

export function isActivityComplete(id: string): boolean {
  return Boolean(loadActivityProgress().completed[id]);
}

export function clearActivityProgress(): void {
  localStorage.removeItem(ACTIVITY_PROGRESS_KEY);
}

export function replaceActivityProgress(value: unknown): number {
  const state = parseActivityProgress(value);
  saveActivityProgress(state);
  return Object.keys(state.completed).length;
}
