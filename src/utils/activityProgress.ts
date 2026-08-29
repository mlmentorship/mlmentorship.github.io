export const ACTIVITY_PROGRESS_KEY = 'mlm:activity-progress:v1';

export interface ActivityProgressState {
  version: 1;
  completed: Record<string, string>;
}

export function loadActivityProgress(): ActivityProgressState {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(ACTIVITY_PROGRESS_KEY) ?? 'null');
    if (!value || typeof value !== 'object') return { version: 1, completed: {} };
    const state = value as Partial<ActivityProgressState>;
    if (state.version !== 1 || !state.completed || typeof state.completed !== 'object') return { version: 1, completed: {} };
    return { version: 1, completed: state.completed };
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
  if (!value || typeof value !== 'object') {
    saveActivityProgress({ version: 1, completed: {} });
    return 0;
  }
  const candidate = value as Partial<ActivityProgressState>;
  const completed = candidate.completed && typeof candidate.completed === 'object' ? candidate.completed : {};
  const safeCompleted = Object.fromEntries(Object.entries(completed).filter(([id, date]) => typeof id === 'string' && typeof date === 'string'));
  saveActivityProgress({ version: 1, completed: safeCompleted });
  return Object.keys(safeCompleted).length;
}
