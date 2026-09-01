/**
 * @typedef {'pending' | 'reviewed'} ReviewStatus
 * @typedef {{key: string, kind: string, x: number, y: number, label: string}} MotionEntity
 * @typedef {{label: string, note: string, key?: string, scene: Record<string, unknown>}} VisualFrame
 * @typedef {{objective: string, frames: VisualFrame[]}} VisualDraft
 * @typedef {{
 *   pattern: string,
 *   recognitionCue: string,
 *   invariant: string,
 *   stateModel: string,
 *   visualRationale: string,
 *   rejectedAlternatives: string[],
 *   transferLesson: string,
 *   reviewStatus: ReviewStatus
 * }} VisualReview
 * @typedef {{slug: string, objective: string, frames: VisualFrame[], review: VisualReview}} VisualDefinition
 */

function text(value) {
  return String(value ?? '');
}

function motionEntities(scene) {
  /** @type {MotionEntity[]} */
  const entities = [];
  const add = (key, kind, x, y, label) => {
    if (!entities.some((entity) => entity.key === key)) entities.push({ key, kind, x, y, label: text(label) || 'empty' });
  };

  const linear = scene.items ?? scene.values ?? [];
  if (Array.isArray(linear)) {
    linear.forEach((value, index) => add(`value-${index}`, 'value', index, 0, value));
  }
  for (const item of scene.marks ?? []) {
    const x = Number.isInteger(item.col) ? item.col : item.index ?? 0;
    const y = Number.isInteger(item.row) ? item.row : 0;
    add(item.key ?? `marker-${text(item.label).toLowerCase().replace(/[^a-z0-9]+/g, '-')}`, 'pointer', x, y, item.label);
  }
  for (const [index, node] of (scene.nodes ?? []).entries()) {
    const value = typeof node === 'object' ? node.value ?? node.id : node;
    add(typeof node === 'object' && node.key ? node.key : `node-${text(value)}-${index}`, 'node', index, 0, value);
  }
  for (const [index, edge] of (scene.edges ?? []).entries()) add(`edge-${text(edge)}-${index}`, 'link', index, 1, edge);
  const treeValueCounts = new Map();
  for (const [levelIndex, level] of (scene.levels ?? []).entries()) {
    level.forEach((value, index) => {
      const count = treeValueCounts.get(value) ?? 0;
      treeValueCounts.set(value, count + 1);
      add(`tree-node-${text(value)}-${count}`, 'node', index, levelIndex, value);
    });
  }
  for (const [index, item] of (scene.paths ?? []).entries()) add(`path-${text(item.word)}-${index}`, 'path', index, 0, item.prefix);
  for (const [index, item] of (scene.order ?? []).entries()) add(`order-${text(item)}`, 'node', index, 0, item);
  for (const [index, item] of (scene.queue ?? []).entries()) add(`frontier-${text(item)}`, 'frontier', index, 0, item);
  if (entities.length === 0) add(`state-${scene.type}`, 'state', 0, 0, scene.type);
  return entities;
}

function withMotion(scene) {
  return { ...scene, motion: scene.motion ?? motionEntities(scene) };
}

export const frame = (label, note, scene, key) => ({ label, note, scene: withMotion(scene), ...(key ? { key } : {}) });
export const visual = (objective, frames) => ({ objective, frames });
export const mark = (index, label, tone = 'focus', key) => ({ index, label, tone, ...(key ? { key } : {}) });
export const array = (items, marks = [], extra = {}) => ({ type: 'array', items, marks, ...extra });
export const arrayMap = (items, map, marks = [], extra = {}) => ({ type: 'array-map', items, map, marks, ...extra });
export const table = (columns, rows, active = [], extra = {}) => ({ type: 'table', columns, rows, active, ...extra });
export const grid = (rows, marks = [], extra = {}) => ({ type: 'grid', rows, marks, ...extra });
export const stack = (input, values, extra = {}) => ({ type: 'stack', input, values, ...extra });
export const queueGrid = (rows, queue, extra = {}) => ({ type: 'queue-grid', rows, queue, ...extra });
export const graph = (nodes, edges, extra = {}) => ({ type: 'graph', nodes, edges, ...extra });
export const tree = (levels, marks = [], extra = {}) => ({ type: 'tree', levels, marks, ...extra });
export const intervals = (items, extra = {}) => ({ type: 'intervals', items, ...extra });
export const linked = (nodes, extra = {}) => ({ type: 'linked', nodes, ...extra });
export const trie = (paths, extra = {}) => ({ type: 'trie', paths, ...extra });
export const bits = (values, marks = [], extra = {}) => ({ type: 'bits', values, marks, ...extra });
export const bars = (values, extra = {}) => ({ type: 'bars', values, ...extra });
export const shapes = (items, extra = {}) => ({ type: 'shapes', items, ...extra });
export const attention = (rows, extra = {}) => ({ type: 'attention', rows, ...extra });
export const buckets = (items, extra = {}) => ({ type: 'buckets', items, ...extra });
export const prefix = (items, extra = {}) => ({ type: 'prefix', items, ...extra });
export const dualWindow = (items, extra = {}) => ({ type: 'dual-window', items, ...extra });
export const choices = (path, branches, extra = {}) => ({ type: 'choices', path, branches, ...extra });
export const lru = (map, order, extra = {}) => ({ type: 'lru', map, order, ...extra });
export const heap = (values, extra = {}) => ({ type: 'heap', values, ...extra });

/** @returns {VisualReview} */
export function pendingReview(objective) {
  return {
    pattern: 'Migrated problem-specific trace',
    recognitionCue: `Use this state model when the task asks you to ${objective.toLowerCase()}`,
    invariant: objective,
    stateModel: 'The visible authored frames preserve the algorithm state before and after each transition.',
    visualRationale: 'The selected primitive exposes the values, topology, and state changes used by the algorithm.',
    rejectedAlternatives: ['Prose-only explanation', 'Generic decorative diagram'],
    transferLesson: `Transfer the invariant: ${objective}`,
    reviewStatus: 'pending',
  };
}

/** @returns {VisualDefinition} */
export function defineVisual(slug, draft, review) {
  return Object.freeze({
    slug,
    objective: draft.objective,
    frames: draft.frames.map((item, index) => Object.freeze({
      ...item,
      key: item.key ?? `frame-${index + 1}`,
      scene: Object.freeze(withMotion(item.scene)),
    })),
    review,
  });
}
