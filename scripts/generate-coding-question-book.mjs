import fs from 'node:fs';
import path from 'node:path';
import { codingQuestionVisuals } from './coding-question-visuals.mjs';

const root = process.cwd();
const argumentsList = process.argv.slice(2);
const slugsArgumentIndex = argumentsList.findIndex((argument) => argument === '--slugs');
const slugsArgument = argumentsList.find((argument) => argument.startsWith('--slugs='))
  ?.slice('--slugs='.length) ?? (slugsArgumentIndex >= 0 ? argumentsList[slugsArgumentIndex + 1] : '');
const requestedSlugs = new Set(slugsArgument.split(',').map((slug) => slug.trim()).filter(Boolean));
const sourceArgument = argumentsList.find((argument, index) =>
  !argument.startsWith('--') && index !== slugsArgumentIndex + 1);
const sourcePath = sourceArgument || path.resolve(root, '../ml_interview_book/docs/dsa/Coding_Questions_Phone_Guide.md');
const postsDir = path.join(root, 'src/content/posts');
const auditsDir = path.join(root, 'data/visual-audits');
const publicationDate = '2026-09-01';

const chapterDefinitions = [
  { id: 'remember-the-past', title: 'Remember the past', description: 'Use maps, sets, and saved boundary values to turn repeated searching into one pass.', difficulty: 'Foundation', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['1', '2', '3', '4', '5', '6'] },
  { id: 'move-boundaries', title: 'Move boundaries', description: 'Use sorted order, windows, and answer-space search to discard impossible regions safely.', difficulty: 'Foundation', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['7', '8', '9', '10', '11', '12', '13', '14', '15', '16'] },
  { id: 'unfinished-work', title: 'Keep unfinished work', description: 'Use stacks and monotonic state when the newest unresolved item must be handled first.', difficulty: 'Foundation', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['17', '18', '19', '20'] },
  { id: 'next-best-item', title: 'Process the next best item', description: 'Let queues, heaps, and shortest-path frontiers decide which reachable item comes next.', difficulty: 'Intermediate', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['21', '22', '23', '24', '25'] },
  { id: 'explore-choices', title: 'Explore choices', description: 'Traverse graphs, trees, choice paths, and smaller dynamic-programming states without losing the invariant.', difficulty: 'Intermediate', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42'] },
  { id: 'useful-order', title: 'Create a useful order', description: 'Sort ranges, commit safe greedy choices, remove prerequisites, and join connected groups.', difficulty: 'Intermediate', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['43', '44', '45', '46', '47', '48', '49', '50'] },
  { id: 'change-links', title: 'Change links', description: 'Rewire linked lists and prefix trees while preserving the pointer or path you still need.', difficulty: 'Intermediate', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['51', '52', '53', '54', '55', '56'] },
  { id: 'core-coverage', title: 'Complete core coverage', description: 'Reuse the main patterns across bits, strings, matrices, trees, graphs, intervals, and caches.', difficulty: 'Mixed', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88'] },
  { id: 'practical-ai-coding', title: 'Practical AI coding', description: 'Make array shapes, masks, numerical stability, batching, selection, and metrics visible before coding.', difficulty: 'Intermediate', priority: 'Role-specific', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation', 'ML breadth'], numbers: ['AI1', 'AI2', 'AI3', 'AI4', 'AI5', 'AI6', 'AI7', 'AI8'] },
  { id: 'hard-problems', title: 'Hard problems', description: 'Combine boundaries, stacks, trees, grids, tries, heaps, and answer search after the core patterns feel natural.', difficulty: 'Advanced', priority: 'Specialist', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'Work sample'], numbers: ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10'] },
];

const allowedTones = new Set(['focus', 'state', 'output', 'warning', 'neutral']);

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function renderBarsScene(scene) {
  const values = scene.values.map(Number);
  const max = Math.max(...values.map(Math.abs), 1);
  const bars = values.map((value, index) => {
    const scan = scene.scan === index ? `<em class="coding-trace-bar-scan"${motionKey('scan-index')}>scan</em>` : '';
    const sentinel = scene.sentinel === index ? ' is-sentinel' : '';
    return `<span class="coding-trace-bar${sentinel}" style="--bar-height:${Math.max(4, Math.abs(value) / max * 100)}%"${motionKey(`value-${index}`)}><b>${scan}${escapeHtml(scene.labels?.[index] ?? value)}</b><i></i><small>${index}</small></span>`;
  }).join('');
  const area = scene.area ? (() => {
    const spansBars = scene.area.mode === 'bars';
    const left = (spansBars ? scene.area.left : scene.area.left + 0.5) / values.length * 100;
    const width = (spansBars ? scene.area.right - scene.area.left + 1 : scene.area.right - scene.area.left) / values.length * 100;
    const height = Math.abs(scene.area.height) / max * 100;
    return `<div class="coding-trace-measured-area" style="--area-left:${left}%;--area-width:${width}%;--area-height:${height}%"><span>${escapeHtml(scene.area.label)}</span></div>`;
  })() : '';
  const stack = Array.isArray(scene.stack) ? `<div class="coding-trace-bar-stack"><span class="coding-trace-label">unresolved stack</span>${scene.stack.map((entry, index) => { const start = String(entry).match(/\d+/)?.[0] ?? index; return `<span class="coding-trace-bar-stack-entry"${motionKey(`stack-entry-${start}`)}>${escapeHtml(entry)}</span>`; }).join('') || '<span class="coding-trace-empty">empty</span>'}<span class="coding-trace-label">top &rarr;</span></div>` : '';
  return `<div class="coding-trace-bars-wrap"><div class="coding-trace-bars" role="img" aria-label="Vertical values and measured area">${area}${bars}</div>${stack}</div>${renderMeta(scene, ['type', 'values', 'labels', 'area', 'scan', 'sentinel', 'stack', 'motion'])}`;
}

function slugify(value) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

function collapse(value) {
  return value.replace(/\s+/g, ' ').trim();
}

function extractField(section, label) {
  const expression = new RegExp(`\\*\\*${label}:\\*\\*\\s*([\\s\\S]*?)(?=\\n\\n(?:\\*\\*|<a id=)|$)`);
  return collapse(section.match(expression)?.[1] || '');
}

function wrapParagraphs(text) {
  const lines = text.split('\n');
  const output = [];
  let inFence = false;
  for (const line of lines) {
    if (line.startsWith('```')) {
      inFence = !inFence;
      output.push(line);
      continue;
    }
    if (inFence || !line.trim() || /^\s*(?:[#*<`]|\||[-*+] |\d+\. )/.test(line)) {
      output.push(line);
      continue;
    }
    const words = line.trim().split(/\s+/);
    let current = '';
    for (const word of words) {
      if (current && `${current} ${word}`.length > 88) {
        output.push(current);
        current = word;
      } else {
        current = current ? `${current} ${word}` : word;
      }
    }
    if (current) output.push(current);
  }
  return output.join('\n');
}

function toneClass(tone) {
  return allowedTones.has(tone) && tone !== 'neutral' ? ` trace-tone-${tone}` : '';
}

function motionKey(key) {
  return ` data-motion-key="${escapeHtml(key)}"`;
}

function renderMeta(scene, excluded) {
  const items = Object.entries(scene)
    .filter(([key, value]) => !excluded.includes(key) && value !== undefined && value !== null && (typeof value === 'string' || typeof value === 'number'))
    .map(([key, value]) => `<span><b>${escapeHtml(key.replaceAll('-', ' '))}</b>${escapeHtml(value)}</span>`)
    .join('');
  return items ? `<div class="coding-trace-meta">${items}</div>` : '';
}

function renderArrayScene(scene) {
  const marks = new Map();
  for (const item of scene.marks ?? []) marks.set(item.index, [...(marks.get(item.index) ?? []), item]);
  const cells = scene.items.map((item, index) => {
    const cellMarks = marks.get(index) ?? [];
    const labels = cellMarks.map((item) => item.label).join(' · ');
    const classes = cellMarks.map((item) => toneClass(item.tone)).join('');
    const display = item === '' ? '""' : item;
    const pointers = cellMarks.map((mark) => `<span class="coding-trace-array-pointer"${motionKey(mark.key ?? `marker-${mark.label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`)}>${escapeHtml(mark.label)}</span>`).join('');
    return `<span class="coding-trace-array-item${classes}" role="listitem"${motionKey(`value-${index}`)}>${pointers || `<span class="coding-trace-array-mark">${escapeHtml(labels)}</span>`}<span class="coding-trace-array-cell">${escapeHtml(display)}</span><small class="coding-trace-array-index">${index}</small></span>`;
  }).join('');
  return `<div class="coding-trace-array" role="list" aria-label="Array state">${cells}</div>${renderMeta(scene, ['type', 'items', 'marks'])}`;
}

function renderArrayMapScene(scene) {
  const map = (scene.map ?? []).map(([key, value]) => `<span class="coding-trace-map-entry"><b>${escapeHtml(key)}</b><span>${escapeHtml(value)}</span></span>`).join('');
  return `<div class="coding-trace-array-map">${renderArrayScene(scene)}<div class="coding-trace-map"><span class="coding-trace-label">${escapeHtml(scene.mapLabel ?? 'saved state')}</span>${map || '<span class="coding-trace-empty">empty</span>'}</div></div>`;
}

function renderTableScene(scene) {
  const active = new Set(scene.active ?? []);
  const head = scene.columns.map((column) => `<th scope="col">${escapeHtml(column)}</th>`).join('');
  const rows = scene.rows.map((row, rowIndex) => `<tr>${row.map((value, columnIndex) => `<td class="${active.has(rowIndex * row.length + columnIndex) ? 'is-active' : ''}">${escapeHtml(value)}</td>`).join('')}</tr>`).join('');
  return `<div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>${renderMeta(scene, ['type', 'columns', 'rows', 'active'])}`;
}

function renderGridCells(scene) {
  const marks = new Map((scene.marks ?? []).map((item) => [`${item.row}:${item.col}`, item]));
  return scene.rows.map((row, rowIndex) => row.map((value, colIndex) => {
    const mark = marks.get(`${rowIndex}:${colIndex}`);
    const key = mark?.key ?? (mark?.label ? `marker-${mark.label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}` : `grid-${rowIndex}-${colIndex}`);
    return `<span class="coding-trace-grid-cell${toneClass(mark?.tone)}"${motionKey(key)}><span>${escapeHtml(value)}</span>${mark?.label ? `<small>${escapeHtml(mark.label)}</small>` : ''}</span>`;
  }).join('')).join('');
}

function renderGridScene(scene) {
  return `<div class="coding-trace-grid" style="--trace-cols:${Math.max(1, scene.rows[0]?.length ?? 1)}" role="group" aria-label="Grid state">${renderGridCells(scene)}</div>${renderMeta(scene, ['type', 'rows', 'marks'])}`;
}

function renderQueueGridScene(scene) {
  const rows = scene.rows.map((row, rowIndex) => `<div class="coding-trace-queue-grid-row"><span class="coding-trace-label">${rowIndex}</span>${row.map((value) => `<span class="coding-trace-grid-cell">${escapeHtml(value)}</span>`).join('')}</div>`).join('');
  const queue = (scene.queue ?? []).map((item) => `<span class="coding-trace-queue-item">${escapeHtml(item)}</span>`).join('');
  return `<div class="coding-trace-queue-grid">${rows}</div><div class="coding-trace-queue"><span class="coding-trace-label">queue</span>${queue || '<span class="coding-trace-empty">empty</span>'}</div>`;
}

function renderStackScene(scene) {
  const values = scene.values.map((value, index) => `<span class="coding-trace-stack-item${index === scene.values.length - 1 ? ' is-top' : ''}">${escapeHtml(value)}</span>`).join('');
  return `<div class="coding-trace-stack-layout"><div class="coding-trace-stack-input"><span class="coding-trace-label">input</span><strong>${escapeHtml(scene.input)}</strong></div><div class="coding-trace-stack-column"><span class="coding-trace-label">top</span>${values || '<span class="coding-trace-empty">empty</span>'}</div></div>${renderMeta(scene, ['type', 'input', 'values'])}`;
}

let graphTopologyId = 0;

function renderGraphScene(scene) {
  const width = 480;
  const height = 230;
  const positions = scene.nodes.map((node, index) => ({
    node,
    x: scene.positions?.[node]?.x ?? width / 2 + Math.cos(-Math.PI / 2 + index * Math.PI * 2 / scene.nodes.length) * Math.min(170, 45 * scene.nodes.length),
    y: scene.positions?.[node]?.y ?? height / 2 + Math.sin(-Math.PI / 2 + index * Math.PI * 2 / scene.nodes.length) * 78,
  }));
  const markerId = `coding-graph-arrow-${graphTopologyId++}`;
  const endpoint = (edge, first) => {
    const tokens = String(edge).match(/[A-Za-z]*\s*\d+|[A-Za-z]+/g) ?? [];
    const token = tokens[first ? 0 : tokens.length - 1]?.trim();
    return positions.find((item) => item.node === token || String(item.node).split(' ').at(-1) === token)?.node;
  };
  const edges = scene.edges.map((edge, index) => {
    const from = positions.find((item) => item.node === endpoint(edge, true)) ?? positions[index % positions.length];
    const to = positions.find((item) => item.node === endpoint(edge, false)) ?? positions[(index + 1) % positions.length];
    const deltaX = to.x - from.x;
    const deltaY = to.y - from.y;
    const length = Math.hypot(deltaX, deltaY) || 1;
    const directed = String(edge).includes('->') && !String(edge).includes('<->');
    const label = scene.edgeLabels === false ? '' : scene.edgeLabelMode === 'weight' ? String(edge).match(/-(\d+)->/)?.[1] ?? edge : edge;
    return `<g${motionKey(`edge-${edge}-${index}`)}><line class="coding-trace-edge-line" x1="${from.x + deltaX / length * 26}" y1="${from.y + deltaY / length * 26}" x2="${to.x - deltaX / length * 29}" y2="${to.y - deltaY / length * 29}"${directed ? ` marker-end="url(#${markerId})"` : ''} />${label ? `<text class="coding-trace-graph-edge-label" x="${(from.x + to.x) / 2}" y="${(from.y + to.y) / 2 - 7}">${escapeHtml(label)}</text>` : ''}</g>`;
  }).join('');
  const nodes = positions.map(({ node, x, y }, index) => {
    const state = scene.start === node ? ' is-focus' : scene.visited?.some((item) => String(item).startsWith(`${node}:`) || item === node) ? ' is-state' : '';
    return `<g class="coding-trace-graph-node${state}"${motionKey(`node-${node}`)}><circle cx="${x}" cy="${y}" r="23" /><text x="${x}" y="${y + 4}">${escapeHtml(node)}</text></g>`;
  }).join('');
  const regions = (scene.regions ?? []).map((region) => `<text class="coding-trace-graph-region" x="${region.x}" y="${region.y}">${escapeHtml(region.label)}</text>`).join('');
  const keys = ['visited', 'frontier', 'ready', 'order', 'roots', 'components', 'indegree'];
  const lists = keys.flatMap((key) => scene[key] ? [`<span><b>${key}</b>${escapeHtml(Array.isArray(scene[key]) ? scene[key].join(', ') : scene[key])}</span>`] : []).join('');
  return `<div class="coding-trace-graph"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Connected graph topology"><defs><marker id="${markerId}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-graph-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs>${regions}${edges}${nodes}</svg>${lists ? `<div class="coding-trace-meta">${lists}</div>` : ''}</div>${renderMeta(scene, ['type', 'nodes', 'edges', 'positions', 'regions', 'edgeLabels', 'edgeLabelMode', 'start', ...keys, 'motion'])}`;
}

function renderTreeScene(scene) {
  const width = 640;
  const rowHeight = 72;
  let nodeIndex = 0;
  const positions = scene.levels.flatMap((level, levelIndex) => level.map((node, index) => ({
    node,
    levelIndex,
    index,
    flatIndex: nodeIndex++,
    x: (index + 0.5) * width / level.length,
    y: 28 + levelIndex * rowHeight,
  })));
  const edges = positions.filter((item) => item.levelIndex > 0 && item.node !== '-').map((item) => {
    const parent = positions.find((candidate) => candidate.levelIndex === item.levelIndex - 1 && candidate.index === Math.floor(item.index / 2));
    return parent && parent.node !== '-' ? `<line class="coding-trace-edge-line" x1="${parent.x}" y1="${parent.y}" x2="${item.x}" y2="${item.y}" />` : '';
  }).join('');
  const nodes = positions.filter((item) => item.node !== '-').map((item) => {
    const mark = scene.marks?.find((candidate) => candidate.index === item.flatIndex);
    const label = mark?.label ? `<text class="coding-trace-node-state" x="${item.x}" y="${item.y + 30}">${escapeHtml(mark.label)}</text>` : '';
    const occurrence = positions.slice(0, item.flatIndex).filter((candidate) => candidate.node === item.node).length;
    return `<g class="coding-trace-tree-node${toneClass(mark?.tone)}"${motionKey(`tree-node-${item.node}-${occurrence}`)}><circle cx="${item.x}" cy="${item.y}" r="18" /><text x="${item.x}" y="${item.y + 4}">${escapeHtml(item.node)}</text>${label}</g>`;
  }).join('');
  return `<div class="coding-trace-tree"><svg viewBox="0 0 ${width} ${scene.levels.length * rowHeight}" role="img" aria-label="Binary tree with parent-child edges and call state">${edges}${nodes}</svg></div>${renderMeta(scene, ['type', 'levels', 'marks', 'motion'])}`;
}

function renderIntervalsScene(scene) {
  const max = Number(scene.max ?? Math.max(...scene.items.map((item) => item.end), 1));
  const rows = scene.items.map((item) => {
    const left = Math.max(0, item.start / max * 100);
    const width = Math.max(2, (item.end - item.start) / max * 100);
    return `<div class="coding-trace-interval-row"><span>${escapeHtml(item.label)}</span><div class="coding-trace-track"><i class="${toneClass(item.tone).trim()}" style="--trace-start:${left}%;--trace-width:${width}%"></i></div></div>`;
  }).join('');
  return `<div class="coding-trace-intervals">${rows}</div>${renderMeta(scene, ['type', 'items', 'max'])}`;
}

function renderLinkedRow(nodes, label = '') {
  const rendered = nodes.map((node, index) => `${index > 0 ? `<span class="coding-trace-link-arrow"${motionKey(`link-${nodes[index - 1].value}-${node.value}`)}>&rarr;</span>` : ''}<span class="coding-trace-linked-node${toneClass(node.tone)}"${motionKey(node.key ?? `node-${node.value}`)}><span>${escapeHtml(node.value)}</span>${node.pointer ? `<small${motionKey(`pointer-${node.pointer}`)}>${escapeHtml(node.pointer)}</small>` : ''}</span>`).join('');
  return `<div class="coding-trace-linked-row">${label ? `<span class="coding-trace-label">${escapeHtml(label)}</span>` : ''}${rendered}</div>`;
}

let linkedTopologyId = 0;

function renderLinkedTopology(scene) {
  const width = Number(scene.width ?? 480);
  const height = Number(scene.height ?? 210);
  const markerId = `coding-linked-arrow-${linkedTopologyId++}`;
  const warningMarkerId = `${markerId}-warning`;
  const nodes = new Map(scene.nodes.map((node) => [node.key, node]));
  const rows = (scene.rowLabels ?? []).map((row) => `<text class="coding-trace-topology-row-label" x="8" y="${row.y}">${escapeHtml(row.label)}</text>`).join('');
  const edges = scene.edges.map((edge) => {
    const from = nodes.get(edge.from);
    const to = nodes.get(edge.to);
    const path = edge.curve
      ? `M ${from.x} ${from.y + 18} C ${from.x} ${from.y + edge.curve} ${to.x} ${to.y + edge.curve} ${to.x} ${to.y + 22}`
      : `M ${from.x + Math.sign(to.x - from.x) * 28} ${from.y} L ${to.x - Math.sign(to.x - from.x) * 31} ${to.y}`;
    const label = edge.label ? `<text class="coding-trace-topology-edge-label" x="${edge.labelX ?? (from.x + to.x) / 2}" y="${edge.labelY ?? (from.y + to.y) / 2 - 8}">${escapeHtml(edge.label)}</text>` : '';
    return `<g class="coding-trace-topology-edge${toneClass(edge.tone)}"${motionKey(edge.key)}><path d="${path}" marker-end="url(#${edge.tone === 'warning' ? warningMarkerId : markerId})" />${label}</g>`;
  }).join('');
  const renderedNodes = scene.nodes.map((node) => {
    const pointers = (Array.isArray(node.pointer) ? node.pointer : [node.pointer]).filter(Boolean);
    const pointerLabels = pointers.map((pointer, index) => `<text class="coding-trace-topology-pointer" x="${node.x}" y="${node.y - 25 - index * 13}"${motionKey(`pointer-${slugify(pointer)}`)}>${escapeHtml(pointer)}</text>`).join('');
    return `<g class="coding-trace-topology-node${toneClass(node.tone)}"${motionKey(node.key)}>${pointerLabels}<rect x="${node.x - 25}" y="${node.y - 17}" width="50" height="34" rx="3" /><text x="${node.x}" y="${node.y + 5}">${escapeHtml(node.value)}</text></g>`;
  }).join('');
  return `<div class="coding-trace-linked-topology"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Linked nodes and next-pointer edges"><title>Linked-list topology</title><desc>Nodes, next pointers, and moving algorithm pointers.</desc><defs><marker id="${markerId}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-topology-marker" d="M 0 0 L 8 4 L 0 8 z" /></marker><marker id="${warningMarkerId}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="coding-trace-topology-marker is-warning" d="M 0 0 L 8 4 L 0 8 z" /></marker></defs>${rows}${edges}${renderedNodes}</svg></div>${renderMeta(scene, ['type', 'nodes', 'edges', 'rowLabels', 'width', 'height', 'motion'])}`;
}

function renderLinkedScene(scene) {
  if (scene.edges) return renderLinkedTopology(scene);
  return `<div class="coding-trace-linked">${renderLinkedRow(scene.nodes)}${scene.second ? renderLinkedRow(scene.second.map((value) => ({ value })), 'second') : ''}${scene.arrows ? `<p class="coding-trace-inline-note">${escapeHtml(scene.arrows.join(' · '))}</p>` : ''}</div>${renderMeta(scene, ['type', 'nodes', 'second', 'arrows'])}`;
}

function renderTrieTopology(scene) {
  const nodes = new Map(scene.nodes.map((node) => [node.key, node]));
  const active = new Set(scene.active ?? []);
  const queued = new Set(scene.queued ?? []);
  const edges = scene.edges.map((edge) => {
    const from = nodes.get(edge.from);
    const to = nodes.get(edge.to);
    return `<g${motionKey(edge.key)}><line class="coding-trace-edge-line" x1="${from.x}" y1="${from.y + 16}" x2="${to.x}" y2="${to.y - 16}" /><text class="coding-trace-trie-edge-label" x="${(from.x + to.x) / 2 + 14}" y="${(from.y + to.y) / 2 + 4}">${escapeHtml(edge.label)}</text></g>`;
  }).join('');
  const renderedNodes = scene.nodes.map((node) => {
    const cursor = active.has(node.key) ? `<text class="coding-trace-trie-cursor" x="${node.x}" y="${node.y - 23}"${motionKey('trie-cursor')}>current</text>` : '';
    const classes = `${active.has(node.key) ? ' is-focus' : ''}${queued.has(node.key) ? ' is-state' : ''}${node.terminal ? ' is-terminal' : ''}`;
    return `<g class="coding-trace-trie-node${classes}"${motionKey(node.key)}>${cursor}<rect x="${node.x - 33}" y="${node.y - 16}" width="66" height="32" rx="3" /><text x="${node.x}" y="${node.y + 5}">${escapeHtml(node.label)}</text></g>`;
  }).join('');
  return `<div class="coding-trace-trie-topology"><svg viewBox="0 0 ${scene.width ?? 480} ${scene.height ?? 260}" role="img" aria-label="Shared-prefix trie topology"><title>Shared-prefix trie</title><desc>One node per shared prefix; double borders mark complete words.</desc>${edges}${renderedNodes}</svg><p class="coding-trace-trie-key"><span>double border</span> complete word${queued.size ? '<span>muted fill</span> queued wildcard branch' : ''}</p></div>${renderMeta(scene, ['type', 'paths', 'nodes', 'edges', 'active', 'queued', 'width', 'height', 'motion'])}`;
}

function renderTrieScene(scene) {
  if (scene.nodes) return renderTrieTopology(scene);
  const width = 560;
  const paths = scene.paths.map((item, index) => {
    const letters = [...item.word];
    const rowY = 30 + index * 58;
    const edges = letters.slice(1).map((_, letterIndex) => `<line class="coding-trace-edge-line" x1="${55 + letterIndex * 55}" y1="${rowY}" x2="${110 + letterIndex * 55}" y2="${rowY}" />`).join('');
    const nodes = letters.map((letter, letterIndex) => `<g${motionKey(`trie-${item.word}-${letterIndex}`)}><circle cx="${55 + letterIndex * 55}" cy="${rowY}" r="16" /><text x="${55 + letterIndex * 55}" y="${rowY + 4}">${escapeHtml(letter)}</text></g>`).join('');
    return `${edges}${nodes}<text class="coding-trace-node-state" x="${Math.min(width - 80, 80 + letters.length * 55)}" y="${rowY + 4}">${escapeHtml(item.prefix)}</text>`;
  }).join('');
  return `<div class="coding-trace-trie"><svg viewBox="0 0 ${width} ${Math.max(80, scene.paths.length * 58)}" role="img" aria-label="Trie prefix topology">${paths}</svg></div>${renderMeta(scene, ['type', 'paths', 'motion'])}`;
}

function renderBitsScene(scene) {
  const marks = new Map((scene.marks ?? []).map((item) => [item.index, item]));
  const values = scene.values.map((value, index) => `<span class="coding-trace-bit${toneClass(marks.get(index)?.tone)}"><b>${escapeHtml(value)}</b>${marks.get(index)?.label ? `<small>${escapeHtml(marks.get(index).label)}</small>` : ''}</span>`).join('');
  return `<div class="coding-trace-bits">${values}</div>${renderMeta(scene, ['type', 'values', 'marks'])}`;
}

function renderShapesScene(scene) {
  return `<div class="coding-trace-shapes">${scene.items.map((item, index) => `${index > 0 ? '<span class="coding-trace-link-arrow">&rarr;</span>' : ''}<span class="coding-trace-shape${index === scene.items.length - 1 ? ' is-output' : index === 0 ? ' is-input' : ' is-state'}">${escapeHtml(item)}</span>`).join('')}</div>${renderMeta(scene, ['type', 'items'])}`;
}

function renderAttentionScene(scene) {
  const cells = scene.rows.map((row) => row.map((value) => `<span class="coding-trace-attention-cell ${value === 'mask' ? 'is-mask' : value === 'read' ? 'is-read' : ''}">${escapeHtml(value)}</span>`).join('')).join('');
  return `<div class="coding-trace-attention" style="--trace-cols:${Math.max(1, scene.rows[0]?.length ?? 1)}">${cells}</div>${renderMeta(scene, ['type', 'rows'])}`;
}

function renderBucketsScene(scene) {
  const columns = scene.items.map((item) => `<div class="coding-trace-bucket${toneClass(item.tone)}"><strong>${escapeHtml(item.count)}</strong>${item.items.map((value) => `<span>${escapeHtml(value)}</span>`).join('')}</div>`).join('');
  return `<div class="coding-trace-buckets">${columns}</div>${renderMeta(scene, ['type', 'items'])}`;
}

function renderPrefixScene(scene) {
  const rows = [['input', scene.items], ['left', scene.left], ['right', scene.right], ['answer', scene.answer]].map(([label, values]) => `<div class="coding-trace-prefix-row"><span class="coding-trace-label">${escapeHtml(label)}</span>${(values ?? []).map((value, index) => `<span class="coding-trace-prefix-cell${index === scene.active ? ' is-active' : ''}">${escapeHtml(value)}</span>`).join('')}</div>`).join('');
  return `<div class="coding-trace-prefix">${rows}</div>${renderMeta(scene, ['type', 'items', 'left', 'right', 'answer', 'active'])}`;
}

function renderDualWindowScene(scene) {
  const rows = scene.windows.map((window) => `<div class="coding-trace-window-row"><span class="coding-trace-label">${escapeHtml(window.label)}</span>${scene.items.map((item, index) => `<span class="coding-trace-window-cell${index >= window.range[0] && index <= window.range[1] ? ' is-inside' : ''}">${escapeHtml(item)}</span>`).join('')}<b>${escapeHtml(window.count)}</b></div>`).join('');
  return `<div class="coding-trace-dual-window">${rows}</div>${renderMeta(scene, ['type', 'items', 'windows'])}`;
}

function renderChoicesScene(scene) {
  const pathText = scene.path.length > 0 ? scene.path.join(' -> ') : 'empty';
  return `<div class="coding-trace-choices"><div class="coding-trace-choice-path"><span class="coding-trace-label">path</span><strong>${escapeHtml(pathText)}</strong></div><div class="coding-trace-choice-branches">${scene.branches.map((branch) => `<span>${escapeHtml(branch)}</span>`).join('')}</div></div>${renderMeta(scene, ['type', 'path', 'branches'])}`;
}

function renderLruScene(scene) {
  const entries = scene.map.map(([key, value]) => `<span class="coding-trace-lru-map"><b>${escapeHtml(key)}</b>${escapeHtml(value)}</span>`).join('');
  const order = scene.order.map((item) => `<span class="coding-trace-lru-node">${escapeHtml(item)}</span>`).join('<span class="coding-trace-link-arrow">&rarr;</span>');
  return `<div class="coding-trace-lru"><div class="coding-trace-lru-map-row"><span class="coding-trace-label">map</span>${entries}</div><div class="coding-trace-lru-order"><span class="coding-trace-label">list</span>${order}</div></div>${renderMeta(scene, ['type', 'map', 'order'])}`;
}

function renderHeapScene(scene) {
  const groups = scene.groups ?? [{ label: scene.label ?? 'heap', values: scene.values }];
  const occurrences = new Map();
  const keyedGroups = groups.map((group) => ({
    ...group,
    values: group.values.map((value) => {
      const count = occurrences.get(value) ?? 0;
      occurrences.set(value, count + 1);
      return { value, key: `heap-value-${value}-${count}` };
    }),
  }));
  const renderNode = (values, index) => {
    if (index >= values.length) return '';
    const item = values[index];
    const left = index * 2 + 1;
    const right = index * 2 + 2;
    const children = [renderNode(values, left), renderNode(values, right)].filter(Boolean).join('');
    const current = String(item.value) === String(scene.current) ? ' is-current' : '';
    return `<li><span class="coding-trace-heap-value${index === 0 ? ' is-root' : ''}${current}"${motionKey(item.key)}>${escapeHtml(item.value)}</span>${children ? `<ul>${children}</ul>` : ''}</li>`;
  };
  const renderedGroups = keyedGroups.map((group) => `<section class="coding-trace-heap-group"><h4>${escapeHtml(group.label)}</h4>${group.values.length ? `<ul class="coding-trace-heap-tree">${renderNode(group.values, 0)}</ul>` : '<span class="coding-trace-empty">empty</span>'}</section>`).join('');
  return `<div class="coding-trace-heap-groups" role="img" aria-label="Complete binary heap topology">${renderedGroups}</div>${renderMeta(scene, ['type', 'values', 'groups', 'label', 'motion'])}`;
}

function renderScene(scene) {
  const renderers = {
    array: renderArrayScene,
    'array-map': renderArrayMapScene,
    table: renderTableScene,
    grid: renderGridScene,
    stack: renderStackScene,
    'queue-grid': renderQueueGridScene,
    graph: renderGraphScene,
    tree: renderTreeScene,
    intervals: renderIntervalsScene,
    linked: renderLinkedScene,
    trie: renderTrieScene,
    bits: renderBitsScene,
    bars: renderBarsScene,
    shapes: renderShapesScene,
    attention: renderAttentionScene,
    buckets: renderBucketsScene,
    prefix: renderPrefixScene,
    'dual-window': renderDualWindowScene,
    choices: renderChoicesScene,
    lru: renderLruScene,
    heap: renderHeapScene,
  };
  const renderer = renderers[scene.type];
  if (!renderer) throw new Error(`Unknown coding visual scene type: ${scene.type}`);
  return renderer(scene);
}

function renderVisual(problem) {
  const definition = codingQuestionVisuals[problem.slug];
  if (!definition) throw new Error(`Missing visual definition for ${problem.slug}`);
  const visualId = `${problem.slug}-state`;
  const titleId = `${visualId}-title`;
  const frames = definition.frames.map((item, index) => `<div class="coding-trace-frame" data-coding-frame="${index}" data-frame-key="${escapeHtml(item.key)}"${index > 0 ? ' hidden' : ''} role="group" aria-label="${escapeHtml(item.label)}"><div class="coding-trace-frame-heading"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.note)}</strong></div>${renderScene(item.scene)}</div>`).join('');
  const buttons = definition.frames.map((item, index) => `<button type="button" data-coding-frame-button="${index}"${index === 0 ? ' aria-current="step"' : ''}><span>${index + 1}</span><strong>${escapeHtml(item.label)}</strong></button>`).join('');
  const controls = `<div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of ${definition.frames.length}</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps">${buttons}</div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p>`;
  const lessons = `<dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">${escapeHtml(definition.review.recognitionCue)}</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">${escapeHtml(definition.review.invariant)}</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">${escapeHtml(definition.review.transferLesson)}</dd></div></dl>`;
  return {
    visualId,
    source: `<!-- visual:${visualId} -->\n<figure class="learning-figure coding-visual-figure" aria-labelledby="${titleId}"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="${titleId}">${escapeHtml(problem.title)}: ${escapeHtml(definition.objective)}</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="${problem.slug}" role="group" tabindex="0" aria-label="${escapeHtml(`${problem.title}: ${definition.objective}`)}"><div class="coding-visual-example"><span>Input and goal</span><strong>${escapeHtml(problem.task)}</strong></div><div class="coding-trace" data-coding-trace>${frames}${controls}</div>${lessons}</div><figcaption><strong>Read it this way:</strong> ${escapeHtml(definition.frames[0].note)} Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>`,
    audit: {
      schemaVersion: 1,
      slug: problem.slug,
      article: `src/content/posts/${publicationDate}-${problem.slug}.md`,
      status: 'implemented',
      medium: 'semantic-html',
      learningObjective: definition.objective,
      mechanismReview: definition.review,
      mediumRationale: 'An original, problem-specific trace shows the actual input, changing state, safe transition, and result. The static first frame works without JavaScript; controls add step-by-step playback without replacing the proof with decoration.',
      mediumComparison: {
        mermaid: 'Rejected: automatic graph layout would hide the exact data values and state changes that this problem needs.',
        svg: 'Rejected: the trace is a changing data state, so semantic HTML keeps values selectable, readable, and responsive.',
        semanticHtml: 'Selected: each problem owns concrete frames rendered as arrays, maps, grids, trees, stacks, queues, intervals, pointers, bits, or tensor shapes.',
        interaction: 'Selected as progressive enhancement: Play, Previous, Next, and step buttons reveal the same authored frames while static HTML remains complete.',
        paperReuse: 'Rejected: the visual is an original synthesis of the supplied implementation and does not reuse source artwork.',
        noVisual: 'Rejected: prose alone would make the learner simulate the changing data structure in working memory.',
      },
      deckReview: { pages: [], notes: 'No source slide deck is part of the supplied coding guide. The visual was designed from this problem\'s source implementation and worked trace.' },
      sourceReview: { sources: [], notes: 'The supplied Coding_Questions_Phone_Guide.md is the curriculum source. The task, implementation sketch, and complexity statement are preserved; the trace frames are original explanatory artwork.' },
      agentReview: { reviewer: 'GitHub Copilot', reviewedAt: publicationDate, summary: 'Checked each authored frame against the implementation path, verified that the displayed state changes are problem-specific, and checked the static and interactive accessibility paths.' },
      implementation: { visualIds: [visualId], accessibility: 'The figure has a labelled title, a complete group description, visible values and state labels, a direct Read it this way caption, keyboard controls, and a static first-frame fallback.' },
    },
  };
}

function parseProblems(source) {
  const headings = [...source.matchAll(/^## ((?:\d+|AI\d+|H\d+)\. .+)$/gm)];
  return headings.map((match, index) => {
    const sourceHeading = match[1];
    const sectionStart = match.index;
    const sectionEnd = headings[index + 1]?.index ?? source.length;
    let section = source.slice(sectionStart, sectionEnd).replace(/^## .+\n/, '').trim();
    const nextHeading = section.search(/\n\s*(?:<a id="[^"]+"><\/a>\s*)?#{1,4} /);
    if (nextHeading >= 0) section = section.slice(0, nextHeading).trim();
    section = section.replace(/\n\s*<a id="[^"]+"><\/a>\s*$/, '').trim();
    const identifierMatch = sourceHeading.match(/^((?:\d+|AI\d+|H\d+))\. (.+)$/);
    const identifier = identifierMatch[1];
    const title = identifierMatch[2].trim();
    const task = extractField(section, 'Task');
    const pattern = extractField(section, 'Pattern');
    section = section.replace(/^\*\*Task:\*\*[\s\S]*?(?=\n\n)/, '').trim();
    return { identifier, title, task, pattern, slug: slugify(title), section };
  });
}

function chapterFor(problem) {
  return chapterDefinitions.find((chapter) => chapter.numbers.includes(problem.identifier));
}

function writeProblem(problem) {
  const visual = renderVisual(problem);
  const platformNote = problem.section.includes('TreeNode') && !problem.section.includes('class TreeNode')
    ? '\n\nThe platform supplies `TreeNode` with `val`, `left`, and `right`; this snippet assumes that definition.'
    : problem.section.includes('ListNode') && !problem.section.includes('class ListNode')
      ? '\n\nThe platform supplies `ListNode` with `val` and `next`; this snippet assumes that definition.'
      : '';
  const body = `> ${problem.task}\n\nStart with the concrete trace below. It shows the state the algorithm must carry as it runs.\n\n${visual.source}\n\n${wrapParagraphs(problem.section)}${platformNote}`;
  const difficulty = problem.identifier.startsWith('H') ? 'Advanced' : problem.identifier.startsWith('AI') ? 'Intermediate' : chapterFor(problem).difficulty;
  const priority = problem.identifier.startsWith('AI') ? 'Role-specific' : problem.identifier.startsWith('H') ? 'Specialist' : 'Core';
  const frontmatter = [
    '---',
    `title: ${JSON.stringify(problem.title)}`,
    `description: ${JSON.stringify(problem.task)}`,
    `date: "${publicationDate}"`,
    'draft: false',
    'tags: ["coding interview", "data structures"]',
    'category: "questions"',
    'roles: ["MLE", "RE", "AS"]',
    'rounds: ["Coding", "ML implementation"]',
    `difficulty: "${difficulty}"`,
    `priority: "${priority}"`,
    'aliases: []',
    'prerequisites: []',
    '---',
  ].join('\n');
  fs.writeFileSync(path.join(postsDir, `${publicationDate}-${problem.slug}.md`), `${frontmatter}\n\n${body}\n`);
  fs.writeFileSync(path.join(auditsDir, `${problem.slug}.json`), `${JSON.stringify(visual.audit, null, 2)}\n`);
  return problem;
}

const source = fs.readFileSync(sourcePath, 'utf8');
const allProblems = parseProblems(source);
const problems = requestedSlugs.size > 0
  ? allProblems.filter((problem) => requestedSlugs.has(problem.slug))
  : allProblems;
const force = process.argv.includes('--force');
if (allProblems.length !== 106) throw new Error(`Expected 106 problems, found ${allProblems.length}`);
if (Object.keys(codingQuestionVisuals).length !== allProblems.length) throw new Error(`Visual registry count does not match problem count`);
if (requestedSlugs.size > 0 && problems.length !== requestedSlugs.size) {
  const found = new Set(problems.map((problem) => problem.slug));
  throw new Error(`Unknown requested slug(s): ${[...requestedSlugs].filter((slug) => !found.has(slug)).join(', ')}`);
}
const existingSlugs = new Set(fs.readdirSync(postsDir).map((name) => name.replace(/\.mdx?$/, '').replace(/^\d{4}-\d{2}-\d{2}-/, '')));
for (const problem of problems) {
  if (existingSlugs.has(problem.slug) && !force && requestedSlugs.size === 0) throw new Error(`Slug already exists: ${problem.slug}`);
  if (!problem.task || !problem.pattern) throw new Error(`Missing metadata for ${problem.identifier}`);
  if (!chapterFor(problem)) throw new Error(`No chapter for ${problem.identifier}`);
  if (!codingQuestionVisuals[problem.slug]) throw new Error(`No visual definition for ${problem.slug}`);
}
fs.mkdirSync(postsDir, { recursive: true });
fs.mkdirSync(auditsDir, { recursive: true });
const generated = problems.map(writeProblem);
const registry = chapterDefinitions.map((chapter) => ({ ...chapter, slugs: allProblems.filter((problem) => chapter.numbers.includes(problem.identifier)).map((problem) => problem.slug) }));
const registryPath = process.env.CODING_BOOK_REGISTRY || '/tmp/coding-interview-book-registry.json';
if (requestedSlugs.size === 0) fs.writeFileSync(registryPath, `${JSON.stringify(registry, null, 2)}\n`);
console.log(`Generated ${generated.length} coding question pages and audits.`);
if (requestedSlugs.size === 0) console.log(`Registry written to ${registryPath}.`);
console.log(`Visual scenes: ${[...new Set(Object.values(codingQuestionVisuals).flatMap((definition) => definition.frames.map((item) => item.scene.type)))].sort().join(', ')}.`);
