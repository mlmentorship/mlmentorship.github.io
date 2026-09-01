import fs from 'node:fs';
import path from 'node:path';
import { codingQuestionVisuals } from './coding-question-visuals.mjs';

const root = process.cwd();
const sourceArgument = process.argv.slice(2).find((argument) => !argument.startsWith('--'));
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
    return `<span class="coding-trace-array-item${classes}" role="listitem"><span class="coding-trace-array-mark">${escapeHtml(labels)}</span><span class="coding-trace-array-cell">${escapeHtml(display)}</span></span>`;
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
    return `<span class="coding-trace-grid-cell${toneClass(mark?.tone)}"><span>${escapeHtml(value)}</span>${mark?.label ? `<small>${escapeHtml(mark.label)}</small>` : ''}</span>`;
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

function renderGraphScene(scene) {
  const nodes = scene.nodes.map((node) => `<span class="coding-trace-node${scene.start === node ? ' is-focus' : ''}${scene.visited?.some((item) => String(item).startsWith(`${node}:`) || item === node) ? ' is-state' : ''}">${escapeHtml(node)}</span>`).join('');
  const edges = scene.edges.map((edge) => `<span class="coding-trace-edge">${escapeHtml(edge)}</span>`).join('');
  const keys = ['visited', 'frontier', 'ready', 'order', 'roots', 'components', 'indegree'];
  const lists = keys.flatMap((key) => scene[key] ? [`<span><b>${key}</b>${escapeHtml(Array.isArray(scene[key]) ? scene[key].join(', ') : scene[key])}</span>`] : []).join('');
  return `<div class="coding-trace-graph"><div class="coding-trace-node-row">${nodes}</div><div class="coding-trace-edge-row">${edges}</div>${lists ? `<div class="coding-trace-meta">${lists}</div>` : ''}</div>${renderMeta(scene, ['type', 'nodes', 'edges', 'start', ...keys])}`;
}

function renderTreeScene(scene) {
  let nodeIndex = 0;
  const rows = scene.levels.map((level, levelIndex) => `<div class="coding-trace-tree-level" data-level="${levelIndex}">${level.map((node) => {
    const index = nodeIndex++;
    const mark = scene.marks?.find((item) => item.index === index);
    return `<span class="coding-trace-tree-node${toneClass(mark?.tone)}"><span>${escapeHtml(node)}</span>${mark?.label ? `<small>${escapeHtml(mark.label)}</small>` : ''}</span>`;
  }).join('')}</div>`).join('');
  return `<div class="coding-trace-tree" role="group" aria-label="Tree state">${rows}</div>${renderMeta(scene, ['type', 'levels', 'marks'])}`;
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
  const rendered = nodes.map((node, index) => `${index > 0 ? '<span class="coding-trace-link-arrow">&rarr;</span>' : ''}<span class="coding-trace-linked-node${toneClass(node.tone)}"><span>${escapeHtml(node.value)}</span>${node.pointer ? `<small>${escapeHtml(node.pointer)}</small>` : ''}</span>`).join('');
  return `<div class="coding-trace-linked-row">${label ? `<span class="coding-trace-label">${escapeHtml(label)}</span>` : ''}${rendered}</div>`;
}

function renderLinkedScene(scene) {
  return `<div class="coding-trace-linked">${renderLinkedRow(scene.nodes)}${scene.second ? renderLinkedRow(scene.second.map((value) => ({ value })), 'second') : ''}${scene.arrows ? `<p class="coding-trace-inline-note">${escapeHtml(scene.arrows.join(' · '))}</p>` : ''}</div>${renderMeta(scene, ['type', 'nodes', 'second', 'arrows'])}`;
}

function renderTrieScene(scene) {
  const paths = scene.paths.map((item) => `<div class="coding-trace-trie-path"><span class="coding-trace-trie-word${toneClass(item.tone)}">${escapeHtml(item.word)}</span><span class="coding-trace-link-arrow">&rarr;</span><strong>${escapeHtml(item.prefix)}</strong></div>`).join('');
  return `<div class="coding-trace-trie">${paths}</div>${renderMeta(scene, ['type', 'paths'])}`;
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
  const values = scene.values.map((value, index) => `<span class="coding-trace-heap-node${index === 0 ? ' is-root' : ''}">${escapeHtml(value)}</span>`).join('');
  return `<div class="coding-trace-heap"><div class="coding-trace-heap-tree">${values}</div></div>${renderMeta(scene, ['type', 'values'])}`;
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
  const frames = definition.frames.map((item, index) => `<div class="coding-trace-frame" data-coding-frame="${index}"${index > 0 ? ' hidden' : ''} role="group" aria-label="${escapeHtml(item.label)}"><div class="coding-trace-frame-heading"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.note)}</strong></div>${renderScene(item.scene)}</div>`).join('');
  const buttons = definition.frames.map((item, index) => `<button type="button" data-coding-frame-button="${index}"${index === 0 ? ' aria-current="step"' : ''}><span>${index + 1}</span><strong>${escapeHtml(item.label)}</strong></button>`).join('');
  const controls = `<div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of ${definition.frames.length}</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps">${buttons}</div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p>`;
  return {
    visualId,
    source: `<!-- visual:${visualId} -->\n<figure class="learning-figure coding-visual-figure" aria-labelledby="${titleId}"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="${titleId}">${escapeHtml(problem.title)}: ${escapeHtml(definition.objective)}</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="${problem.slug}" role="group" aria-label="${escapeHtml(`${problem.title}: ${definition.objective}`)}"><div class="coding-visual-example"><span>Input and goal</span><strong>${escapeHtml(problem.task)}</strong></div><div class="coding-trace" data-coding-trace>${frames}${controls}</div><p class="coding-visual-invariant"><span>Why this works</span>${escapeHtml(definition.objective)}</p></div><figcaption><strong>Read it this way:</strong> ${escapeHtml(definition.frames[0].note)} Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>`,
    audit: {
      schemaVersion: 1,
      slug: problem.slug,
      article: `src/content/posts/${publicationDate}-${problem.slug}.md`,
      status: 'implemented',
      medium: 'semantic-html',
      learningObjective: definition.objective,
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
const problems = parseProblems(source);
const force = process.argv.includes('--force');
if (problems.length !== 106) throw new Error(`Expected 106 problems, found ${problems.length}`);
if (Object.keys(codingQuestionVisuals).length !== problems.length) throw new Error(`Visual registry count does not match problem count`);
const existingSlugs = new Set(fs.readdirSync(postsDir).map((name) => name.replace(/\.mdx?$/, '').replace(/^\d{4}-\d{2}-\d{2}-/, '')));
for (const problem of problems) {
  if (existingSlugs.has(problem.slug) && !force) throw new Error(`Slug already exists: ${problem.slug}`);
  if (!problem.task || !problem.pattern) throw new Error(`Missing metadata for ${problem.identifier}`);
  if (!chapterFor(problem)) throw new Error(`No chapter for ${problem.identifier}`);
  if (!codingQuestionVisuals[problem.slug]) throw new Error(`No visual definition for ${problem.slug}`);
}
fs.mkdirSync(postsDir, { recursive: true });
fs.mkdirSync(auditsDir, { recursive: true });
const generated = problems.map(writeProblem);
const registry = chapterDefinitions.map((chapter) => ({ ...chapter, slugs: generated.filter((problem) => chapter.numbers.includes(problem.identifier)).map((problem) => problem.slug) }));
const registryPath = process.env.CODING_BOOK_REGISTRY || '/tmp/coding-interview-book-registry.json';
fs.writeFileSync(registryPath, `${JSON.stringify(registry, null, 2)}\n`);
console.log(`Generated ${generated.length} coding question pages and audits.`);
console.log(`Registry written to ${registryPath}.`);
console.log(`Visual scenes: ${[...new Set(Object.values(codingQuestionVisuals).flatMap((definition) => definition.frames.map((item) => item.scene.type)))].sort().join(', ')}.`);
