import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { parseFragment } from 'parse5';
import { figureTakeaways } from './illustration-takeaways.mjs';

const escape = (value) => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const ink = (tone) => `var(--${tone === 'text' ? 'c-text' : tone === 'soft' ? 'c-text-soft' : `viz-${tone}-stroke`})`;

function words(value, maximum) {
  return String(value).split('\n').flatMap((line) => {
    const lines = [''];
    for (const word of line.split(/\s+/)) {
      const current = lines.at(-1);
      if (current && `${current} ${word}`.length > maximum) lines.push(word);
      else lines[lines.length - 1] = current ? `${current} ${word}` : word;
    }
    return lines;
  });
}

function text(horizontal, vertical, label, { size = 14, width = 300, anchor = 'middle', weight = 400, tone = 'text' } = {}) {
  const lines = words(label, Math.floor(width / (size * .57)));
  return `<text x="${horizontal}" y="${vertical}" style="font-size:${size}px;font-weight:${weight};fill:${ink(tone)};text-anchor:${anchor}">${lines.map((line, index) => `<tspan x="${horizontal}" dy="${index ? 19 : 0}">${escape(line)}</tspan>`).join('')}</text>`;
}

function rule(fromX, fromY, toX, toY, tone = 'neutral', dashed = false) {
  return `<path d="M${fromX} ${fromY}L${toX} ${toY}" style="fill:none;stroke:${ink(tone)};stroke-width:1.75${dashed ? ';stroke-dasharray:5 4' : ''}"></path>`;
}

function arrow(fromX, fromY, toX, toY, tone = 'neutral') {
  const angle = Math.atan2(toY - fromY, toX - fromX);
  const tip = (offset) => `${toX - 7 * Math.cos(angle + offset)},${toY - 7 * Math.sin(angle + offset)}`;
  return `${rule(fromX, fromY, toX, toY, tone)}<path d="M${tip(-.5)}L${toX},${toY}L${tip(.5)}" style="fill:none;stroke:${ink(tone)};stroke-width:1.75"></path>`;
}

function box(horizontal, vertical, width, height, tone = 'neutral', dashed = false) {
  return `<rect x="${horizontal}" y="${vertical}" width="${width}" height="${height}" rx="3" style="fill:var(--viz-${tone}-bg);stroke:${ink(tone)};stroke-width:1.3${dashed ? ';stroke-dasharray:5 3' : ''}"></rect>`;
}

function node(horizontal, vertical, width, height, label, tone = 'neutral') {
  const lineCount = words(label, Math.floor((width - 16) / (14 * .57))).length;
  return `<g data-editorial-node>${box(horizontal, vertical, width, height, tone)}${text(horizontal + width / 2, vertical + height / 2 - (lineCount - 1) * 9.5 + 5, label, { width: width - 16 })}</g>`;
}

function gate(model) {
  const outcomeHeight = Math.max(100, ...[model.yes, model.no].map((label) => words(label, Math.floor((152 - 16) / (14 * .57))).length * 19 + 24));
  const exampleOffset = (words(model.example, Math.floor(300 / (12 * .57))).length - 1) * 19;
  return {
    height: 290 + outcomeHeight + exampleOffset,
    body: text(180, 22, model.example, { size: 12, tone: 'soft' })
      + `<g transform="translate(0 ${exampleOffset})">`
      + node(30, 43, 300, 58, model.input, 'input')
      + arrow(180, 101, 180, 128)
      + node(30, 132, 300, 76, model.test, 'focus')
      + rule(180, 208, 180, 229) + rule(90, 229, 270, 229)
      + arrow(90, 229, 90, 266, 'output') + arrow(270, 229, 270, 266, 'warning')
      + text(90, 253, model.yesLabel ?? 'PASS', { size: 12, tone: 'output', weight: 700, width: 140 })
      + text(270, 253, model.noLabel ?? 'FAIL', { size: 12, tone: 'warning', weight: 700, width: 140 })
      + node(14, 270, 152, outcomeHeight, model.yes, 'output')
      + node(194, 270, 152, outcomeHeight, model.no, 'warning') + '</g>',
  };
}

function comparison(model) {
  const headingLines = Math.max(...[model.left, model.right].map((label) => words(label, Math.floor(152 / (14 * .57))).length));
  const exampleOffset = (words(model.example, Math.floor(300 / (12 * .57))).length - 1) * 19;
  let vertical = 75 + exampleOffset + (headingLines - 1) * 19;
  let body = text(180, 22, model.example, { size: 12, tone: 'soft' })
    + text(90, 51 + exampleOffset, model.left, { width: 152, weight: 700 })
    + text(270, 51 + exampleOffset, model.right, { width: 152, weight: 700 });
  for (const row of model.rows) {
    const labelOffset = (words(row.label, Math.floor(300 / (12 * .57))).length - 1) * 19;
    const rowHeight = Math.max(67, ...[row.left, row.right].map((label) => words(label, Math.floor((150 - 16) / (14 * .57))).length * 19 + 16));
    body += rule(15, vertical - 3, 345, vertical - 3)
      + text(180, vertical + 17, row.label, { size: 12, tone: 'soft' })
      + node(15, vertical + 31 + labelOffset, 150, rowHeight, row.left, row.leftTone ?? 'input')
      + node(195, vertical + 31 + labelOffset, 150, rowHeight, row.right, row.rightTone ?? 'output');
    vertical += labelOffset + rowHeight + 49;
  }
  return { height: vertical + 12, body };
}

function actorCriticPaths() {
  return { height: 456, body:
    node(125, 12, 110, 40, 'state s', 'input')
    + arrow(180, 52, 180, 84)
    + node(90, 88, 180, 50, 'shared features', 'state')
    + rule(180, 138, 180, 158) + rule(90, 158, 270, 158)
    + arrow(90, 158, 90, 172) + arrow(270, 158, 270, 172)
    + node(15, 172, 150, 58, 'actor\nlog pi(a|s)', 'input')
    + node(195, 172, 150, 58, 'critic\nV(s)', 'focus')
    + arrow(90, 230, 90, 340) + arrow(270, 230, 270, 340)
    + text(180, 253, 'fixed target y', { size: 12, width: 120 })
    + node(120, 263, 120, 42, 'Ahat = y - V', 'neutral')
    + arrow(270, 280, 240, 280)
    + arrow(120, 305, 102, 340)
    + node(15, 340, 150, 62, 'policy loss\n-log pi x Ahat', 'input')
    + node(195, 340, 150, 62, 'value loss\n(V - y)^2', 'focus')
    + rule(165, 370, 180, 370, 'warning', true)
    + rule(180, 370, 180, 320, 'warning', true)
    + arrow(180, 320, 180, 305, 'warning')
    + rule(240, 293, 258, 293, 'warning', true)
    + rule(258, 286, 258, 300, 'warning')
    + text(240, 319, 'STOP', { size: 12, width: 58, tone: 'warning', weight: 700 })
    + rule(15, 370, 8, 370, 'warning', true)
    + rule(8, 370, 8, 201, 'warning', true)
    + arrow(8, 201, 15, 201, 'warning')
    + rule(345, 370, 352, 370, 'warning', true)
    + rule(352, 370, 352, 201, 'warning', true)
    + arrow(352, 201, 345, 201, 'warning')
    + rule(75, 172, 75, 150, 'warning', true)
    + rule(75, 150, 140, 150, 'warning', true)
    + arrow(140, 150, 140, 138, 'warning')
    + rule(285, 172, 285, 150, 'warning', true)
    + rule(285, 150, 220, 150, 'warning', true)
    + arrow(220, 150, 220, 138, 'warning')
    + text(180, 435, 'solid: values | dashed: loss gradients', { size: 12, width: 330 }) };
}

function feedback() {
  return { height: 340, body:
    text(180, 24, 'A short experiment closes here', { weight: 700 })
    + rule(172, 46, 172, 305, 'focus', true)
    + text(82, 64, 'INSIDE WINDOW', { size: 12, width: 140, weight: 700 })
    + text(266, 64, 'NOT OBSERVED YET', { size: 12, width: 160, weight: 700 })
    + text(18, 110, 'Viewer response', { anchor: 'start', weight: 700 })
    + rule(26, 137, 326, 137)
    + box(26, 126, 118, 22, 'input')
    + text(86, 173, 'seconds to days', { size: 12, width: 140 })
    + text(18, 219, 'Creator supply response', { anchor: 'start', weight: 700 })
    + rule(26, 249, 326, 249)
    + box(204, 238, 122, 22, 'state', true)
    + text(266, 284, 'weeks; direction unknown', { size: 12, width: 150 })
    + text(180, 325, 'Ordered horizons, not a measured time scale', { size: 12, tone: 'soft' }) };
}

function editorRace() {
  const events = [
    [81, 'version 41', 'R1 captures v41'],
    [145, 'typing: v42', 'R1 still running'],
    [209, 'keep new text', 'R1 returns v41'],
    [290, 'current: v42', 'R2 returns v42'],
  ];
  let body = text(90, 25, 'EDITOR', { weight: 700 }) + text(270, 25, 'SERVER', { weight: 700 })
    + rule(90, 45, 90, 340, 'input') + rule(270, 45, 270, 340, 'state');
  for (const [vertical, left, right] of events) {
    body += box(20, vertical - 20, 140, 38, 'input') + text(90, vertical + 4, left, { width: 130, size: 13 })
      + box(200, vertical - 20, 140, 38, 'state') + text(270, vertical + 4, right, { width: 130, size: 13 });
  }
  body += arrow(160, 81, 200, 81)
    + arrow(200, 209, 160, 209, 'warning')
    + text(180, 251, '41 != 42: DISCARD R1', { tone: 'warning', weight: 700 })
    + arrow(200, 290, 160, 290, 'output')
    + text(180, 334, 'R2 still needs live version + policy checks', { width: 325, size: 12 })
    + text(180, 374, 'A tool task needs separate authorization', { size: 12, tone: 'soft' });
  return { height: 390, body };
}

function confounding() {
  return { height: 315, body:
    node(106, 20, 148, 65, 'D: user difficulty', 'state')
    + node(15, 174, 140, 62, 'T: support message', 'input')
    + node(205, 174, 140, 62, 'Y: churn', 'output')
    + arrow(141, 85, 85, 174, 'state') + arrow(219, 85, 275, 174, 'state')
    + text(67, 117, 'assignment', { size: 12, width: 110, tone: 'state' })
    + text(292, 117, 'outcome', { size: 12, width: 100, tone: 'state' })
    + arrow(155, 204, 205, 204, 'focus')
    + text(180, 262, 'BACKDOOR: T <- D -> Y', { weight: 700, tone: 'state' })
    + text(180, 289, 'Adjustment conditions on D; it does not erase arrows.', { size: 12, width: 325 }) };
}

function memory() {
  const model = comparison({ example: 'Two-byte elements; same intermediate h [t, F]', left: 'TWO KERNELS', right: 'FUSED TILE', rows: [
    { label: 'Where h waits between GEMMs', left: 'HBM', right: 'SRAM / registers' },
    { label: 'Intermediate round trip', left: 'write 2tF\nread 2tF bytes', right: 'no HBM round trip' },
    { label: 'What still moves', left: 'inputs, weights, output', right: 'inputs, weights, output', rightTone: 'input' },
  ] });
  model.body += text(180, model.height + 12, '4tF intermediate bytes avoided', { weight: 700, tone: 'output' })
    + text(180, model.height + 42, 'Tile fit and other traffic still limit performance.', { size: 12, width: 320 });
  model.height += 88;
  return model;
}

function moderation() {
  const left = 25;
  const width = 310;
  return { height: 345, body:
    text(180, 23, 'Illustrative policy P7; not universal thresholds', { size: 12, tone: 'soft' })
    + text(180, 65, 'One item receives score 0.62', { weight: 700 })
    + box(left, 98, width * .4, 27, 'output') + box(left + width * .4, 98, width * .4, 27, 'focus')
    + box(left + width * .8, 98, width * .2, 27, 'warning')
    + text(left, 146, '0', { size: 12 }) + text(left + width * .4, 146, '0.40', { size: 12 })
    + text(left + width * .8, 146, '0.80', { size: 12 }) + text(left + width, 146, '1', { size: 12 })
    + arrow(left + width * .62, 70, left + width * .62, 98, 'focus')
    + text(180, 183, '0.40 <= 0.62 < 0.80: HUMAN REVIEW', { size: 13, weight: 700, tone: 'focus' })
    + node(30, 211, 300, 63, 'Decision + policy P7 + reviewer provenance', 'state')
    + text(180, 307, 'Allowed-item audits and appeals also update the record.', { size: 12, width: 320 }) };
}

function paper() {
  let body = text(180, 22, 'Reported comparison; not matched compute', { size: 12, tone: 'soft' });
  const bars = [{ label: 'accuracy / 100%', values: [80, 84], maximum: 100, unit: '%' }, { label: 'GPU-hours / 20', values: [10, 20], maximum: 20, unit: 'h' }];
  let vertical = 60;
  for (const metric of bars) {
    body += text(15, vertical, metric.label, { anchor: 'start', weight: 700 });
    for (const [index, value] of metric.values.entries()) {
      const row = vertical + 24 + index * 36;
      body += text(15, row + 14, index ? 'new' : 'base', { size: 12, anchor: 'start', width: 40 })
        + box(62, row, 220 * value / metric.maximum, 20, index ? 'focus' : 'input')
        + text(332, row + 15, `${value}${metric.unit}`, { size: 12, anchor: 'end' });
    }
    vertical += 119;
  }
  body += text(180, 316, 'Next: 3 paired runs at 10 hours per recipe', { size: 13, weight: 700 })
    + text(180, 376, '?          ?          ?', { size: 25, tone: 'state' })
    + text(180, 415, 'Matched-compute gain is not known yet.', { size: 13 });
  return { height: 441, body };
}

function availability() {
  return { height: 355, body:
    text(180, 22, 'Prediction cutoff: Monday 12:00', { weight: 700 })
    + box(202, 44, 142, 257, 'warning', true)
    + rule(201, 44, 201, 301, 'focus', true)
    + text(105, 69, 'KNOWABLE', { size: 12, weight: 700 }) + text(272, 69, 'FUTURE', { size: 12, weight: 700 })
    + text(20, 111, 'Payment A', { anchor: 'start', weight: 700 })
    + text(20, 137, 'event Mon 09:00', { anchor: 'start', size: 13 })
    + node(25, 151, 160, 49, 'available Mon 09:05', 'output')
    + text(20, 236, 'Payment B', { anchor: 'start', weight: 700 })
    + text(20, 262, 'event Mon 10:00', { anchor: 'start', size: 13 })
    + node(214, 226, 120, 61, 'available Wednesday', 'warning')
    + arrow(138, 269, 211, 260, 'warning')
    + text(180, 330, 'Earlier event != earlier availability', { weight: 700, tone: 'warning' }) };
}

function divergence() {
  let body = text(180, 22, 'Fictional alarm grid: find the first bad cell', { size: 12, tone: 'soft' });
  for (let step = 0; step < 4; step++) body += text(110 + step * 62, 55, `s${step + 1}`, { size: 12 });
  for (let rank = 0; rank < 4; rank++) {
    body += text(50, 88 + rank * 37, `rank ${rank}`, { size: 12 });
    for (let step = 0; step < 4; step++) {
      const bad = rank === 2 && step >= 2;
      body += box(88 + step * 62, 68 + rank * 37, 44, 29, bad ? 'warning' : 'input')
        + text(110 + step * 62, 87 + rank * 37, bad ? 'NaN' : 'ok', { size: 12 });
    }
  }
  body += arrow(234, 173, 180, 244, 'warning')
    + node(25, 248, 310, 63, 'rank 2 / step 3 -> layer 6 -> gradient tensor', 'focus')
    + text(180, 345, 'Replay: change ONE condition; compare first failure', { size: 12, width: 325, weight: 700 })
    + text(180, 387, 'Recovery from a good checkpoint is a separate task.', { size: 12, width: 325 });
  return { height: 415, body };
}

function losses() {
  let body = text(180, 22, 'Illustrative loss traces; same vertical scale', { size: 12, tone: 'soft' });
  const plots = [
    { title: 'Tiny fixed batch: reaches attainable minimum', train: [2, 1.4, .6, .2, 0] },
    { title: 'Tiny batch stuck: inspect training', train: [2, 2, 1.95, 2, 2] },
    { title: 'Full data: growing gap is a clue, not a cause', train: [2, 1.4, .8, .4, .2], validation: [2, 1.5, 1.4, 1.5, 1.6] },
  ];
  for (const [index, plot] of plots.entries()) {
    const top = 58 + index * 126;
    body += text(18, top, plot.title, { anchor: 'start', size: 12, width: 325, weight: 700 })
      + rule(45, top + 20, 45, top + 91) + rule(45, top + 91, 325, top + 91)
      + text(32, top + 27, '2', { size: 12 }) + text(32, top + 95, '0', { size: 12 });
    const curve = (values, tone, dashed) => `<polyline points="${values.map((value, position) => `${45 + position * 70},${top + 91 - value / 2 * 66}`).join(' ')}" style="fill:none;stroke:${ink(tone)};stroke-width:2.5${dashed ? ';stroke-dasharray:5 3' : ''}"></polyline>`;
    body += curve(plot.train, 'input', false);
    if (plot.validation) body += curve(plot.validation, 'warning', true);
  }
  body += text(180, 456, 'solid: training     dashed: validation', { size: 12 });
  return { height: 475, body };
}

function delivered() {
  return { height: 430, body:
    text(180, 22, 'Illustrative: 1,000 assigned users in each arm', { size: 12, tone: 'soft' })
    + text(20, 61, 'A: 800 successes / 1,000 = 80%', { anchor: 'start', weight: 700 })
    + box(25, 82, 310, 29, 'input')
    + text(180, 102, 'all 1,000 assigned', { size: 13 })
    + text(20, 155, 'B: delivery is part of the treatment', { anchor: 'start', weight: 700 })
    + box(25, 177, 279, 29, 'input') + box(304, 177, 31, 29, 'warning')
    + text(154, 197, '900 model', { size: 13 })
    + rule(319, 206, 319, 229, 'warning') + text(319, 249, '100', { size: 12 })
    + text(117, 244, '738 successes', { width: 170 })
    + text(275, 278, '40 fallback successes', { width: 145, size: 12 })
    + node(30, 305, 300, 58, '(738 + 40) / 1,000 = 77.8%', 'focus')
    + text(180, 385, '82% model-only != 77.8% assigned-user result', { size: 12, tone: 'warning' }) };
}

function gradientMode() {
  return comparison({ example: 'Two independent kinds of persistent state', left: 'UNCORRECTED', right: 'EXPLICIT RESET', rows: [
    { label: 'Before batch 2 backward', left: '.grad = g1', right: 'zero_grad() -> None' },
    { label: 'After batch 2 backward', left: '.grad = g1 + g2', right: '.grad = g2' },
    { label: 'After validation calls eval()', left: 'still eval mode', right: 'train() -> train mode' },
  ] });
}

function portfolio() {
  let body = text(180, 22, 'Toy planning ledger; values are assumptions', { size: 12, tone: 'soft' })
    + text(20, 61, '100 capacity units', { anchor: 'start', weight: 700 })
    + box(25, 82, 62, 28, 'state') + box(87, 82, 248, 28, 'neutral')
    + text(56, 103, '20', { size: 13 }) + text(211, 103, '80 available', { size: 13 })
    + text(25, 137, 'Mandatory work is reserved first.', { anchor: 'start', size: 12 });
  const options = [{ label: 'A: capacity 80 / value 100', parts: [[80, 'input']] }, { label: 'B + C: capacity 80 / value 115', parts: [[45, 'focus'], [35, 'output']] }];
  for (const [index, option] of options.entries()) {
    const top = 184 + index * 100;
    body += text(20, top, option.label, { anchor: 'start', weight: 700 });
    let horizontal = 25;
    for (const [capacity, tone] of option.parts) {
      const width = capacity / 80 * 310;
      body += box(horizontal, top + 20, width, 28, tone) + text(horizontal + width / 2, top + 40, capacity, { size: 13 });
      horizontal += width;
    }
  }
  body += text(180, 378, 'Choosing B + C displaces A, not the mandatory work.', { size: 12, width: 325 })
    + text(180, 420, 'Revisit when value, risk, or capacity evidence changes.', { size: 12, width: 325 });
  return { height: 450, body };
}

function labelHorizons() {
  let body = text(180, 22, 'Four different reasons a label may not be usable', { size: 12, tone: 'soft' });
  const rows = [
    ['Selected; positive arrives early', 'KNOWN +', 'output', 169],
    ['Selected; window still open', 'PENDING', 'state', 321],
    ['Selected; observation ends early', 'CENSORED', 'warning', 161],
    ['Action not selected by policy', 'UNOBSERVED', 'warning', undefined],
  ];
  for (const [index, [label, status, tone, end]] of rows.entries()) {
    const top = 66 + index * 100;
    body += text(20, top, label, { anchor: 'start', width: 325, size: 13, weight: 700 })
      + rule(35, top + 28, 325, top + 28)
      + rule(251, top + 10, 251, top + 52, 'focus', true)
      + text(251, top + 72, 'cutoff', { size: 12 })
      + `<circle cx="35" cy="${top + 28}" r="4" style="fill:${ink('input')}"></circle>`;
    if (end !== undefined) body += box(35, top + 19, end - 35, 18, tone, tone === 'warning');
    body += text(97, top + 72, status, { size: 12, weight: 700, tone });
  }
  body += text(180, 483, 'A task-specific maturity rule decides eligibility.', { size: 12, width: 325 })
    + text(180, 520, 'Unknown is never a synonym for negative.', { size: 13, weight: 700, tone: 'warning' });
  return { height: 542, body };
}

function enterprise() {
  return { height: 465, body:
    text(180, 22, 'Illustrative credit-123; exact amount 50', { size: 12, tone: 'soft' })
    + node(25, 49, 310, 57, '1. CONTROL: authorize exact args', 'input')
    + arrow(180, 106, 180, 133)
    + node(25, 137, 310, 57, '2. EVIDENCE: durable credit-123 record', 'state')
    + arrow(180, 194, 180, 221)
    + node(25, 225, 310, 57, '3. EXECUTION: dispatch key k-456', 'focus')
    + arrow(180, 282, 180, 309)
    + node(25, 313, 310, 57, 'Account commits 50; reply is lost', 'warning')
    + arrow(180, 370, 180, 397)
    + node(25, 401, 310, 49, 'Reconcile k-456 -> same commit receipt', 'output') };
}

function featureVersions() {
  return comparison({ example: 'Illustrative entity E; historical query at noon', left: 'VERSION A', right: 'VERSION B', rows: [
    { label: 'Value and event time', left: '3 at 11:00', right: '7 at 11:50', rightTone: 'input' },
    { label: 'Actually available', left: '11:05', right: '12:10', rightTone: 'warning' },
    { label: 'Historical noon / live 12:15', left: 'noon returns 3', right: '12:15 returns 7' },
  ] });
}

function deletion() {
  return { height: 490, body:
    text(180, 22, 'Illustrative lineage; identities remain linked', { size: 12, tone: 'soft' })
    + node(25, 49, 310, 55, 'source D7 -> shard S2 -> training R3', 'input')
    + arrow(180, 104, 180, 131)
    + node(25, 135, 310, 55, 'R3 -> checkpoint C4 -> released M5', 'state')
    + text(180, 232, 'D7 is excluded from a new data build', { size: 13, weight: 700 })
    + rule(180, 248, 180, 267) + rule(90, 267, 270, 267)
    + arrow(90, 267, 90, 294, 'output') + arrow(270, 267, 270, 294, 'warning')
    + node(15, 299, 150, 100, 'new shard S2b\nexcludes D7', 'output')
    + node(195, 299, 150, 100, 'existing M5\ninfluence unresolved', 'warning')
    + text(180, 439, 'Data cleanup != verified model unlearning', { size: 13, weight: 700, tone: 'warning' })
    + text(180, 472, 'Model intervention needs its own validation.', { size: 12 }) };
}

function redTeam() {
  return { height: 410, body:
    text(180, 22, 'Fictional boundary: private read-only canary', { size: 12, tone: 'soft' })
    + text(79, 66, 'PATH', { size: 12, weight: 700 })
    + text(202, 66, 'BEFORE', { size: 12, weight: 700 })
    + text(300, 66, 'AFTER', { size: 12, weight: 700 })
    + text(79, 110, 'direct request', { size: 12, width: 120 })
    + node(160, 88, 82, 45, 'leaks', 'warning') + node(258, 88, 82, 45, 'blocked', 'output')
    + text(79, 168, 'quoted text', { size: 12, width: 120 })
    + node(160, 146, 82, 45, '?', 'neutral') + node(258, 146, 82, 45, '?', 'neutral')
    + text(79, 226, 'tool output', { size: 12, width: 120 })
    + node(160, 204, 82, 45, '?', 'neutral') + node(258, 204, 82, 45, '?', 'neutral')
    + text(180, 293, 'One repaired path != all paths covered', { weight: 700, tone: 'warning' })
    + text(180, 337, 'Adaptive testing targets the remaining question marks.', { size: 12, width: 325 })
    + text(180, 383, 'Release still needs residual-risk + utility evidence.', { size: 12, width: 325 }) };
}

function experiment() {
  let body = text(180, 22, 'Illustrative precommitted worthwhile effect: +1 point', { size: 12, tone: 'soft' });
  const scale = (value) => 70 + value / 2 * 240;
  for (const [index, item] of [{ label: 'Significant, but too small', range: [.2, .8], estimate: .5 }, { label: 'Worthwhile, but unsafe', range: [1.1, 1.8], estimate: 1.4 }].entries()) {
    const top = 72 + index * 138;
    body += text(18, top, item.label, { anchor: 'start', weight: 700 })
      + rule(70, top + 45, 310, top + 45) + rule(scale(1), top + 18, scale(1), top + 62, 'focus', true)
      + rule(scale(item.range[0]), top + 38, scale(item.range[1]), top + 38, 'input')
      + `<circle cx="${scale(item.estimate)}" cy="${top + 38}" r="4" style="fill:${ink('input')}"></circle>`
      + text(70, top + 80, '0', { size: 12 }) + text(scale(1), top + 80, '+1', { size: 12 }) + text(310, top + 80, '+2 points', { size: 12 });
  }
  body += text(180, 332, 'Second case: errors 2% > 1% guardrail', { size: 13, weight: 700, tone: 'warning' })
    + text(180, 374, 'Both HOLD. Invalid assignment? Interpret neither.', { size: 12, width: 325 });
  return { height: 405, body };
}

function monitoring() {
  return { height: 425, body:
    text(180, 22, 'Same request, different evidence clocks', { size: 12, tone: 'soft' })
    + text(18, 68, 'At 12:00: freshness can fail NOW', { anchor: 'start', weight: 700 })
    + node(25, 90, 310, 64, 'snapshot 11:40\nage 20 min > 2 min limit', 'warning')
    + text(18, 204, 'Later: outcome evidence must mature', { anchor: 'start', weight: 700 })
    + rule(35, 245, 325, 245)
    + box(35, 233, 290, 24, 'state', true)
    + text(35, 281, 'day 0', { anchor: 'start', size: 12 }) + text(325, 281, 'day 7', { anchor: 'end', size: 12 })
    + text(180, 324, 'Mature label + request ID + join coverage', { size: 13, width: 325, weight: 700 })
    + text(180, 378, 'Score drift alone does not establish a quality change.', { size: 12, width: 325 }) };
}

function rankingBudget() {
  let body = text(180, 22, 'Illustrative counts, not measured stage latencies', { size: 12, tone: 'soft' });
  for (const [index, [label, count, width, tone]] of [
    ['retrieve', '10,000 candidates', 310, 'input'],
    ['pre-rank', '1,000 candidates', 230, 'state'],
    ['final rank', '100 candidates', 150, 'output'],
  ].entries()) {
    const top = 67 + index * 87;
    body += node(180 - width / 2, top, width, 59, `${label}\n${count}`, tone);
    if (index < 2) body += arrow(180, top + 59, 180, top + 83);
  }
  body += text(180, 348, 'Widths show narrowing, not a linear count scale.', { size: 12, width: 325 })
    + text(180, 391, 'Measure request p95 < 80 ms and cost <= $0.08 / 1K.', { size: 12, width: 325 })
    + text(180, 437, 'Overload: fewer candidates or a bounded fallback.', { size: 12, width: 325 });
  return { height: 465, body };
}

function commitRace() {
  return { height: 405, body:
    text(180, 22, 'One run R, two attempts, expected version 0', { size: 12, tone: 'soft' })
    + node(15, 53, 150, 61, 'attempt A\nartifact A', 'input') + node(195, 53, 150, 61, 'attempt B\nartifact B', 'input')
    + arrow(90, 114, 90, 148, 'output') + arrow(270, 114, 270, 210, 'warning')
    + node(15, 152, 150, 61, 'A commits v0 -> v1\nartifact A wins', 'output')
    + node(195, 214, 150, 61, 'B expects v0\nactual v1: reject', 'warning')
    + text(180, 312, 'B cannot overwrite A\nits attempt evidence remains', { width: 320, weight: 700 })
    + text(180, 367, 'Retry deployment key K -> same D1; recheck writer epoch.', { width: 325, size: 12 }) };
}

function scheduler() {
  let body = text(180, 22, 'Toy units: KV pages persist; token slots refresh', { size: 12, tone: 'soft' })
    + text(18, 65, 'PERSISTENT KV POOL: 12 PAGES', { anchor: 'start', weight: 700 });
  for (let page = 0; page < 12; page++) body += box(25 + page * 26, 84, 23, 30, page < 5 ? 'input' : page < 9 ? 'state' : 'neutral')
    + text(36.5 + page * 26, 104, page < 5 ? 'A' : page < 9 ? 'B' : '-', { size: 12 });
  body += text(180, 145, 'A: 5 reserved + B: 4 reserved + 3 free', { size: 12, width: 325 })
    + text(18, 202, 'ONE ITERATION: 8 TOKEN-WORK SLOTS', { anchor: 'start', size: 13, weight: 700 });
  for (let slot = 0; slot < 8; slot++) body += box(25 + slot * 39, 224, 35, 30, slot < 3 ? 'focus' : 'output')
    + text(42.5 + slot * 39, 244, slot < 3 ? 'D' : 'P', { size: 12 });
  body += text(180, 288, '3 decode slots first; 5 bounded prefill slots remain', { size: 12, width: 325 })
    + text(180, 335, 'A stops -> 5 KV pages released -> 8 pages free', { size: 12, width: 325, weight: 700 })
    + text(180, 385, 'KV admission and iteration work are different budgets.', { size: 12, width: 325 });
  return { height: 420, body };
}

function frameAge() {
  return { height: 480, body:
    text(180, 22, 'One mapped clock; the old frame keeps its capture time', { size: 12, tone: 'soft' })
    + node(25, 55, 310, 52, 'capture 10.00 -> arrival 14.20 -> question 14.30', 'input')
    + text(20, 153, 'ACTUAL AGE: 14.30 - 10.00 = 4.30 s', { anchor: 'start', size: 13, weight: 700 })
    + box(25, 173, 310, 25, 'warning')
    + text(20, 240, 'ARRIVAL AGE: 14.30 - 14.20 = 0.10 s', { anchor: 'start', size: 13, weight: 700 })
    + box(25, 260, 310 * .1 / 4.3, 25, 'input')
    + text(180, 320, 'Same seconds-per-unit scale; arrival is not capture.', { size: 12, width: 325 })
    + node(25, 356, 310, 88, 'Replacement snapshot: a NEW decision\nage bound 0.068 s; revalidate action', 'output') };
}

function reclaim() {
  let body = text(180, 22, 'Quarter-hours and current nodes are different ledgers', { size: 12, tone: 'soft' })
    + text(18, 68, '248,832 usable accelerator-hours', { anchor: 'start', weight: 700 })
    + box(25, 89, 310 * .65, 30, 'state') + box(25 + 310 * .65, 89, 310 * .35, 30, 'focus')
    + text(120, 110, '65% protected', { size: 12 }) + text(282, 110, '35% offline', { size: 12 })
    + text(18, 178, 'INSTANTANEOUS TOY: 8 NODES', { anchor: 'start', size: 13, weight: 700 });
  for (let index = 0; index < 8; index++) {
    body += box(25 + index * 39, 200, 35, 31, index < 6 ? 'output' : 'focus')
      + text(42.5 + index * 39, 221, index < 6 ? 'S' : 'O', { size: 12 })
      + box(25 + index * 39, 309, 35, 31, 'output') + text(42.5 + index * 39, 330, 'S', { size: 12 });
  }
  body += text(180, 273, 'Pressure -> checkpoint + pause offline work', { size: 12, width: 325, weight: 700 })
    + text(180, 381, 'S = serving; O = admitted offline work', { size: 12 })
    + text(180, 423, 'Offline nodes return; recovery reserve is not free capacity.', { size: 12, width: 325 });
  return { height: 455, body };
}

function interference() {
  return { height: 405, body:
    node(15, 25, 150, 65, 'TREATMENT\nnew ranking policy', 'input')
    + node(195, 25, 150, 65, 'CONTROL\ncurrent policy', 'state')
    + arrow(90, 90, 135, 141) + arrow(270, 90, 225, 141)
    + node(45, 145, 270, 65, 'ONE shared creator pool\nboth cells allocate opportunity', 'focus')
    + arrow(180, 210, 180, 252)
    + node(45, 256, 270, 65, 'Weeks later: changed supply\ndirection not assumed', 'neutral')
    + arrow(135, 321, 90, 372, 'warning') + arrow(225, 321, 270, 372, 'warning')
    + text(90, 395, 'future treatment', { width: 150, size: 12 })
    + text(270, 395, 'future control', { width: 150, size: 12 }) };
}

function shelves() {
  let body = text(180, 22, 'Toy whole-page utility, not measured listener value', { size: 12, tone: 'soft' });
  for (const [index, [label, songs, score]] of [['A', [1, 2], 6], ['B', [1, 3], 5], ['C', [4, 5], 4]].entries()) {
    const top = 61 + index * 78;
    body += box(22, top, 316, 61, index === 1 ? 'state' : 'input')
      + text(47, top + 36, label, { weight: 700, size: 18 })
      + node(80, top + 12, 72, 36, `song ${songs[0]}`, songs[0] === 1 ? 'focus' : 'neutral')
      + node(165, top + 12, 72, 36, `song ${songs[1]}`)
      + text(286, top + 36, `score ${score}`, { size: 12 });
  }
  body += text(180, 326, 'A + B: 6 + 5 - 4 repeated-song penalty = 7', { size: 12, width: 325 })
    + text(180, 370, 'A + C = 10 > B + C = 9', { weight: 700, tone: 'output' })
    + text(180, 412, 'Choose intact A and C; display order is a separate choice.', { size: 12, width: 325 });
  return { height: 445, body };
}

function exposed() {
  let body = text(180, 22, 'Illustrative: 1,000 unique candidates; 900 eligible', { size: 12, tone: 'soft' })
    + text(180, 65, 'Rank 900, but display only 10', { weight: 700 });
  for (let position = 0; position < 10; position++) {
    body += box(25 + (position % 5) * 63, 91 + Math.floor(position / 5) * 42, 56, 32, 'output')
      + text(53 + (position % 5) * 63, 112 + Math.floor(position / 5) * 42, position + 1, { size: 13 });
  }
  body += text(180, 201, 'exposure IDs -> mature observed outcomes', { size: 12, width: 325, tone: 'output' })
    + node(25, 242, 310, 68, '890 eligible but unshown\noutcomes UNKNOWN, not negative', 'warning')
    + text(180, 355, 'Training retains policy, support, and selection metadata.', { size: 12, width: 325 })
    + text(180, 400, 'Only shown items provide exposure outcomes here.', { size: 12, width: 325 });
  return { height: 430, body };
}

function stairPaths() {
  const build = (remaining) => remaining === 0 ? [[]] : [1, 2].filter((step) => step <= remaining)
    .flatMap((step) => build(remaining - step).map((suffix) => [step, ...suffix]));
  const paths = build(5).sort((left, right) => left.at(-1) - right.at(-1));
  assert.equal(paths.length, 8);
  assert.equal(paths.filter((steps) => steps.at(-1) === 1).length, 5);
  let body = text(180, 22, 'Every row travels exactly five steps', { size: 12, tone: 'soft' });
  for (const [index, steps] of paths.entries()) {
    const top = 60 + index * 36 + (index >= 5 ? 43 : 0);
    let horizontal = 105;
    for (const [move, length] of steps.entries()) {
      const tone = move === steps.length - 1 ? 'focus' : 'input';
      body += box(horizontal, top, length * 30 - 3, 25, tone)
        + text(horizontal + (length * 30 - 3) / 2, top + 18, length, { size: 12 });
      horizontal += length * 30;
    }
    body += text(73, top + 18, `${index + 1}`, { size: 12 });
  }
  body += text(180, 257, '5 end in +1; 3 end in +2', { size: 13, weight: 700 })
    + text(180, 442, 'Disjoint final moves: 5 + 3 = 8', { size: 14, weight: 700 });
  return { height: 465, body };
}

function prerequisite() {
  return { height: 350, body:
    node(140, 20, 80, 46, '0', 'output')
    + arrow(151, 66, 95, 116) + arrow(209, 66, 265, 116)
    + node(55, 120, 80, 46, '1', 'input') + node(225, 120, 80, 46, '2', 'input')
    + arrow(95, 166, 151, 216) + arrow(265, 166, 209, 216)
    + node(140, 220, 80, 46, '3', 'focus')
    + text(180, 305, 'Course 3 unmet requirements: 2 -> 1 -> 0', { size: 12, width: 325, weight: 700 })
    + text(180, 338, 'Only after BOTH parents are emitted is 3 ready.', { size: 12, width: 325 }) };
}

function houses() {
  let body = text(180, 22, 'Evaluate house 3; neighbors cannot both be taken', { size: 12, tone: 'soft' });
  for (const [index, value] of [2, 7, 9, 3].entries()) {
    body += node(22 + index * 85, 63, 61, 56, value, index === 3 ? 'focus' : 'input')
      + text(52.5 + index * 85, 145, `index ${index}`, { size: 12 });
  }
  body += rule(52.5, 173, 222.5, 173, 'output')
    + text(180, 209, 'SKIP house 3: keep houses 0 + 2 = 11', { size: 13, width: 325, weight: 700 })
    + rule(137.5, 251, 307.5, 251, 'focus')
    + text(180, 287, 'TAKE house 3: best through 1 is 7; 7 + 3 = 10', { size: 12, width: 325 })
    + text(180, 341, 'max(11, 10) = 11; never combine adjacent 2 and 3', { size: 12, width: 325, weight: 700 });
  return { height: 375, body };
}

function lcsCells() {
  const matrix = [[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 2]];
  let body = text(180, 22, 'Rows: prefix of AB; columns: prefix of ACB', { size: 12, tone: 'soft' });
  for (const [column, label] of ['', 'A', 'C', 'B'].entries()) body += text(79 + column * 70, 64, label || 'empty', { size: 12 });
  for (const [row, values] of matrix.entries()) {
    body += text(8, 106 + row * 67, ['', 'A', 'B'][row] || 'empty', { size: 12, anchor: 'start', width: 40 });
    for (const [column, value] of values.entries()) {
      body += node(52 + column * 70, 82 + row * 67, 54, 45, value, row === 2 && column >= 2 ? 'focus' : 'neutral');
    }
  }
  body += arrow(246, 171, 289, 216, 'output')
    + text(180, 304, 'B = B: diagonal 1 + 1 -> 2', { size: 13, weight: 700, tone: 'output' })
    + text(180, 348, 'B != C: max(up 1, left 1) -> 1', { size: 13, weight: 700 });
  return { height: 380, body };
}

function dsuCompression() {
  const point = (horizontal, vertical, label, tone = 'input') => `<circle cx="${horizontal}" cy="${vertical}" r="16" style="fill:var(--viz-${tone}-bg);stroke:${ink(tone)}"></circle>${text(horizontal, vertical + 5, label)}`;
  let body = text(180, 22, 'The accepted problem graph NEVER changes here', { size: 12, tone: 'soft' });
  for (let index = 0; index < 3; index++) body += rule(60 + index * 80 + 16, 67, 60 + (index + 1) * 80 - 16, 67);
  for (let index = 0; index < 4; index++) body += point(60 + index * 80, 67, index + 1);
  body += text(180, 121, 'DSU parents BEFORE find(4)', { size: 13, weight: 700 })
    + arrow(83, 211, 165, 168) + arrow(257, 211, 195, 168) + arrow(270, 264, 270, 237)
    + point(180, 160, 1) + point(70, 220, 2) + point(270, 220, 3) + point(270, 280, 4, 'focus')
    + text(180, 331, 'AFTER: only parent(4) changes from 3 to 1', { size: 12, width: 325, weight: 700 })
    + arrow(84, 423, 166, 379) + arrow(197, 423, 184, 382) + arrow(286, 423, 197, 377, 'focus')
    + point(180, 365, 1) + point(70, 435, 2) + point(200, 435, 3) + point(300, 435, 4, 'focus');
  return { height: 470, body };
}

function subsetOrder() {
  const positions = [
    [180, 22, '[]'], [60, 106, '[1]'], [180, 106, '[2]'], [300, 106, '[3]'],
    [60, 210, '[1,2]'], [180, 210, '[1,3]'], [300, 210, '[2,3]'], [60, 314, '[1,2,3]'],
  ];
  let body = '';
  for (const [from, to] of [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 6], [4, 7]]) {
    const start = positions[from];
    const end = positions[to];
    body += arrow(start[0], start[1] + 45, end[0], end[1]);
  }
  for (const [horizontal, vertical, label] of positions) body += node(horizontal - 46, vertical, 92, 45, label, 'input');
  body += node(166, 310, 178, 62, '[2,1] REJECTED\nindex would go backward', 'warning')
    + text(180, 418, 'Increasing indices emit each subset once, not every order.', { size: 12, width: 325 });
  return { height: 450, body };
}

function encoderMemory() {
  let body = node(25, 24, 310, 62, 'Encode the complete source ONCE\ncross-attention K/V stay fixed', 'input');
  for (const [index, prefix] of ['<start>', '<start> A', '<start> A B'].entries()) {
    const top = 134 + index * 82;
    body += rule(38, 86, 38, top + 25, 'input') + arrow(38, top + 25, 82, top + 25, 'input')
      + node(86, top, 248, 53, `decode ${index + 1}: ${prefix}`, 'state');
  }
  body += text(180, 425, 'The output prefix and decoder cache grow; source K/V do not.', { size: 12, width: 325 });
  return { height: 460, body };
}

function rankerSlices() {
  let body = text(180, 22, 'Illustrative fixed slices; head/tail weights 90% / 10%', { size: 12, tone: 'soft' });
  for (const [index, [label, values]] of [['HEAD', [.8, .9]], ['TAIL', [.8, .7]]].entries()) {
    const top = 72 + index * 138;
    body += text(18, top, label, { anchor: 'start', weight: 700 });
    for (const [candidate, value] of values.entries()) {
      body += text(35, top + 36 + candidate * 37, candidate ? 'B' : 'A', { size: 13 })
        + box(60, top + 17 + candidate * 37, 260 * value, 26, candidate ? 'focus' : 'input')
        + text(330, top + 36 + candidate * 37, value.toFixed(2), { size: 12 });
    }
    if (index === 1) body += rule(60 + 260 * .78, top + 10, 60 + 260 * .78, top + 84, 'warning', true)
      + text(60 + 260 * .78, top + 110, 'floor 0.78', { size: 12, tone: 'warning' });
  }
  body += text(180, 377, 'Weighted: A = 0.80; B = 0.88', { size: 14, weight: 700 })
    + text(180, 418, 'B tail 0.70 < 0.78 -> HOLD', { weight: 700, tone: 'warning' });
  return { height: 445, body };
}

function dataLineage() {
  return comparison({ example: 'Illustrative stable IDs; rejection evidence is retained', left: 'R17: ACCEPTED', right: 'R18: REJECTED', rows: [
    { label: 'Versioned gate', left: 'policy permits', right: 'policy denies', rightTone: 'warning' },
    { label: 'Data manifest', left: 'S1 -> M1', right: 'reason retained\nno shard membership', rightTone: 'warning' },
    { label: 'Before checkpoint C200', left: 'consumption log: R17 read', right: 'not consumed', rightTone: 'neutral' },
  ] });
}

function ganUpdates() {
  return { height: 430, body:
    text(180, 23, '1. DISCRIMINATOR STEP', { weight: 700 })
    + node(18, 54, 126, 67, 'G produces fake\nG NOT updated', 'neutral')
    + node(216, 54, 126, 67, 'D sees real + fake\nUPDATE D', 'focus')
    + arrow(144, 88, 216, 88)
    + text(180, 145, 'fake is detached: no gradient back into G', { size: 12, width: 325 })
    + rule(18, 180, 342, 180)
    + text(180, 223, '2. GENERATOR STEP', { weight: 700 })
    + node(18, 252, 126, 67, 'G produces fake\nUPDATE G', 'focus')
    + node(216, 252, 126, 67, 'D evaluates fake\nD NOT updated', 'neutral')
    + arrow(144, 285, 216, 285)
    + rule(278, 319, 278, 349, 'focus', true) + rule(278, 349, 81, 349, 'focus', true)
    + arrow(81, 349, 81, 319, 'focus')
    + text(180, 383, 'Gradient passes through D into G.', { size: 13, weight: 700 })
    + text(180, 411, 'Frozen parameters != a detached computation', { size: 12, width: 325 }) };
}

function loraGradients() {
  return { height: 470, body:
    text(180, 22, 'At initialization: A random, B = 0; scale s = alpha/r', { size: 12, tone: 'soft' })
    + node(20, 58, 140, 62, 'Frozen base W\nWx', 'input')
    + node(200, 58, 140, 62, 'Adapter\nsBAx = 0', 'state')
    + arrow(90, 120, 145, 163) + arrow(270, 120, 215, 163)
    + node(80, 167, 200, 53, 'Initial y = Wx', 'output')
    + text(180, 267, 'First backward pass; g = dL/dy', { weight: 700 })
    + node(20, 291, 320, 55, 'dL/dA = s B^T g x^T = 0', 'warning')
    + node(20, 367, 320, 55, 'dL/dB = s g (Ax)^T\ngenerally nonzero', 'focus')
    + text(180, 457, 'W is not updated; input gradients can still flow.', { size: 12, width: 325 }) };
}

function autograd() {
  return { height: 430, body:
    node(125, 25, 110, 53, 'x = 2', 'input')
    + arrow(180, 78, 180, 105)
    + node(100, 109, 160, 62, 'u = 3x = 6\nshared node', 'state')
    + arrow(125, 171, 90, 217) + arrow(235, 171, 270, 217)
    + node(15, 221, 150, 61, 'a = u^2 = 36', 'neutral')
    + node(195, 221, 150, 61, 'b = 2u = 12', 'neutral')
    + arrow(90, 282, 145, 325) + arrow(270, 282, 215, 325)
    + node(95, 329, 170, 53, 'L = a + b = 48', 'output')
    + rule(40, 221, 40, 140, 'focus', true) + arrow(40, 140, 100, 140, 'focus')
    + rule(320, 221, 320, 140, 'focus', true) + arrow(320, 140, 260, 140, 'focus')
    + text(50, 200, '+12', { size: 13, weight: 700, tone: 'focus' })
    + text(310, 200, '+2', { size: 13, weight: 700, tone: 'focus' })
    + rule(100, 130, 85, 130, 'focus', true) + rule(85, 130, 85, 51, 'focus', true)
    + arrow(85, 51, 125, 51, 'focus')
    + text(43, 82, '14 x 3\n= 42', { size: 12, width: 75, tone: 'focus', weight: 700 })
    + text(180, 406, 'At u: 12 + 2 = 14; propagate once: 14 x 3 = 42', { size: 12, width: 325, weight: 700 }) };
}

function trainingGate() {
  return { height: 395, body:
    text(180, 22, 'Illustrative known 32-example fitting check', { size: 12, tone: 'soft' })
    + text(18, 67, 'Loss stays at 1.10 across updates', { anchor: 'start', weight: 700 })
    + rule(35, 136, 325, 136, 'warning')
    + text(180, 167, 'same batch / no regularization', { size: 12 })
    + node(18, 205, 150, 68, 'gradient norm\n0.8: nonzero', 'input')
    + node(192, 205, 150, 68, 'max weight change\n0: unchanged', 'warning')
    + text(180, 318, 'Inspect step, learning rate, and optimizer-bound parameters.', { size: 12, width: 325, weight: 700 })
    + text(180, 370, 'Recheck the same batch before full-data training.', { size: 12, width: 325 }) };
}

function rrfSurvival() {
  return comparison({ example: 'Exact-title A; RRF offset 60; unprotected cap = 1', left: 'CANDIDATE A', right: 'CANDIDATE B', rows: [
    { label: 'Lexical / semantic ranks', left: '1 / absent', right: '2 / 1', rightTone: 'input' },
    { label: 'Reciprocal-rank fusion score', left: '1/61 = 0.01639', right: '1/62 + 1/61\n= 0.03252', rightTone: 'focus' },
    { label: 'After cap; later reranking', left: 'REMOVED\ncannot be recovered', leftTone: 'warning', right: 'survives to reranker' },
  ] });
}

function inferenceCost() {
  return { height: 430, body:
    text(180, 22, 'Toy cost units per successful request; fixed task mix', { size: 12, tone: 'soft' })
    + text(20, 66, 'BASELINE: 10 units', { anchor: 'start', weight: 700 })
    + box(25, 86, 310, 29, 'input')
    + text(20, 162, 'CHANGE 1: 8 units; quality gates pass', { anchor: 'start', size: 13, weight: 700 })
    + box(25, 182, 248, 29, 'output')
    + text(20, 258, 'CHANGE 2: 6 units; tail quality fails', { anchor: 'start', size: 13, weight: 700 })
    + box(25, 278, 186, 29, 'warning', true)
    + text(180, 340, 'Reject change 2. Current baseline stays 8, not 6.', { size: 12, width: 325, weight: 700 })
    + text(180, 387, '10 / 8 = 1.25x cost reduction; no 10x claim.', { size: 12, width: 325 }) };
}

function shrinkage() {
  let body = text(180, 22, 'Fixed preconditioner; not the full Adam recurrence', { size: 12, tone: 'soft', width: 335 })
    + text(180, 64, 'Same start: theta = 1', { weight: 700 })
    + text(180, 89, 'alpha = 0.01; lambda = 0.1', { size: 12 });
  for (const [section, method] of ['Adam + coupled L2', 'AdamW: decoupled'].entries()) {
    const top = 130 + section * 180;
    body += text(18, top, method, { anchor: 'start', weight: 700 });
    for (const [index, scale] of [1, 10].entries()) {
      const vertical = top + 44 + index * 64;
      const amount = section ? .001 : .001 / scale;
      const end = 320 - amount * 250000;
      body += text(18, vertical - 12, `${index ? 'B' : 'A'}: s = ${scale}`, { anchor: 'start', size: 12 })
        + rule(70, vertical, 320, vertical, 'neutral', true)
        + rule(320, vertical - 10, 320, vertical + 10)
        + `<g data-shrink="${amount}" data-start="320" data-end="${end}">${arrow(320, vertical, end, vertical, section ? 'output' : 'focus')}</g>`
        + text(195, vertical + 26, `shrink ${amount}; end ${1 - amount}`, { size: 12 });
    }
  }
  body += text(180, 493, 'Every arrow uses the same parameter-distance scale.', { size: 12, width: 330 });
  return { height: 515, body };
}

function patchHandoff() {
  return { height: 460, body: text(90, 24, 'ENGINEER', { weight: 700 }) + text(270, 24, 'AGENT', { weight: 700 })
    + rule(180, 43, 180, 405, 'state', true)
    + node(12, 54, 150, 76, 'Invariant:\nexpired token denied\nallowed: auth.py', 'input')
    + arrow(162, 92, 198, 92)
    + node(198, 54, 150, 76, 'Bounded patch\nfocused tests\nactual diff', 'focus')
    + arrow(273, 130, 273, 160)
    + node(198, 164, 150, 73, 'Return evidence\nnot acceptance', 'state')
    + arrow(198, 200, 162, 200)
    + node(12, 164, 150, 73, 'Independent test\nexpiry == now\nnot just earlier', 'input')
    + arrow(87, 237, 87, 275)
    + node(12, 279, 150, 80, 'Counterexample fails\nRETURN for revision', 'warning')
    + arrow(162, 319, 198, 319, 'warning')
    + node(198, 279, 150, 80, 'Repair same scope\nrerun evidence', 'focus')
    + text(180, 407, 'Acceptance stays with the engineer.', { weight: 700 })
    + text(180, 437, 'Fictional patch; passing tests alone is not acceptance.', { size: 12, width: 330 }) };
}

function metricVeto() {
  const rows = [
    { label: 'Completion gain: lower bound', value: 4.66, limit: 2, maximum: 6, unit: ' points', pass: true },
    { label: 'Cost per completed task', value: 1.97, limit: 2, maximum: 2.5, unit: ' dollars', pass: true },
    { label: 'Latency', value: 2.1, limit: 2, maximum: 2.5, unit: ' seconds', pass: false },
  ];
  let body = text(180, 22, 'Declared gates; each row has its own scale', { size: 12, tone: 'soft' });
  for (const [index, row] of rows.entries()) {
    const top = 60 + index * 112;
    const position = (value) => 30 + 300 * value / row.maximum;
    body += text(18, top, row.label, { anchor: 'start', weight: 700 })
      + rule(30, top + 38, 330, top + 38)
      + rule(position(row.limit), top + 23, position(row.limit), top + 55, 'state', true)
      + `<circle cx="${position(row.value)}" cy="${top + 38}" r="6" style="fill:${ink(row.pass ? 'output' : 'warning')}"></circle>`
      + text(30, top + 82, `observed ${row.value}${row.unit}`, { anchor: 'start', size: 12 })
      + text(330, top + 82, `${row.pass ? 'PASS' : 'FAIL'}; limit ${row.limit}`, { anchor: 'end', size: 12, tone: row.pass ? 'output' : 'warning' });
  }
  body += node(30, 407, 300, 58, 'AND gate: 2 passes + 1 failure\nHOLD full release', 'warning')
    + text(180, 499, 'Retries and edits can diagnose; they cannot override.', { size: 12, width: 330 });
  return { height: 525, body };
}

function densityNeighborhoods() {
  const positions = { A: -.1, B: 0, C: .1, D: .2, E: .5 };
  const horizontal = (value) => 32 + (value + .5) * 211.4;
  let body = text(180, 22, 'radius 0.31; minPts 4; include the point itself', { size: 12, tone: 'soft', width: 330 })
    + text(180, 62, 'Zoom: actual one-dimensional distances', { weight: 700 });
  for (const [index, center] of ['D', 'E'].entries()) {
    const top = 125 + index * 147;
    body += text(18, top - 25, `${center}'s closed neighborhood`, { anchor: 'start', size: 12, weight: 700 })
      + box(horizontal(positions[center] - .31), top - 8, .62 * 211.4, 30, index ? 'warning' : 'focus')
      + rule(32, top + 7, 328, top + 7);
    for (const [name, value] of Object.entries(positions)) {
      const included = Math.abs(value - positions[center]) <= .31;
      body += `<circle cx="${horizontal(value)}" cy="${top + 7}" r="4" style="fill:${ink(included ? index ? 'warning' : 'focus' : 'neutral')}"></circle>`
        + text(horizontal(value), top + (name === 'B' || name === 'D' ? 63 : 43), name, { size: 12, width: 20 });
    }
    body += text(180, top + 91, index ? 'D + E = 2: BORDER; cannot extend' : 'A + B + C + D + E = 5: CORE', { size: 12, weight: 700 });
  }
  body += text(180, 407, 'A=-0.1; B=0; C=0.1; D=0.2; E=0.5', { size: 12, width: 335 })
    + text(180, 446, 'F=2: nearest point E is 1.5 away', { size: 13 })
    + text(180, 477, 'F counts only itself: 1; NOISE', { weight: 700, tone: 'warning' });
  return { height: 500, body };
}

function authorityPorts() {
  let body = node(30, 20, 300, 60, 'Permission-aware context broker\nrepository + current buffer', 'input')
    + rule(180, 80, 180, 96) + rule(10, 96, 350, 96)
    + rule(10, 96, 10, 255) + rule(350, 96, 350, 393);
  for (const [index, mode] of ['inline -> text', 'chat -> answer', 'review -> comments'].entries()) {
    const top = 111 + index * 62;
    body += arrow(10, top + 20, 30, top + 20)
      + node(30, top, 300, 40, `${mode}; no tool execution`, 'neutral');
  }
  body += box(14, 319, 332, 132, 'state', true)
    + text(180, 343, 'ISOLATED AGENT AUTHORITY', { size: 12, weight: 700 })
    + node(30, 358, 300, 70, 'Allowed paths + tools + time + cost\nedit / test -> patch + evidence', 'focus')
    + arrow(350, 393, 330, 393, 'state')
    + text(180, 491, 'DEVELOPER: apply, revise, or reject', { weight: 700 })
    + arrow(180, 451, 180, 472, 'output');
  return { height: 520, body };
}

function jointShifts() {
  let body = text(180, 22, 'Toy joint probabilities P(X, Y); each panel sums to 1', { size: 12, tone: 'soft', width: 335 });
  const matrix = (left, top, label, values) => {
    let markup = text(left + 66, top, label, { weight: 700, size: 13 })
      + text(left + 42, top + 27, 'Y=0', { size: 12 }) + text(left + 96, top + 27, 'Y=1', { size: 12 });
    for (const [index, value] of values.entries()) {
      const row = Math.floor(index / 2);
      const column = index % 2;
      markup += box(left + 15 + column * 54, top + 38 + row * 37, 54, 37, row ? 'focus' : 'input')
        + text(left + 42 + column * 54, top + 62 + row * 37, value.toFixed(2), { size: 13 });
    }
    return markup;
  };
  const sections = [
    { title: 'COVARIATE: only P(X) changes', source: [.72, .08, .02, .18], target: [.18, .02, .08, .72], rule: 'P(Y=1 | X=0)=.1; P(Y=1 | X=1)=.9', foot: 'P(X=1): .2 -> .8; support overlaps' },
    { title: 'LABEL: only P(Y) changes', source: [.6, .05, .2, .15], target: [.15, .2, .05, .6], rule: 'P(X=1 | Y=0)=.25; P(X=1 | Y=1)=.75', foot: 'P(Y=1): .2 -> .8; identifiable statistic needed' },
  ];
  for (const [index, section] of sections.entries()) {
    const top = 66 + index * 241;
    body += text(180, top, section.title, { weight: 700, size: 13 })
      + matrix(33, top + 30, 'SOURCE', section.source) + matrix(207, top + 30, 'TARGET', section.target)
      + text(15, top + 105, 'X=0', { size: 12 }) + text(15, top + 142, 'X=1', { size: 12 })
      + text(180, top + 177, section.rule, { size: 12, width: 340 })
      + text(180, top + 207, section.foot, { size: 12, width: 340 });
  }
  body += text(180, 554, 'CONCEPT: same target inputs, incompatible labels', { size: 13, weight: 700, width: 335 })
    + node(25, 586, 310, 52, 'Both worlds: P(X=0)=P(X=1)=0.5', 'input')
    + node(15, 662, 150, 70, 'World A\nY = X', 'focus')
    + node(195, 662, 150, 70, 'World B\nY = 1 - X', 'warning')
    + text(180, 770, 'Unlabeled X cannot distinguish these worlds.', { size: 12 });
  return { height: 794, body };
}

function banditRules() {
  const means = [.62, .55, .35];
  const counts = [40, 4, 20];
  const draws = [.58, .78, .31];
  const deviations = [.06, .18, .09];
  const position = (value) => 35 + 280 * value;
  let body = text(180, 22, 'Same evidence: A .62/40; B .55/4; C .35/20', { size: 12, tone: 'soft', width: 335 })
    + text(18, 68, '1. EPSILON = 0.3: random branch', { anchor: 'start', size: 13, weight: 700 });
  let start = 35;
  for (const [index, probability] of [.8, .1, .1].entries()) {
    body += box(start, 91, probability * 280, 30, ['input', 'focus', 'state'][index])
      + text(start + probability * 140, 112, ['A', 'B', 'C'][index], { size: 12 });
    start += probability * 280;
  }
  body += text(180, 148, 'P(A)=0.7+0.3/3=0.8; P(B)=P(C)=0.1', { size: 12 })
    + text(18, 198, '2. UCB: mean + 0.2 sqrt(log(64) / n)', { anchor: 'start', size: 13, weight: 700 });
  for (const [index, mean] of means.entries()) {
    const vertical = 240 + index * 56;
    const upper = mean + .2 * Math.sqrt(Math.log(64) / counts[index]);
    body += text(18, vertical + 4, ['A', 'B', 'C'][index], { size: 12 })
      + rule(35, vertical, 315, vertical, 'neutral', true)
      + rule(position(mean), vertical, position(upper), vertical, 'focus')
      + `<circle cx="${position(mean)}" cy="${vertical}" r="4" style="fill:${ink('input')}"></circle>`
      + rule(position(upper), vertical - 8, position(upper), vertical + 8, 'focus')
      + text(330, vertical + 23, `index ${upper.toFixed(2)}${index === 1 ? ': choose B' : ''}`, { size: 12, anchor: 'end' });
  }
  body += text(18, 419, '3. THOMPSON: maximize one posterior draw', { anchor: 'start', size: 13, weight: 700 })
    + text(180, 449, 'Assumed Normal posteriors for this toy round', { size: 12, width: 330 });
  for (const [index, mean] of means.entries()) {
    const baseline = 520 + index * 83;
    const deviation = deviations[index];
    const points = Array.from({ length: 101 }, (_, step) => {
      const value = step / 100;
      const density = Math.exp(-.5 * ((value - mean) / deviation) ** 2) / (deviation * Math.sqrt(2 * Math.PI));
      return `${step ? 'L' : 'M'}${position(value)} ${baseline - density * 5}`;
    }).join('');
    body += text(18, baseline, ['A', 'B', 'C'][index], { size: 12 }) + rule(35, baseline, 315, baseline)
      + `<path d="${points}" style="fill:none;stroke:${ink(index === 1 ? 'focus' : 'input')};stroke-width:1.75"></path>`
      + rule(position(draws[index]), baseline - 34, position(draws[index]), baseline + 5, 'state', true)
      + text(330, baseline + 29, `draw ${draws[index]}${index === 1 ? ': choose B' : ''}`, { size: 12, anchor: 'end' });
  }
  body += text(180, 749, 'Toy posteriors are declared assumptions.', { size: 12, width: 335 })
    + text(180, 778, 'Later draws may choose a different arm.', { size: 12, width: 335 })
    + text(180, 807, 'Means and counts do not determine these curves.', { size: 12, width: 335 });
  return { height: 835, body };
}

function benchmarkBoundary() {
  return { height: 490, body: text(90, 24, 'DEVELOPMENT', { size: 13, weight: 700 }) + text(270, 24, 'CONFIRMATION', { size: 13, weight: 700 })
    + node(18, 63, 144, 78, 'Public set B\nscore + errors\nvisible to team', 'input')
    + arrow(90, 141, 90, 183)
    + node(18, 187, 144, 78, 'Select revised M2\nusing B feedback', 'focus')
    + rule(18, 226, 7, 226, 'focus') + rule(7, 226, 7, 102, 'focus') + arrow(7, 102, 18, 102, 'focus')
    + text(90, 312, 'B now shapes M2', { size: 12, width: 145 })
    + node(198, 63, 144, 78, 'Freeze selected M2\nbefore opening H', 'input')
    + arrow(270, 141, 270, 183)
    + node(198, 187, 144, 78, 'Fresh hidden H\nscore once\nrecord scope', 'output')
    + text(270, 312, 'No return arrow\ninto this selection', { size: 12, width: 145 })
    + rule(180, 45, 180, 359, 'state', true)
    + text(180, 401, 'Revising M2 after H uses H as development evidence.', { size: 13, width: 330 })
    + text(180, 449, 'Confirm the revised system on another independent source.', { size: 12, width: 330 }) };
}

function adaptationGates() {
  const position = (day) => 40 + day * 9;
  return { height: 548, body: text(180, 22, 'Toy policy limits; both shown on the same day axis', { size: 12, tone: 'soft', width: 330 })
    + text(18, 68, 'A: 7 days', { anchor: 'start', size: 13, weight: 700 })
    + box(40, 83, 7 * 9, 22, 'warning')
    + text(18, 143, 'B: 30 days; applicable current version', { anchor: 'start', size: 13, weight: 700 })
    + box(40, 158, 30 * 9, 22, 'output')
    + rule(position(14), 51, position(14), 195, 'focus', true)
    + text(position(14), 218, 'day 14', { size: 12, weight: 700 })
    + text(40, 218, '0', { size: 12 }) + text(310, 218, '30', { size: 12 })
    + node(25, 251, 310, 74, 'Counterexample: valid JSON\n{"eligible":false,"policy_version":"A"}', 'input')
    + node(15, 362, 150, 77, 'Schema check\nPASS', 'output')
    + node(195, 362, 150, 77, 'Current evidence\nFAIL: stale A', 'warning')
    + arrow(110, 325, 90, 362) + arrow(250, 325, 270, 362)
    + text(180, 486, 'HOLD: valid structure cannot repair wrong policy.', { size: 13, weight: 700, width: 330 })
    + text(180, 524, 'B permits day 14; authorization still applies.', { size: 12 }) };
}

function tuningMatrix() {
  let body = text(180, 22, 'Objective and update scope are independent choices', { size: 12, tone: 'soft', width: 330 })
    + text(150, 74, 'FULL', { size: 13, weight: 700 })
    + text(280, 74, 'LoRA', { size: 13, weight: 700 });
  const rows = [
    ['Domain text', 'next-token loss'],
    ['Prompt + ideal response', 'SFT loss'],
    ['Prompt + preferred / rejected', 'preference loss'],
  ];
  for (const [index, [data, objective]] of rows.entries()) {
    const top = 105 + index * 156;
    body += text(18, top, `${index + 1}. ${data}`, { anchor: 'start', size: 13, weight: 700, width: 330 })
      + text(18, top + 29, objective, { anchor: 'start', size: 12, tone: 'soft' })
      + node(92, top + 48, 116, 69, 'base W\nTRAINABLE', 'input')
      + node(222, top + 48, 116, 69, 'W FROZEN\nadapter A,B\nTRAINABLE', index === 1 ? 'focus' : 'state');
  }
  body += text(180, 606, 'Highlighted cell: SFT objective + LoRA scope', { size: 13, weight: 700 })
    + text(180, 644, 'Rows are alternatives, not a required three-stage sequence.', { size: 12, width: 330 });
  return { height: 674, body };
}

function evidenceExchange() {
  return { height: 595, body: text(180, 22, 'Fictional mock; observable evidence, not personality scores', { size: 12, tone: 'soft', width: 330 })
    + node(25, 62, 310, 62, 'Interviewer: overall completion is up.\nWould you ship?', 'input')
    + node(25, 151, 310, 62, 'Candidate: which users lose?\nWhat is the stop threshold?', 'focus')
    + node(25, 240, 310, 86, 'Interviewer: the high-risk slice fell.\nCandidate: hold that path; investigate.', 'input')
    + text(18, 349, 'OBSERVED', { anchor: 'start', size: 12, weight: 700, tone: 'output' })
    + text(18, 378, 'Scopes harm; updates under pushback; states a decision.', { anchor: 'start', size: 13, width: 325 })
    + text(18, 431, 'UNKNOWN', { anchor: 'start', size: 12, weight: 700, tone: 'state' })
    + text(18, 460, 'Delivery ownership and scientific depth were not tested.', { anchor: 'start', size: 13, width: 325 })
    + arrow(180, 496, 180, 520, 'focus')
    + text(180, 553, 'Next drill: explain a slice test and its uncertainty.', { size: 13, weight: 700, width: 325 }) };
}

function querySharing() {
  let body = text(180, 22, 'Brackets denote sharing; arrows preserve head identity', { size: 12, tone: 'soft', width: 335 });
  for (const [group, center] of [90, 270].entries()) {
    body += node(center - 75, 51, 150, 43, `K${group + 1} + V${group + 1} reused`, 'state')
      + rule(center, 94, center, 115, 'state', true)
      + rule(center - 65, 115, center + 65, 115, 'state', true)
      + rule(center - 65, 115, center - 65, 128, 'state', true)
      + rule(center + 65, 115, center + 65, 128, 'state', true);
  }
  for (let index = 0; index < 8; index++) {
    const horizontal = 6 + index * 45;
    body += node(horizontal, 141, 33, 37, `Q${index + 1}`, index < 4 ? 'input' : 'focus')
      + arrow(horizontal + 16.5, 178, horizontal + 16.5, 207)
      + node(horizontal, 211, 33, 37, `O${index + 1}`, 'output');
  }
  body += text(180, 287, 'Eight distinct queries -> eight output heads', { size: 13, weight: 700 })
    + text(180, 327, 'Head-dependent KV cache; fixed length and head size', { size: 12, width: 335 });
  for (const [index, [name, heads]] of [['MHA', 8], ['GQA', 2], ['MQA', 1]].entries()) {
    const vertical = 358 + index * 57;
    body += text(15, vertical + 18, name, { anchor: 'start', size: 12 })
      + box(65, vertical, heads * 30, 25, index ? 'focus' : 'input')
      + text(335, vertical + 18, `${heads}/8`, { anchor: 'end', size: 12 });
  }
  body += text(180, 537, 'GQA: one-quarter cache, not one-quarter of all compute.', { size: 12, width: 335 });
  return { height: 565, body };
}

function forwardChoices() {
  let body = text(180, 22, 'Fictional options from today; not three simultaneous spends', { size: 12, tone: 'soft', width: 335 })
    + text(180, 72, 'Past spend: recorded, excluded from this choice', { size: 13, weight: 700, width: 335 });
  for (const [index, [label, spend]] of [['Bounded test', 10], ['Continue', 100], ['Redirect', 100]].entries()) {
    const vertical = 126 + index * 91;
    body += text(18, vertical, label, { anchor: 'start', weight: 700 })
      + box(25, vertical + 20, 300, 24, 'neutral', true)
      + box(25, vertical + 20, spend * 3, 24, index === 0 ? 'focus' : 'input')
      + text(330, vertical + 66, `${spend} / 100 future resource units`, { anchor: 'end', size: 12 });
  }
  body += node(25, 428, 310, 106, 'Test owner: project lead; stop: day 7\nneed >=95/100 contract passes AND zero leaks\notherwise stop; no automatic renewal', 'focus')
    + text(180, 552, 'Returns remain uncertain for all three options.', { size: 13, weight: 700, width: 330 })
    + text(180, 590, 'The test buys decision-changing evidence, not a new quarter.', { size: 12, width: 330 });
  return { height: 622, body };
}

function contextTracks() {
  const horizontal = (length) => 38 + (length - 1) * 40;
  let body = text(180, 22, 'Relative context n; no measured usability threshold', { size: 12, tone: 'soft', width: 335 })
    + text(18, 68, '1. POSITION QUALITY', { anchor: 'start', weight: 700 })
    + text(18, 98, 'Toy trained span ends at n=2; beyond it is unvalidated.', { anchor: 'start', size: 12, width: 330 })
    + rule(38, 146, horizontal(2), 146, 'output')
    + rule(horizontal(2), 146, horizontal(8), 146, 'state', true)
    + text(220, 137, 'needs evaluation', { size: 12, tone: 'state', width: 170 });
  const lengths = [1, 2, 4, 8];
  for (const length of lengths) body += text(horizontal(length), 179, String(length), { size: 12 });
  for (const [index, title] of ['2. DENSE ATTENTION WORK', '3. LOGICAL KV BYTES'].entries()) {
    const top = 233 + index * 229;
    const baseline = top + 119;
    const maximum = index ? 8 : 64;
    body += text(18, top, title, { anchor: 'start', weight: 700 }) + rule(38, baseline, 318, baseline);
    const points = lengths.map((length) => [horizontal(length), baseline - (index ? length : length ** 2) / maximum * 78]);
    body += `<path d="${points.map(([x, y], position) => `${position ? 'L' : 'M'}${x} ${y}`).join('')}" style="fill:none;stroke:${ink(index ? 'input' : 'focus')};stroke-width:2"></path>`;
    for (const [position, length] of lengths.entries()) {
      body += `<circle cx="${points[position][0]}" cy="${points[position][1]}" r="3" style="fill:${ink(index ? 'input' : 'focus')}"></circle>`
        + text(horizontal(length), points[position][1] - 12, String(index ? length : length ** 2), { size: 12 })
        + text(horizontal(length), baseline + 24, String(length), { size: 12 });
    }
    body += text(180, baseline + 57, index ? 'Paging removes allocation waste, not this linear growth.' : 'FlashAttention removes score materialization, not n^2 work.', { size: 12, width: 335 });
  }
  body += node(20, 716, 320, 82, 'USABLE means all three hold:\nvalidated positions AND executable work\nAND affordable serving state', 'output');
  return { height: 822, body };
}

function advantageSteps() {
  let body = text(180, 22, 'Toy 3-action softmax: initial logits (0,0,0)', { size: 12, tone: 'soft', width: 335 })
    + text(180, 60, 'Sampled A; baseline b=3; ascent step eta=0.1', { size: 12, width: 335 });
  for (const [index, advantage] of [2, 0, -2].entries()) {
    const top = 111 + index * 154;
    const logits = [advantage * .1 * 2 / 3, -advantage * .1 / 3, -advantage * .1 / 3];
    const denominator = logits.reduce((sum, value) => sum + Math.exp(value), 0);
    const probabilities = logits.map((value) => Math.exp(value) / denominator);
    body += text(18, top, `Return ${advantage + 3} - baseline 3 = ${advantage > 0 ? '+' : ''}${advantage}`, { anchor: 'start', weight: 700 });
    let horizontal = 30;
    for (const [action, probability] of probabilities.entries()) {
      body += `<g data-policy-case="${advantage}" data-action="${action}" data-probability="${probability}">${box(horizontal, top + 27, probability * 300, 36, ['focus', 'input', 'state'][action])}</g>`
        + text(horizontal + probability * 150, top + 50, `${['A', 'B', 'C'][action]} ${(probability * 100).toFixed(1)}%`, { size: 12 });
      horizontal += probability * 300;
    }
    body += text(180, top + 96, advantage > 0 ? 'A rises above 1/3; B and C renormalize.' : advantage < 0 ? 'Return is positive, yet A falls below 1/3.' : 'This sample contributes zero, not the whole batch.', { size: 12, width: 335 });
  }
  body += text(180, 599, 'logit step = eta * advantage * (one_hot(A) - p)', { size: 12, width: 335 });
  return { height: 631, body };
}

function purchaseRelations() {
  let body = node(92, 24, 176, 62, 'Anchor printer P1\ntask: complete purchase', 'input');
  const items = [
    ['Co-purchase', 'ink C7\n50 baskets\ndenominator unknown'],
    ['Co-view', 'printer P2\npossible alternative'],
    ['Attributes', 'printer P3\nsimilar features'],
  ];
  for (const [index, [relation, label]] of items.entries()) {
    const horizontal = 5 + index * 120;
    body += arrow(180, 86, horizontal + 55, 144, index ? 'neutral' : 'focus')
      + text(horizontal + 55, 132, relation, { size: 12, width: 112, weight: 700 })
      + node(horizontal, 154, 110, 108, label, index ? 'neutral' : 'focus');
  }
  body += arrow(60, 262, 60, 300, 'focus')
    + node(8, 305, 344, 78, 'Does current catalog verify C7 fits P1?\nYES: complementary shelf\nNO / UNKNOWN: withhold C7', 'state')
    + text(180, 425, 'Observed association is not product compatibility.', { weight: 700, size: 13, width: 335 })
    + text(180, 466, 'P2 and P3 answer a different comparison task.', { size: 12, width: 335 });
  return { height: 496, body };
}

function proxyShift() {
  const horizontal = (value) => 51 + value * 27;
  const vertical = (value) => 301 - value * 42;
  let body = text(180, 22, 'Fictional outputs; possible failure, not an inevitable curve', { size: 12, tone: 'soft', width: 335 })
    + box(horizontal(3), vertical(5), 3 * 27, 5 * 42, 'neutral', true)
    + text(180, 61, 'Fresh human rubric score (0 to 5)', { size: 13, weight: 700 })
    + rule(51, 91, 51, 301) + rule(51, 301, 321, 301);
  for (const score of [0, 1, 3, 5]) body += text(35, vertical(score) + 4, String(score), { size: 12 });
  for (const score of [0, 3, 6, 9, 10]) body += text(horizontal(score), 325, String(score), { size: 12 });
  for (const [id, reward, human] of [['R1', 4, 4], ['R2', 6, 5], ['R3', 9, 1]]) {
    body += `<circle cx="${horizontal(reward)}" cy="${vertical(human)}" r="6" style="fill:${ink(id === 'R3' ? 'warning' : 'input')}"></circle>`
      + text(horizontal(reward) + (id === 'R2' ? 18 : 0), vertical(human) - 15, id, { size: 12, weight: 700 });
  }
  body += text(180, 359, 'Proxy reward (arbitrary units, 0 to 10)', { size: 12 })
    + text(180, 397, 'Dashed band: represented proxy-score range, 3 to 6', { size: 12, width: 335 })
    + node(25, 436, 310, 80, 'Optimized R3: proxy 9, fresh human score 1\npossible style shortcut; inspect it\nnew on-policy pairs -> revalidate', 'warning')
    + text(180, 553, 'A high proxy score outside coverage is not validation.', { size: 12, width: 335 });
  return { height: 580, body };
}

function legalClaims() {
  return { height: 682, body: text(180, 22, 'Fictional jurisdiction J; not legal advice', { size: 12, tone: 'soft' })
    + node(20, 57, 320, 95, 'Authority V2026-08-15 > section 4 > P-017\n"Notice must be filed within 10 days."\nP-042: "The clerk records receipt."', 'input')
    + node(20, 186, 320, 72, 'C1: notice deadline is 10 days.\nP-017 directly supplies the bound: SUPPORT', 'output')
    + node(20, 290, 320, 72, 'C2: receipt guarantees approval.\nP-042 says nothing about approval: HOLD', 'warning')
    + arrow(180, 152, 180, 182, 'output')
    + node(20, 397, 320, 138, 'Release only supported C1, IF\njurisdiction + effective-time checks pass\nAND required attorney approval is recorded', 'state')
    + text(180, 579, 'C2 is removed or escalated, not rescued by its citation.', { size: 13, weight: 700, width: 335 })
    + text(180, 630, 'Keep query, exact passages, version, verdicts, and reviewer.', { size: 12, width: 335 }) };
}

function chunkLineage() {
  return { height: 624, body: text(180, 22, 'Toy authorized request: what is the return window?', { size: 12, tone: 'soft', width: 335 })
    + node(25, 63, 310, 72, 'OFFLINE document D@v3\nc1: "Returns within 30 days."\nc2: "Contact support by email."', 'input')
    + arrow(180, 135, 180, 165)
    + node(25, 169, 310, 72, 'Same chunk IDs in both indexes\nlexical -> c1; dense -> c1,c2\nfusion -> {c1,c2}', 'state')
    + arrow(180, 241, 180, 271)
    + node(25, 275, 310, 72, 'ONLINE reranker selects c1\nc2 removed from context\nrequest authorization retained', 'focus')
    + arrow(180, 347, 180, 377)
    + node(25, 381, 310, 72, 'Answer: "30 days [D@v3:c1]"\nexact source version preserved', 'output')
    + text(180, 499, 'A later corpus v4 changes indexed evidence,', { size: 12, width: 330 })
    + text(180, 528, 'not necessarily the language model\'s weights.', { size: 12, width: 330 })
    + text(180, 574, 'A citation still needs an independent support check.', { size: 13, weight: 700, width: 330 }) };
}

function regularizationEvidence() {
  const training = [.9, .65, .4, .25, .15];
  const validation = [.92, .7, .5, .55, .7];
  const coordinate = (values) => values.map((value, index) => `${index ? 'L' : 'M'}${45 + index * 64} ${278 - value * 180}`).join('');
  return { height: 592, body: text(180, 22, 'Illustrative matched loss traces; same task and split', { size: 12, tone: 'soft', width: 335 })
    + text(18, 68, 'Loss', { size: 12, anchor: 'start' })
    + rule(45, 80, 45, 278) + rule(45, 278, 301, 278)
    + `<path d="${coordinate(training)}" style="fill:none;stroke:${ink('input')};stroke-width:2"></path>`
    + `<path d="${coordinate(validation)}" style="fill:none;stroke:${ink('warning')};stroke-width:2;stroke-dasharray:6 4"></path>`
    + text(180, 310, 'Training steps: 1 -> 5', { size: 12 })
    + text(180, 344, 'Solid: train; dashed: validation', { size: 12 })
    + node(20, 379, 320, 106, 'First verify split + train/eval modes.\nIf valid: test an early-stop checkpoint\non independent held-out evidence; result ?', 'focus')
    + text(18, 509, 'Both poor: check optimization/data before constraining.', { anchor: 'start', size: 12, width: 330 })
    + text(18, 548, 'Both good: more regularization need not help.', { anchor: 'start', size: 12, width: 330 }) };
}

function gradedEpisode() {
  return { height: 669, body: text(180, 22, 'Fictional episode E17; declared zero-unauthorized-attempt rule', { size: 12, tone: 'soft', width: 335 })
    + text(18, 70, 'POLICY OBSERVES', { anchor: 'start', size: 12, weight: 700 })
    + node(20, 87, 320, 66, 'Task: close own order 17\nscoped identity; no hidden grader answers', 'input')
    + text(18, 197, 'IMMUTABLE EVENT EVIDENCE', { anchor: 'start', size: 12, weight: 700 })
    + node(20, 215, 320, 76, 'e1: own order 17 closed; allowed\ne2: other-account read attempted; DENIED\nno forbidden side effect committed', 'focus')
    + arrow(180, 291, 180, 326)
    + text(18, 350, 'GRADER ONLY: hidden target + actual final state', { anchor: 'start', size: 12, weight: 700, width: 330 })
    + node(15, 386, 150, 100, 'Task outcome\norder 17 closed\nSUCCESS', 'output')
    + node(195, 386, 150, 100, 'Process hard gate\n1 forbidden attempt\nFAIL', 'warning')
    + text(180, 535, 'INELIGIBLE under this declared grader contract', { size: 13, weight: 700, tone: 'warning', width: 330 })
    + text(180, 580, 'Successful outcome cannot average away a hard failure.', { size: 12, width: 330 })
    + text(180, 620, 'A different contract could grade blocked attempts differently.', { size: 12, width: 330 }) };
}

function preferencePaths() {
  return { height: 695, body: text(180, 22, 'Toy stored pair x: which policy applies?', { size: 12, tone: 'soft' })
    + node(20, 51, 320, 70, 'Preferred yw: "Use current P2."\nRejected yl: "Use stale P1."\nsame fixed SFT reference on both paths', 'input')
    + text(90, 158, 'RLHF', { weight: 700 }) + text(270, 158, 'DPO', { weight: 700 })
    + arrow(100, 121, 90, 179) + arrow(260, 121, 270, 179)
    + node(15, 183, 150, 107, 'Fit reward model\nfrom stored pairs\nthen freeze scorer', 'state')
    + node(195, 183, 150, 107, 'Use stored yw,yl\npolicy TRAINABLE\nreference FROZEN', 'focus')
    + arrow(90, 290, 90, 308) + arrow(270, 290, 270, 308)
    + node(15, 312, 150, 96, 'Current policy\nFRESH rollouts\nreward + KL\nRL update', 'focus')
    + node(195, 312, 150, 96, 'Pairwise loss\nreference-relative\nlikelihood margin\nDIRECT update', 'output')
    + rule(15, 361, 5, 361, 'focus') + rule(5, 361, 5, 298, 'focus') + arrow(5, 298, 90, 298, 'focus')
    + text(180, 463, 'DPO margin: log(pi(yw)/ref(yw))', { size: 12, width: 335 })
    + text(180, 492, '- log(pi(yl)/ref(yl)); all conditional on x', { size: 12, width: 335 })
    + text(180, 535, 'Train with -log sigmoid(beta * margin).', { size: 13, weight: 700 })
    + text(180, 580, 'No separate reward-model fit or rollout loop in DPO.', { size: 12, width: 335 })
    + text(180, 628, 'Reference-relative does not mean unconstrained.', { size: 12, width: 335 }) };
}

function retrievalFanout() {
  let body = text(180, 22, 'Toy corpus N=6; retained K=3', { size: 12, tone: 'soft' })
    + text(180, 63, 'OFFLINE: six reusable item vectors', { weight: 700 });
  for (let index = 0; index < 6; index++) body += node(9 + index * 58, 84, 51, 43, `I${index + 1}`, [1, 3, 5].includes(index) ? 'focus' : 'neutral');
  body += node(25, 166, 310, 62, 'Query vector: "red shoes"\nvector search -> IDs {2,4,6}', 'input')
    + text(180, 265, 'Only these three pairs pay for joint encoding', { size: 13, weight: 700, width: 335 });
  for (const [index, identity] of [2, 4, 6].entries()) {
    const horizontal = 5 + index * 120;
    body += arrow(180, 228, horizontal + 55, 305)
      + node(horizontal, 309, 110, 82, `query + I${identity}\nJOINT score`, 'focus');
  }
  body += node(20, 429, 320, 80, 'Example interaction in pair I4:\nquery "red" <-> item "red running shoes"\nscore can inspect tokens together', 'state')
    + text(180, 552, 'Missing I3 cannot be restored by reranking {2,4,6}.', { size: 13, weight: 700, width: 335 })
    + text(180, 599, 'Candidate recall bounds the final result.', { size: 12 });
  return { height: 628, body };
}

function evaluationBoundaries() {
  let body = text(180, 22, 'Two dependency examples; deployment target still matters', { size: 12, tone: 'soft', width: 335 })
    + text(18, 70, 'TIME: training must precede evaluation', { anchor: 'start', size: 13, weight: 700 });
  const days = ['Mon', 'Tue', 'Wed', 'Thu'];
  for (const [index, day] of days.entries()) {
    const horizontal = 25 + index * 83;
    body += node(horizontal, 102, 62, 62, `${day}\n${index < 2 ? 'train' : 'test'}`, index < 2 ? 'input' : 'output');
  }
  body += rule(179, 91, 179, 176, 'state', true)
    + text(180, 211, 'Random train Thu / test Tue crosses future -> past.', { size: 12, tone: 'warning', width: 335 })
    + text(18, 267, 'GROUPS: evaluate unseen entities', { anchor: 'start', size: 13, weight: 700 })
    + node(15, 297, 150, 77, 'Train: A1,A2\nentity A only', 'input')
    + node(195, 297, 150, 77, 'Test: B1,B2\nentity B only', 'output')
    + text(180, 413, 'Random A1 train / A2 test leaks entity A across.', { size: 12, tone: 'warning', width: 335 })
    + node(20, 460, 320, 108, 'Model selection stays inside training:\ninner tuning -> freeze -> untouched outer test\nnever choose the model on its final grade', 'state')
    + text(180, 591, 'Correct boundaries do not repair deployment shift.', { size: 13, weight: 700, width: 335 });
  return { height: 621, body };
}

function contrastiveMembership() {
  const candidates = [
    ['positive', 'positive j', 'same source A', 4, 'input'],
    ['false-negative', 'sampler-negative', 'source B; same class', 3, 'warning'],
    ['negative-1', 'negative k1', 'different source', 2, 'neutral'],
    ['negative-2', 'negative k2', 'different source', 1, 'state'],
  ];
  let body = text(180, 22, 'Toy exp(similarity / temperature) weights', { size: 12, tone: 'soft', width: 335 })
    + `<circle cx="70" cy="82" r="23" style="fill:var(--viz-input-bg);stroke:${ink('input')};stroke-width:2"></circle>`
    + text(70, 87, 'i', { weight: 700 })
    + `<circle cx="290" cy="82" r="23" style="fill:var(--viz-input-bg);stroke:${ink('input')};stroke-width:2"></circle>`
    + text(290, 87, 'j', { weight: 700 })
    + arrow(100, 82, 260, 82, 'input')
    + text(180, 65, 'same item A', { size: 12 })
    + text(70, 124, 'anchor crop', { size: 12, width: 110 })
    + text(290, 124, 'positive crop', { size: 12, width: 110 })
    + text(180, 158, 'DENOMINATOR: ALL FOUR CANDIDATES', { size: 12, weight: 700, width: 335 })
    + `<g data-infonce-denominator>${box(12, 174, 336, 343, 'neutral')}`;
  for (const [index, [identity, label, source, weight, tone]] of candidates.entries()) {
    const top = 187 + index * 80;
    body += `<g data-infonce-candidate="${identity}" data-weight="${weight}"${index === 0 ? ' data-infonce-numerator' : ''}>`
      + (index === 0 ? box(21, top - 4, 318, 68, 'focus') : '')
      + text(33, top + 16, label, { size: 13, anchor: 'start', width: 170, weight: 700 })
      + text(33, top + 40, source, { size: 12, anchor: 'start', width: 170 });
    for (let unit = 0; unit < weight; unit++) body += `<rect data-infonce-unit x="${218 + unit * 25}" y="${top + 8}" width="23" height="24" style="fill:var(--viz-${tone}-bg);stroke:${ink(tone)}"></rect>`;
    body += text(273, top + 53, `${weight}${index === 0 ? ' in numerator' : ''}`, { size: 12, width: 120 }) + '</g>';
  }
  body += '</g>'
    + text(180, 554, 'Denominator = 4 + 3 + 2 + 1 = 10', { size: 14, weight: 700, width: 335 })
    + text(180, 598, 'Positive probability = 4 / 10 = 0.4', { size: 14, weight: 700, width: 335 });
  for (let unit = 0; unit < 10; unit++) body += `<rect data-infonce-probability-unit="${unit}" x="${30 + unit * 30}" y="623" width="28" height="30" style="fill:var(--viz-${unit < 4 ? 'input' : 'neutral'}-bg);stroke:${ink(unit < 4 ? 'input' : 'neutral')}"></rect>`;
  body += text(180, 686, 'Four of ten equal units belong to the positive.', { size: 12, width: 335 })
    + text(180, 733, 'Same-class B stays in the denominator: the loss still repels it.', { size: 13, width: 335, tone: 'warning' });
  return { height: 774, body };
}

function concurrentRing() {
  const ranks = [[90, 110], [270, 110], [270, 270], [90, 270]];
  const sends = [[114, 110, 246, 110], [270, 134, 270, 246], [246, 270, 114, 270], [90, 246, 90, 134]];
  const labels = [[180, 87], [323, 184], [180, 309], [37, 184]];
  const tones = ['focus', 'input', 'state', 'output'];
  let body = text(180, 22, 'Four ranks; four equal chunks A, B, C, D', { size: 12, tone: 'soft', width: 335 })
    + text(180, 50, 'ONE CONCURRENT SEND STEP', { size: 12, weight: 700 });
  for (const [rank, [horizontal, vertical]] of ranks.entries()) body += `<circle cx="${horizontal}" cy="${vertical}" r="23" style="fill:var(--viz-neutral-bg);stroke:${ink('neutral')};stroke-width:1.75"></circle>`
    + text(horizontal, vertical + 5, `r${rank}`, { size: 14, weight: 700 });
  for (const [rank, endpoints] of sends.entries()) body += `<g data-ring-chunk="${'ABCD'[rank]}" data-sender="${rank}" data-receiver="${(rank + 1) % 4}" data-chunk-bytes="B/4">${arrow(...endpoints, tones[rank])}${text(...labels[rank], `${'ABCD'[rank]}: B/4`, { size: 12, width: rank % 2 ? 48 : 130, tone: tones[rank], weight: 700 })}</g>`;
  body += text(180, 357, 'FOLLOW ONE COORDINATE IN CHUNK A', { size: 12, width: 335, weight: 700 })
    + text(180, 395, 'Local contributions at r0..r3: 1, 2, 3, 4', { size: 12, width: 335 })
    + text(180, 428, 'REDUCE-SCATTER: receive, then add', { size: 12, weight: 700, tone: 'focus' });
  for (const [index, [incoming, local, result]] of [[1, 2, 3], [3, 3, 6], [6, 4, 10]].entries()) {
    const vertical = 466 + index * 52;
    body += `<g data-ring-phase="reduce-scatter" data-step="${index + 1}" data-incoming="${incoming}" data-local="${local}" data-result="${result}">${text(20, vertical, `r${index} -> r${index + 1}`, { size: 13, anchor: 'start' })}${text(235, vertical, `${incoming} + ${local} = ${result}`, { size: 17, weight: 700, tone: 'focus' })}</g>`;
  }
  body += text(180, 610, 'ALL-GATHER: forward the same finished value', { size: 12, width: 335, weight: 700, tone: 'output' });
  for (const [index, [sender, receiver]] of [[3, 0], [0, 1], [1, 2]].entries()) {
    const vertical = 650 + index * 44;
    body += `<g data-ring-phase="all-gather" data-step="${index + 1}" data-incoming="10" data-result="10">${text(20, vertical, `r${sender} -> r${receiver}`, { size: 13, anchor: 'start' })}${text(235, vertical, '10 -> 10', { size: 17, weight: 700, tone: 'output' })}</g>`;
  }
  body += text(180, 789, 'EACH RANK SENDS SIX B/4 CHUNKS', { size: 12, weight: 700 });
  for (let step = 0; step < 6; step++) body += `<g data-ring-sent-chunk="${step}" data-phase="${step < 3 ? 'reduce-scatter' : 'all-gather'}">${box(15 + step * 55, 813, 55, 35, step < 3 ? 'focus' : 'output')}${text(42.5 + step * 55, 836, 'B/4', { size: 12 })}</g>`;
  body += text(180, 885, '(3 + 3) * B/4 = 3B/2 sent; not sent + received.', { size: 12, width: 335 });
  return { height: 924, body };
}

function missingMatmulProducts() {
  const left = [[1, 2], [3, 4]];
  const right = [[5, 6], [7, 8]];
  const products = left.map((row) => [0, 1].map((column) => row.reduce((sum, value, inner) => sum + value * right[inner][column], 0)));
  let body = text(180, 22, 'Toy 2x2 matrices; each output block is one number', { size: 12, tone: 'soft', width: 335 })
    + node(15, 65, 150, 110, 'D0 owns\nA row 0 [1,2]\nB col 0 [5,7]', 'input')
    + node(195, 65, 150, 110, 'D1 owns\nA row 1 [3,4]\nB col 1 [6,8]', 'output');
  for (const [phase, top] of [['before', 235], ['after', 710]]) {
    body += text(180, top - 43, phase === 'before' ? 'LOCAL WORK: TWO MISSING PRODUCTS' : 'NEW PAIRS: ALL FOUR PRODUCTS EXIST', { size: 12, weight: 700, width: 335 });
    for (let column = 0; column < 2; column++) body += text(120 + column * 120, top - 17, `col ${column}`, { size: 12 });
    for (let row = 0; row < 2; row++) {
      body += text(35, top + row * 78 + 44, `r${row}`, { size: 12 });
      for (let column = 0; column < 2; column++) {
        const available = phase === 'after' || row === column;
        const horizontal = 60 + column * 120;
        const vertical = top + row * 78;
        const tone = row === column ? row === 0 ? 'input' : 'output' : available ? 'focus' : 'neutral';
        body += `<g data-matmul-phase="${phase}" data-row="${row}" data-column="${column}" data-value="${available ? products[row][column] : 'missing'}">${box(horizontal, vertical, 120, 78, tone, !available)}${text(horizontal + 60, vertical + 32, available ? products[row][column] : '?', { size: 22, weight: 700, tone: available ? tone : 'soft' })}${text(horizontal + 60, vertical + 58, available ? `D${row}` : 'not computed', { size: 12, width: 110 })}</g>`;
      }
    }
  }
  body += text(180, 430, 'ALL-REDUCE ADDS; IT CANNOT CREATE THE ? CELLS', { size: 12, weight: 700, width: 335, tone: 'warning' })
    + rule(20, 473, 340, 473)
    + text(180, 504, 'REPAIR: GATHER B, THEN MULTIPLY', { size: 13, weight: 700, width: 335 })
    + text(180, 545, 'D0 gains [6,8]; D1 gains [5,7].', { size: 13, width: 335 })
    + text(180, 593, 'D0: [1,2] dot [6,8] = 6 + 16 = 22', { size: 13, width: 335, tone: 'focus' })
    + text(180, 635, 'D1: [3,4] dot [5,7] = 15 + 28 = 43', { size: 13, width: 335, tone: 'focus' })
    + text(180, 913, 'Gather inputs first. New multiplication, not reduction, fills the gaps.', { size: 13, width: 335, weight: 700 });
  return { height: 960, body };
}

function pcaProjection() {
  const cosine = Math.sqrt(3) / 2;
  const scale = .85;
  const point = (along, residual) => [180 + scale * (along * cosine + residual / 2), 218 + scale * (-along / 2 + residual * cosine)];
  const samples = [[130, 10], [-130, -10], [85, -20], [-85, 20], [40, 10], [-40, -10]];
  let body = text(180, 22, 'Same six original toy samples; centered mean 0', { size: 12, tone: 'soft', width: 335 })
    + text(180, 62, 'Circle: sample; diamond: rank-1 reconstruction', { size: 12, width: 335 });
  const alongStart = point(-165, 0);
  const alongEnd = point(165, 0);
  const residualStart = point(0, -75);
  const residualEnd = point(0, 75);
  body += rule(...alongStart, ...alongEnd, 'input') + rule(...residualStart, ...residualEnd, 'state', true);
  for (const [along, residual] of samples) {
    const [sampleX, sampleY] = point(along, residual);
    const [projectionX, projectionY] = point(along, 0);
    body += `<g data-pca-along="${along}" data-pca-residual="${residual}"${along === 85 ? ' data-pca-selected' : ''}>${rule(sampleX, sampleY, projectionX, projectionY, 'focus', true)}<circle cx="${sampleX}" cy="${sampleY}" r="${along === 85 ? 7 : 5}" style="fill:var(--viz-input-bg);stroke:${ink('input')};stroke-width:${along === 85 ? 3 : 1.75}"></circle><path d="M${projectionX} ${projectionY - 5}L${projectionX + 5} ${projectionY}L${projectionX} ${projectionY + 5}L${projectionX - 5} ${projectionY}Z" style="fill:var(--viz-output-bg);stroke:${ink('output')};stroke-width:1.75"></path></g>`;
    if (along === 85) {
      const corner = [point(along - 8, 0), point(along - 8, -8), point(along, -8)];
      body += `<path data-pca-right-angle d="${corner.map(([horizontal, vertical], index) => `${index ? 'L' : 'M'}${horizontal} ${vertical}`).join('')}" style="fill:none;stroke:${ink('focus')};stroke-width:1.5"></path>`
        + text(sampleX - 12, sampleY - 15, 'x', { size: 14, weight: 700 })
        + text(projectionX + 32, projectionY + 20, '85 v1', { size: 12, tone: 'output' });
    }
  }
  body += text(273, 120, 'PC1: retain', { size: 12, tone: 'input' })
    + text(112, 145, 'PC2: discard', { size: 12, tone: 'state' })
    + text(180, 243, '0', { size: 12, tone: 'soft' })
    + text(180, 340, 'THE SAME PC1 COORDINATES, ON ONE AXIS', { size: 12, width: 335, weight: 700 })
    + rule(40, 400, 320, 400, 'input');
  for (const [along] of samples) {
    const horizontal = 180 + scale * along;
    body += `<circle data-pca-score="${along}" cx="${horizontal}" cy="400" r="${along === 85 ? 6 : 3}" style="fill:${ink(along === 85 ? 'focus' : 'input')}"></circle>`;
    if (along === 85) body += text(horizontal, 378, 'z = 85', { size: 14, weight: 700, tone: 'focus' });
  }
  for (const value of [-150, 0, 150]) body += rule(180 + scale * value, 395, 180 + scale * value, 405)
    + text(180 + scale * value, 430, value, { size: 12 });
  body += text(180, 470, 'z is one number; 85 v1 is a reconstructed vector.', { size: 13, weight: 700, width: 335 })
    + text(180, 524, 'x = 85 v1 - 20 v2; discard only the perpendicular part.', { size: 12, width: 335 })
    + text(180, 578, 'This error: 20^2 = 400. All six errors: 1200.', { size: 12, width: 335 });
  return { height: 607, body };
}

function factorizationEntry() {
  const ratings = [[5, 3, '?', 1], [4, '', 2, ''], ['', 1, '', 5], [2, '', 4, '']];
  let body = text(180, 22, 'Original toy factors; one predicted missing rating', { size: 12, tone: 'soft', width: 335 })
    + text(180, 64, '1. SELECT user A, item 3 in sparse R', { weight: 700, size: 13 });
  for (let column = 0; column < 4; column++) body += text(85 + column * 55, 96, column + 1, { size: 12, weight: column === 2 ? 700 : 400 });
  for (const [row, values] of ratings.entries()) {
    body += text(38, 132 + row * 40, ['A', 'B', 'C', 'D'][row], { size: 12 });
    for (const [column, value] of values.entries()) body += box(58 + column * 55, 109 + row * 40, 55, 40, row === 0 && column === 2 ? 'focus' : 'neutral') + text(85 + column * 55, 135 + row * 40, value, { size: 14 });
  }
  body += text(180, 312, '2. MATCH user and item factor rows', { weight: 700, size: 13 })
    + node(15, 337, 150, 73, 'user A\nu_A = [2, 1]', 'input')
    + node(195, 337, 150, 73, 'item 3\nv_3 = [1.5, 1]', 'output')
    + arrow(90, 410, 90, 444) + arrow(270, 410, 270, 444)
    + node(15, 448, 150, 61, 'factor 1\n2 * 1.5 = 3', 'focus')
    + node(195, 448, 150, 61, 'factor 2\n1 * 1 = 1', 'focus')
    + text(180, 551, '3. SUM: predicted R[A,3] = 3 + 1 = 4', { size: 14, weight: 700, width: 335 })
    + text(180, 598, 'A prediction fills the missing cell; it is not an observed rating.', { size: 12, width: 335 });
  return { height: 630, body };
}

function queryBeforeStore() {
  let body = text(180, 22, 'Toy input [4, 2, 7]; target 9', { size: 12, tone: 'soft' });
  for (const [index, value] of [4, 2, 7].entries()) {
    body += box(72 + index * 72, 53, 72, 46, index === 2 ? 'focus' : 'input')
      + text(108 + index * 72, 83, value, { size: 17, weight: 700 })
      + text(108 + index * 72, 121, `index ${index}`, { size: 12 });
  }
  body += node(20, 154, 320, 65, 'Current index 2: value 7\ncomplement = target 9 - 7 = 2', 'focus')
    + arrow(180, 219, 180, 254, 'focus')
    + text(180, 276, 'seen contains ONLY earlier positions', { size: 13, weight: 700 })
    + text(110, 309, 'value', { size: 12, tone: 'soft' })
    + text(252, 309, 'earlier index', { size: 12, tone: 'soft' })
    + rule(70, 319, 310, 319)
    + text(110, 345, '4') + text(252, 345, '0')
    + box(70, 357, 240, 36, 'output')
    + text(110, 381, '2', { weight: 700 }) + text(252, 381, '1', { weight: 700 })
    + text(180, 425, 'Lookup 2 returns earlier index 1.', { size: 13 })
    + node(20, 450, 320, 60, 'Return [1, 2]: 2 + 7 = 9\nTwo different input positions', 'output')
    + text(180, 549, 'Index 2 was absent during the query.', { size: 13, weight: 700 })
    + text(180, 579, 'On a miss, store the current value only after querying.', { size: 12, width: 330 });
  return { height: 618, body };
}

export const redraws = {
  'infonce-positive-candidate-set': {
    title: 'The positive belongs to both numerator and denominator; a false negative stays in the denominator',
    description: 'A toy InfoNCE example uses unnormalized exponentiated similarity weights four, three, two, and one. One denominator enclosure contains all four candidates. Its positive row has a smaller numerator enclosure and four equal units. The same-class different-source item is still a sampler-negative with three units. The denominator totals ten and the positive probability is four tenths, drawn with ten equal units. Weights are illustrative, not observed similarities.',
    printPage: true,
    layout: contrastiveMembership,
    caption: 'The nested positive row belongs to both enclosures. Its four units are part of the denominator total of ten. Source B is semantically similar but remains a sampler-negative, so the loss still pushes it away. The weights are illustrative.',
  },
  'ring-all-reduce-chunk-trace': {
    title: 'All four chunks move concurrently; reduction adds, while all-gather only copies',
    description: 'Four ranks form a directed ring. At one concurrent send step ranks zero, one, two, and three send chunks A, B, C, and D respectively to the next rank. Every message has size B over four. A separate trace follows one coordinate of chunk A with local values one, two, three, and four: three receives and additions produce three, six, then ten at rank three. Three all-gather hops copy ten unchanged to ranks zero, one, and two. Six equal sent-chunk cells give three B over two sent bytes per rank, not sent-plus-received traffic.',
    printPage: true,
    layout: concurrentRing,
    caption: 'The ring inset is one simultaneous step, not the whole schedule. Follow A separately: 1 becomes 3, then 6, then 10; all-gather preserves 10. Every rank sends one B/4 chunk in each of six steps, for 3B/2 sent bytes.',
  },
  'same-mesh-axis-diagonal-gap': {
    title: 'A reduction cannot invent the two products that no device computed',
    description: 'In a declared two-by-two toy, A has rows one-two and three-four, and B has columns five-seven and six-eight. Device zero initially owns only A row zero and B column zero, producing nineteen. Device one owns only A row one and B column one, producing fifty. The same-scale output grid leaves off-diagonal cells uncomputed, not zero. All-gather of B columns gives each device the missing input. New local dot products produce twenty-two and forty-three; a second grid fills those exact gaps while retaining nineteen and fifty.',
    printPage: true,
    layout: missingMatmulProducts,
    caption: 'Compare the same output positions before and after the gather. The new input pairs produce 22 and 43; the existing 19 and 50 do not change. All-reduce alone cannot manufacture the absent cross-index multiplications.',
  },
  'actor-critic-two-objective-flow': {
    title: 'The advantage stops gradients; both heads still train shared features',
    description: 'One state enters shared features and then actor and critic heads. The policy loss uses log action probability and advantage y minus V. A stop mark on the V-to-advantage path blocks policy-loss gradients into the critic. The value loss is an illustrative squared error against fixed target y. Separate dashed paths return each loss through its own head into the shared features. Solid lines construct the losses; dashed lines indicate parameter-gradient ownership.',
    layout: actorCriticPaths,
    caption: 'The advantage value reaches the policy loss, but its stop mark blocks the return gradient into V. Each loss still updates its own head and the shared features. The squared value loss and fixed target y are illustrative.',
  },
  'coding-interview-method-state': {
    title: 'The complement is a value; the saved map entry recovers its earlier index',
    description: 'For toy input four, two, seven and target nine, current index two contains seven. Nine minus seven is two. The earlier-value map contains four to index zero and two to index one. Looking up two returns one, so indices one and two identify two plus seven. The current index is absent during the query; insertion happens only after a miss.',
    layout: queryBeforeStore,
    caption: 'Follow value 2 from the complement calculation to the saved index 1. Returning [1,2] uses two different positions. Query-before-store is the invariant, not a variable spelling.',
  },
  'pca-rank-one-projection': {
    title: 'The retained coordinate stays; the perpendicular coordinate becomes reconstruction error',
    description: 'The original six centered toy samples are drawn at PC coordinates (130,10), (-130,-10), (85,-20), (-85,20), (40,10), and (-40,-10). Orthogonal PC axes are rotated thirty degrees. Each circle connects perpendicularly to its diamond projection. The selected sample x equals 85 v1 minus 20 v2; a right-angle marker identifies the discarded residual. A separate one-dimensional axis draws all six PC1 scores and highlights the same scalar z=85, distinct from the reconstructed vector 85 v1. Cross-covariance is zero and PC1 variance is larger. Squared discarded coordinates sum to twelve hundred.',
    printPage: true,
    layout: pcaProjection,
    caption: 'Follow the highlighted sample to scalar score z=85. Reconstruct the diamond at 85 v1. The perpendicular -20 v2 is discarded, giving squared error 400. These are illustrative coordinates, not measured data.',
  },
  'mf-missing-rating-dot-product': {
    title: 'One missing entry selects two matching rows, then sums their coordinate products',
    description: 'The original toy sparse ratings matrix highlights user A and item three. Matching factors u_A=[2,1] and v_3=[1.5,1] contribute three and one, totaling predicted rating four. The complete selection and calculation remain visible in a vertical phone-width composition.',
    layout: factorizationEntry,
    caption: 'Keep user A and item 3 fixed through every stage. Their matching factor rows contribute 3 and 1. The predicted missing rating is 4, not a newly observed preference.',
  },
  'flashattention-standard-hbm-path': {
    title: 'The avoided HBM round trips belong to intermediates, not every tensor',
    description: 'A declared two-byte-intermediate toy compares a materialized attention baseline with exact tiled attention. Scores and probabilities each incur one n-squared-element write and read in the baseline. Tiling avoids those global intermediates but still moves inputs, weights as applicable, and final output and retains dense quadratic arithmetic.',
    layout: () => comparison({ example: 'Toy 2-byte intermediates; count only score/probability traffic', left: 'MATERIALIZED', right: 'TILED EXACT', rows: [
      { label: 'Scores: n by n', left: 'write 2n^2 bytes\nread 2n^2 bytes', right: 'tile stays on chip\nno global score matrix' },
      { label: 'Probabilities: n by n', left: 'write 2n^2 bytes\nread 2n^2 bytes', right: 'online softmax state\nno global P matrix' },
      { label: 'What does not disappear', left: 'Q, K, V, output\ndense attention work', right: 'Q, K, V, output\ndense attention work', rightTone: 'input' },
    ] }),
    caption: 'For these declared two-byte intermediates, four transfers total 8n^2 bytes in the baseline. Avoiding them does not eliminate all HBM traffic or quadratic arithmetic. Actual intermediate precision can differ.',
  },
  'regularization-diagnose-and-target': {
    title: 'A train/validation gap is evidence to investigate, not a diagnosis by itself',
    description: 'Illustrative same-task train losses .9,.65,.4,.25,.15 and validation losses .92,.7,.5,.55,.7 diverge after step three. Split validity and train/eval mode are checked before testing an early-stop checkpoint on independent held-out evidence. Poor-both and good-both alternatives remain explicit; the intervention result is unknown.',
    layout: regularizationEvidence,
    caption: 'The dashed validation curve turns upward while train loss falls. Check split and mode before attributing the gap to capacity. The proposed held-out intervention has no invented result.',
  },
  'environment-grader-contract-handoff': {
    title: 'A successful final state can still fail the declared process contract',
    description: 'Fictional E17 closes owned order seventeen, then attempts a forbidden other-account read which is denied with no forbidden side effect. Immutable events reach a grader whose declared hard rule is zero unauthorized attempts. Task outcome passes but process fails, making the episode ineligible. Hidden grading evidence is not exposed to the policy.',
    layout: gradedEpisode,
    caption: 'Read actual events, not just the final task state. This declared grader rejects even the blocked unauthorized attempt. Other contracts may differ; quality cannot override this hard rule.',
  },
  'rlhf-dpo-training-paths': {
    title: 'The stored preference pair enters different optimization machinery',
    description: 'Both branches use a fictional preference for current P2 over stale P1 with a fixed SFT reference. RLHF fits a reward model and draws fresh current-policy rollouts for reward-plus-KL RL updates. DPO directly updates the policy on the stored pair using the reference-relative likelihood-ratio margin; it needs no separate reward-model fit or rollout loop.',
    layout: preferencePaths,
    caption: 'Fresh policy rollouts occur only on the RLHF branch. DPO reuses the stored pair and fixed reference directly. Its reference-relative objective is not an unconstrained update.',
  },
  'oversight-control-assumption-test': {
    title: 'Checking a hard answer and containing an intentional attacker need different evidence',
    description: 'The same fictional difficult code task is evaluated under two distinct scenarios. Without assuming attack, assisted decomposition exposes a boundary-case counterexample and tests evaluator correctness. With intentional untrusted action assumed, a trusted scoped boundary denies an unapproved action and records unchanged protected state, testing containment rather than answer quality.',
    layout: () => comparison({ example: 'Fictional hard code task; two separate threat assumptions', left: 'OVERSIGHT', right: 'CONTROL', rows: [
      { label: 'Assumption', left: 'hard to judge\nattack not required', right: 'intentional attacker\nuntrusted actions', rightTone: 'state' },
      { label: 'Concrete evidence', left: 'assisted edge test\nfinds wrong answer', right: 'unapproved action denied\nprotected state unchanged' },
      { label: 'Success test', left: 'distinguish correct\nfrom faulty work', right: 'contain attempted action\nunder trusted boundary' },
    ] }),
    caption: 'Assistance tests whether the evaluator can judge difficult work. Containment tests a trusted boundary under the intentional-attacker assumption. Neither observation substitutes for the other.',
  },
  'two-tower-cross-encoder-fanout': {
    title: 'Retrieve reusable vectors, then jointly encode only the surviving pairs',
    description: 'A toy six-item corpus has independently stored item vectors. One query vector retrieves IDs two, four, six, so only three query/item pairs receive fresh joint scoring. A token-level example pairs query red with red running shoes in I4. Missing item I3 cannot be restored by the reranker.',
    layout: retrievalFanout,
    caption: 'All six items can reuse offline vectors, but only the three retrieved IDs get joint token interaction here. The reranker cannot recover an item outside that candidate set.',
  },
  'cross-validation-fit-the-question': {
    title: 'A valid fold boundary follows the dependency and the deployment question',
    description: 'A time example trains Monday and Tuesday and tests Wednesday and Thursday; random training on Thursday while testing Tuesday leaks future information. A new-entity example trains A1/A2 and tests B1/B2; splitting A1/A2 across train/test leaks the entity. Model selection remains inside an untouched outer evaluation boundary. These repairs do not solve distribution shift.',
    layout: evaluationBoundaries,
    caption: 'The corrected boundaries protect time direction and unseen entities. Model selection must also stay inside the evaluation boundary. Dependency-safe folds still do not guarantee deployment representativeness.',
  },
  'ambitious-project-three-evidence-test': {
    title: 'The same decision must survive three changes of zoom',
    description: 'A fictional multi-tenant migration risks cross-account data exposure. The narrator chose a shadow comparison rather than immediate replacement and owned the comparison harness; a platform team owned rollout. Technical diff P17 and a team decision note support that boundary. Project scope alone is not evidence of personal seniority.',
    layout: () => comparison({ example: 'Fictional tenant migration; not personal interview history', left: 'QUESTION', right: 'SAME EVIDENCE', rows: [
      { label: 'Strategic zoom', left: 'why not replace now?', right: 'cross-account risk\nchoose shadow test' },
      { label: 'Technical zoom', left: 'what changed the path?', right: 'P17 comparison harness\nchecks tenant boundary' },
      { label: 'Organizational zoom', left: 'who owned what?', right: 'I: harness\nplatform team: rollout' },
    ] }),
    caption: 'The shadow-test decision connects stakes, implementation, and ownership. The platform team\'s rollout is not the narrator\'s solo achievement. Size alone proves neither judgment nor seniority.',
  },
  'overrated-technique-claim-stress-test': {
    title: 'A bounded critique leaves room for a reversing result',
    description: 'A proposed matched study pins model M, task inputs, sampling settings, and scoring. Source-fact and intermediate-computation bottlenecks are tested separately with direct versus reasoning-style prompts. Quality, latency, and cost are unmeasured. Equal-budget quality improvement on source-fact cases would reverse the narrow hypothesis that extra reasoning cannot help that bottleneck.',
    layout: () => comparison({ example: 'Proposed study; pin model M, inputs, settings, and scorer', left: 'DIRECT', right: 'REASONING STYLE', rows: [
      { label: 'Missing source-fact slice', left: 'quality ?\nlatency ?; cost ?', right: 'quality ?\nlatency ?; cost ?', rightTone: 'state' },
      { label: 'Intermediate-computation slice', left: 'quality ?\nlatency ?; cost ?', right: 'quality ?\nlatency ?; cost ?', rightTone: 'state' },
      { label: 'What reverses the critique?', left: 'matched budget\nindependent scoring', right: 'source-fact quality gain\nwithout new evidence', rightTone: 'focus' },
    ] }),
    caption: 'Question marks are unmeasured outcomes, not zero gains. Keep task slices separate and compare at matched budgets. A contrary result must be allowed to reverse the hypothesis.',
  },
  'people-also-bought-three-relations': {
    title: 'An ink association becomes a complement only after product-fit verification',
    description: 'A toy printer P1 is the fixed purchase anchor. Co-purchase links ink C7 with a count of fifty baskets and no known denominator. Co-view links another printer P2; attributes link similar printer P3. Only verified C7-to-P1 compatibility can admit ink to the complementary shelf.',
    layout: purchaseRelations,
    caption: 'Each edge names a different relation to P1. Fifty baskets is a count, not a rate. Verify C7 fits P1 before treating it as a complementary purchase.',
  },
  'policy-gradient-advantage-sign': {
    title: 'A positive return can still suppress the sampled action',
    description: 'A declared three-action softmax starts with zero logits and uniform probabilities. After sampling A with action-independent baseline three, returns five, three, one give advantages two, zero, minus two. Exact logit gradient-ascent steps of size .1 produce normalized probabilities; A increases, stays fixed for this sample, or decreases despite positive absolute return.',
    layout: advantageSteps,
    caption: 'Subtract the baseline before following the sign. Every stacked bar sums to one after the exact softmax step. Zero means this sample contributes nothing, not that an entire batch cannot update.',
  },
  'reward-model-distribution-shift': {
    title: 'Optimization can move responses beyond the proxy\'s validated coverage',
    description: 'Fictional outputs R1 and R2 have proxy/human scores four/four and six/five. Optimized R3 has proxy nine and fresh human score one, outside the represented proxy-score band three to six. It is an illustrative possible shortcut, not an inevitable optimization result. New on-policy comparisons must revalidate the proxy.',
    layout: proxyShift,
    caption: 'R3 improves the proxy while failing the fresh rubric in this toy example. A score-range band is not full distributional coverage. Inspect on-policy outputs and revalidate; do not assume every run diverges.',
  },
  'legal-rag-claim-lineage': {
    title: 'A resolvable citation can leave the claim unsupported',
    description: 'Fictional jurisdiction J authority version 2026-08-15 section four contains P-017 requiring filing within ten days and P-042 saying the clerk records receipt. C1 repeats the ten-day deadline and is supported. C2 infers guaranteed approval from receipt and is held. Release separately requires current jurisdiction/time applicability and required attorney approval.',
    layout: legalClaims,
    caption: 'P-017 supplies C1\'s exact bound. P-042 does not establish approval, so C2 is removed or escalated. Support, applicability, and release authority remain separate. This is fictional, not legal advice.',
  },
  'rag-index-request-paths': {
    title: 'Follow one versioned chunk from the corpus into the answer',
    description: 'Fictional document D version three has c1 stating thirty-day returns and c2 giving a support email. Lexical retrieval returns c1 and dense retrieval c1/c2. Fusion retains both; the optional reranker removes c2 from answer context. The answer cites D@v3:c1. Corpus refresh need not retrain model weights, and authorization and support still apply.',
    layout: chunkLineage,
    caption: 'The same c1 identity survives offline indexing, retrieval, selection, and citation. The reranker removes c2 here. Preserve access and source version; a citation alone still does not prove support.',
  },
  'gqa-eight-queries-two-kv-groups': {
    title: 'Share two K/V pairs without merging eight query heads',
    description: 'Eight query heads each retain their own output. A dashed bracket associates Q1 through Q4 with K1 and V1, and Q5 through Q8 with K2 and V2. Proportional head-dependent cache bars compare eight, two, and one KV heads for MHA, GQA, and MQA at fixed sequence length and head dimension.',
    layout: querySharing,
    caption: 'Follow each Q-to-O arrow separately. Brackets mean K/V reuse, not merged queries. Two cached pairs instead of eight use one-quarter of head-dependent cache, not one-quarter of all computation.',
  },
  'project-stop-forward-evidence-ledger': {
    title: 'Past spend cannot fill a future-value column',
    description: 'Three fictional mutually exclusive options share a hundred future resource units. A bounded test spends ten, continued investment one hundred, and redirect one hundred. Past spend is excluded. The test stops on day seven and requires at least ninety-five of a hundred contract passes and zero leaks; passing triggers a new decision, not automatic renewal.',
    layout: forwardChoices,
    caption: 'Compare equal future budgets, not historical investment. The ten-unit test must change a decision by day seven. All returns remain uncertain; continuation displaces the redirect.',
  },
  'judge-calibration-gates': {
    title: 'A hard failure and a position-sensitive judge remain separate failures',
    description: 'Fictional response B lacks a required id while receiving quality 4.8 out of five. That score cannot override the deterministic failure. In an independent order-swap probe, the same response pair yields A when ordered AB and B when ordered BA, revealing first-position sensitivity and requiring semantic review.',
    layout: () => comparison({ example: 'Fictional response pair A/B; fixed independent rubric', left: 'CHECK', right: 'OBSERVED RESULT', rows: [
      { label: 'Deterministic output contract', left: 'B: required id absent', right: 'HARD FAIL\nquality 4.8 cannot veto', rightTone: 'warning' },
      { label: 'Same pair; order swapped', left: 'AB -> prefers A\nBA -> prefers B', right: 'first-position bias\nreview the judge', rightTone: 'warning' },
      { label: 'Independent human rubric', left: 'human prefers A', right: 'judge disagreement\nretain separately', rightTone: 'state' },
    ] }),
    caption: 'Neither high quality nor averaging repairs a missing required field. The swapped pair exposes a separate judge-instability problem. Preserve both outcomes and escalate the semantic disagreement.',
  },
  'llm-eval-chain-of-trust': {
    title: 'A proxy can improve while the omitted user contract fails',
    description: 'A fictional user needs cross-account records withheld. Case C17 is scored only for presence of a citation under incomplete rubric R1. The scorer rises from ninety to ninety-five percent, but a fresh restricted-account slice exposes unauthorized outputs in two of ten cases. The case coverage and rubric must encode the omitted permission contract before scorer recalibration.',
    layout: () => comparison({ example: 'Fictional C17: withhold other-account records', left: 'PROXY CHAIN', right: 'MISSING CONTRACT', rows: [
      { label: 'Incomplete rubric R1', left: 'citation present?\nC17 scored PASS', right: 'access permission\nnot represented', rightTone: 'warning' },
      { label: 'Offline scorer improves', left: '90% -> 95%', right: 'fresh restricted slice\n2 / 10 unauthorized', rightTone: 'warning' },
      { label: 'Repair the broken link', left: 'add restricted cases\nrevise R1', right: 'check permission\nrecalibrate scorer' },
    ] }),
    caption: 'The broken link is the omitted permission criterion, not merely a low aggregate score. Fresh failures require case and rubric repair before trusting the proxy again.',
  },
  'long-context-three-bottlenecks': {
    title: 'Fixing cache allocation does not flatten attention work',
    description: 'Three tracks share relative context-length positions one, two, four, eight. A declared toy trained-position span ends at two, with unvalidated positions beyond it. Exact dense attention work scales one, four, sixteen, sixty-four; logical KV bytes scale one, two, four, eight. FlashAttention does not remove quadratic arithmetic, and paging alone does not reduce logical KV growth. No measured usable threshold is invented.',
    layout: contextTracks,
    caption: 'Compare horizontal context positions, not vertical units across tracks. Paging does not change logical KV growth. FlashAttention does not change dense quadratic arithmetic. Usable context must satisfy every constraint.',
  },
  'coding-assistant-evidence-ladder': {
    title: 'The same patch supports three different kinds of claim',
    description: 'A fictional patch passes eight specified tests, then review finds a new dependency, and the developer accepts but rewrites the patch three hours later. Each observation supports only its own bounded claim. None alone establishes causal productivity improvement.',
    layout: () => comparison({ example: 'Fictional patch P17; follow one artifact', left: 'OBSERVATION', right: 'SUPPORTED CLAIM', rows: [
      { label: 'Execute', left: '8 / 8 tests pass', right: 'tested cases pass\nnot all behavior' },
      { label: 'Inspect', left: 'new dependency found', right: 'maintenance concern\nneeds review', rightTone: 'warning' },
      { label: 'Observe', left: 'accepted -> rewritten\nafter 3 hours', right: 'acceptance is not\nlasting task benefit', rightTone: 'focus' },
    ] }),
    caption: 'Follow P17 through all three observations. Tests, maintainability, and lasting usefulness are different claims. A causal benefit still needs a controlled comparison.',
  },
  'independent-confirmation-vs-adaptive-reuse': {
    title: 'A feedback arrow changes the benchmark\'s role',
    description: 'Public benchmark B feeds model selection and therefore becomes development evidence. In a separate lane, M2 freezes before hidden fresh set H is opened once. There is deliberately no return arrow from H into that already-selected model. Later selection would require new confirmation.',
    layout: benchmarkBoundary,
    caption: 'The loop on B makes it development feedback. The one-way H path confirms only the frozen system and tested scope. Further selection needs another independent evaluation.',
  },
  'bandit-three-selection-rules': {
    title: 'The same leader can lose to an uncertainty bonus or one posterior draw',
    description: 'Arms A, B, C have means .62, .55, .35 and counts forty, four, twenty. Epsilon .3 chooses A with probability .8 because its random branch also includes A. UCB uses mean plus .2 sqrt(log64/n) and selects B. Declared illustrative Normal posteriors with means .62,.55,.35 and standard deviations .06,.18,.09 produce one shown draw .58,.78,.31, selecting B.',
    layout: banditRules,
    caption: 'Dots and whiskers show UCB means and bonuses, not confidence guarantees for this chosen coefficient. Thompson curves are explicit toy assumptions. Only the displayed draw selects B.',
  },
  'llm-adaptation-layers': {
    title: 'Valid JSON can still apply the wrong policy',
    description: 'Policy A permits seven days and applicable B thirty days on a shared day axis. Day fourteen exceeds A but fits B. A response falsely declaring ineligibility under A is structurally valid but fails the independent current-evidence check. Release is held.',
    layout: adaptationGates,
    caption: 'The day-14 marker lies beyond A and inside B. Syntax passes while current-policy evidence fails. Neither a schema nor fine-tuning supplies missing current authority.',
  },
  'fine-tuning-two-decisions': {
    title: 'Choose a training signal and an update scope, not one item from a mixed list',
    description: 'Three objective rows use domain text, prompt/ideal-response pairs, and preference triples. Each crosses full base-weight updates and LoRA adapter updates with the base frozen. The SFT-plus-LoRA cell is highlighted. The rows are alternatives, not a mandatory sequence.',
    layout: tuningMatrix,
    caption: 'SFT tells you which loss and examples train the model. LoRA tells you where trainable updates live. Their highlighted intersection is one valid combination.',
  },
  'as-five-cross-cutting-signals': {
    title: 'One exchange can expose several skills and leave others unknown',
    description: 'A fictional mock asks whether an overall completion gain justifies shipping. The candidate asks about harmed users and stop criteria, then updates after a high-risk slice failure. Scoping, responsiveness, and explicit decision-making are observed; delivery ownership and scientific depth remain untested. The next drill targets uncertainty in the slice test.',
    layout: evidenceExchange,
    caption: 'The exchange provides overlapping evidence, not five separate rounds or personality scores. Unknown delivery and scientific evidence stays unknown. Choose the next drill from the observed gap.',
  },
  'interview-evidence-to-drill': {
    title: 'A confirmed constraint selects a drill; an unknown permission selects a question',
    description: 'A fictional round brief confirms an existing codebase and forty-five minutes, but tool permission is unknown. Practice therefore uses a repository defect and a bounded focused-test exercise. Tool permission returns to the authorized recruiting contact rather than being inferred from a company identity.',
    layout: () => comparison({ example: 'Fictional confirmed brief; not a lab-wide process', left: 'EVIDENCE', right: 'NEXT ACTION', rows: [
      { label: 'Environment confirmed', left: 'existing codebase', right: 'trace one repo defect' },
      { label: 'Duration confirmed', left: '45 minutes', right: 'time-box patch + proof' },
      { label: 'Tool permission UNKNOWN', left: 'no written rule yet', right: 'ask authorized contact\ndo not assume', rightTone: 'warning' },
    ] }),
    caption: 'Match confirmed environment and time to the drill. Unknown tool permission is not an invitation to guess. Ask the authorized contact before relying on that tool.',
  },
  'proof-of-work-three-checks': {
    title: 'Evidence can support a bounded contribution without proving a broader outcome',
    description: 'Fictional contributor claims authorship of parser-boundary patch P17, not the team rollout. A shareable diff supports the implementation, a witness observed review but not productivity impact, and no controlled outcome comparison is available. Authorship is supported, the broad effect unknown, and a solo-deployment claim conflicts with team ownership.',
    layout: () => comparison({ example: 'Fictional P17; shareable evidence only', left: 'CLAIM', right: 'ACTUAL SUPPORT', rows: [
      { label: 'Bounded implementation', left: 'I wrote the check', right: 'P17 diff + reviewer\nSUPPORTED' },
      { label: 'Productivity improvement', left: 'the team became faster', right: 'no causal comparison\nUNKNOWN', rightTone: 'state' },
      { label: 'Deployment ownership', left: 'I shipped it alone', right: 'team owned rollout\nCONFLICT', rightTone: 'warning' },
    ] }),
    caption: 'The diff and witness support the narrow implementation claim, not every surrounding story. Preserve unknown outcomes, team ownership, and confidentiality. Consistency alone does not establish truth.',
  },
  'adamw-decoupled-shrinkage': {
    title: 'Equal weights, unequal coupled penalties',
    description: 'With theta one, alpha .01, lambda .1, and fixed adaptive scales one and ten, coupled L2 contributes shrinkage .001 and .0001. AdamW contributes equal .001 shrinkage. Leftward arrows share the same parameter-distance scale; this isolates the penalty and is not a full Adam recurrence.',
    layout: shrinkage,
    caption: 'Compare arrows on one zoomed parameter scale. Coupled L2 is divided by the fixed adaptive scale. AdamW is not. Coupled penalties also affect moment buffers.',
  },
  'quality-speed-proportionate-path': {
    title: 'New evidence can change which users may be exposed',
    description: 'A fictional staged export release has forty ordinary test cases and ten protected-record cases. Ordinary cases show zero permission violations, but two protected-record cases expose forbidden fields. The decision owner holds the protected path; a reversible ordinary cohort remains subject to authorization and a zero-violation stop rule.',
    layout: () => gate({ example: 'Fictional export launch; security owner can stop it', input: 'Ordinary cases: 0 / 40 violations\nprotected cases: 2 / 10 violations', test: 'Every exposed path authorized\nAND zero permission violations?', yes: 'ordinary cohort only\n1% reversible stage\nstop on first violation', no: 'protected path HELD\nrepair field filtering\nindependent retest' }),
    caption: 'The protected path fails before exposure. A stage is not a permission exception. The named owner stops even the ordinary cohort on its first violation.',
  },
  'agent-review-control-loop': {
    title: 'A passing patch returns to an independent boundary test',
    description: 'A fictional auth.py patch must deny expired tokens. Agent execution returns a diff and passing focused tests. The engineer independently tests expiry exactly equal to now, finds a failure, and returns the same bounded patch for revision. Acceptance remains outside the agent lane.',
    layout: patchHandoff,
    caption: 'The independent equality case breaks the proposed patch. Return it for revision without widening scope. The agent supplies evidence; the engineer retains acceptance authority.',
  },
  'product-metric-decision-authority': {
    title: 'Two passing metrics cannot cancel one latency veto',
    description: 'The completion gain lower bound 4.66 points exceeds its two-point floor, cost 1.97 dollars is below two dollars, but latency 2.1 seconds exceeds two seconds. Each metric has its own labelled scale and all three feed an AND gate. Diagnostics cannot authorize release.',
    layout: metricVeto,
    caption: 'Use each row\'s units and declared inequality. Completion and cost pass; latency fails. Their conjunction holds full release. Diagnostics explain the failure but cannot override it.',
  },
  'dbscan-neighborhood-expansion': {
    title: 'The same radius makes D core and E border',
    description: 'On a shared one-dimensional distance scale, radius .31 around D=.2 includes A=-.1, B=0, C=.1, D, and E=.5. The same radius around E includes only D and E. MinPts four counts self. F=2 is 1.5 from its nearest point and is noise.',
    layout: densityNeighborhoods,
    caption: 'The shaded spans use the same distance scale and radius. D counts five, including itself. E counts two and cannot extend the cluster. F has no core neighbor.',
  },
  'coding-product-authority-boundary': {
    title: 'A shared context broker is not shared tool authority',
    description: 'The permission-aware broker supplies four modes. Inline, chat, and review return generated output without executing tools. Only the bounded agent operates inside the dashed isolated workspace and returns a patch with evidence. The developer still applies, revises, or rejects it.',
    layout: authorityPorts,
    caption: 'The dashed boundary encloses tool execution, not final acceptance. The first three modes stop at generated outputs. A returned patch still needs the developer\'s decision.',
  },
  'domain-adaptation-shift-assumptions': {
    title: 'Which factor moved? Inspect the same joint-probability cells',
    description: 'Toy joint tables change only the declared factor. Covariate shift moves P(X=1) from .2 to .8 with fixed label conditionals. Label shift moves P(Y=1) from .2 to .8 with fixed input conditionals. Two concept-shift worlds then share identical balanced inputs but opposite deterministic label rules.',
    layout: jointShifts,
    caption: 'Rows index X and columns index Y. Check the stated conditional against the cells. Identical target input marginals cannot identify which opposite label rule holds.',
  },
  'ecosystem-delayed-feedback-loop': {
    title: 'Viewer response can arrive before creator supply changes',
    description: 'A shared horizon separates the viewer response inside a short experiment from creator supply feedback arriving weeks later. The later direction is unknown, and the strip is ordered rather than proportional to elapsed time.',
    layout: feedback,
    caption: 'Viewer evidence can arrive within the experiment. Creator retention and production may not. The later direction is unknown; horizon spacing is schematic.',
  },
  'agent-governed-action-path': {
    title: 'One call ID crosses permission, execution, and outcome evidence',
    description: 'Illustrative call c17 requests report A. Identity, purpose, and scope are checked before any narrow credential is issued. A permitted call reaches the executor and receives a tool receipt. Denial means no credential and no dispatch.',
    layout: () => gate({ example: 'Illustrative request c17: read report A', input: 'c17 + identity + purpose + report A', test: 'Does current policy permit this exact scope?', yes: 'narrow credential\nc17 dispatch\ntool receipt', no: 'DENIED\nno credential\nno tool call' }),
    caption: 'The permission decision comes before credential issuance. A dispatch attempt is not an authoritative outcome; the tool receipt establishes the result.',
  },
  'coding-assistant-two-latency-paths': {
    title: 'A correct suggestion for yesterday\'s buffer is still stale',
    description: 'Editor and server lanes share one ordered time direction. R1 captures version 41, typing advances the document to version 42, and R1 returns stale and is discarded. R2 targets version 42 but still needs current version, deadline, and policy checks before acceptance. Tool tasks require separate authorization.',
    layout: editorRace,
    caption: 'R1 cannot overwrite version 42, even when its code is correct. R2 still needs live version, deadline, and policy checks. Vertical spacing is schematic.',
  },
  'causal-confounding-dag': {
    title: 'A shared cause can connect treatment and outcome without the treatment effect',
    description: 'Difficulty D points to both treatment T and outcome Y, creating the backdoor association path T left arrow D right arrow Y. The separate T to Y edge is the effect of interest. Adjustment conditions on D without physically removing a causal arrow.',
    layout: confounding,
    caption: 'Follow the two difficulty arrows as one backdoor route. The horizontal treatment arrow is a different path. Adjustment does not physically delete arrows.',
  },
  'fused-moe-triton-sram-roofline': {
    title: 'Fusion removes the intermediate round trip, not every memory transfer',
    description: 'Aligned unfused and fused paths keep the same intermediate h with shape t by F. For two-byte elements, the unfused path writes 2tF bytes then reads 2tF bytes. A fitting fused tile avoids that 4tF-byte intermediate round trip. Input, weight, and output traffic remain.',
    layout: memory,
    caption: 'The avoided transfer is the same intermediate written and read once: 4tF bytes for two-byte elements. Other traffic and tile-fit constraints remain.',
  },
  'content-moderation-governed-decision-loop': {
    title: 'A score selects an action only through a declared policy',
    description: 'Under explicitly illustrative policy P7, score 0.62 falls between review thresholds 0.40 and 0.80 and therefore enters human review. The corrected decision retains policy version and reviewer provenance. Allowed-item audits and appeals also contribute corrected records.',
    layout: moderation,
    caption: 'In this illustrative policy, 0.62 enters human review, not automatic action. Audits and appeals preserve policy and reviewer provenance. These thresholds are not universal.',
  },
  'paper-critique-prioritization-funnel': {
    title: 'Four more accuracy points came with twice the compute',
    description: 'The reported baseline uses 10 GPU-hours for 80 percent accuracy; the new recipe uses 20 for 84 percent. Separate zero-based common scales show accuracy and compute. Three proposed matched-compute paired runs have question marks because their outcomes are not known.',
    layout: paper,
    caption: 'The report changes accuracy and compute together. Three paired, matched-compute runs test whether a useful gain survives. Their results are unknown, not implied by the drawing.',
  },
  'point-in-time-availability-boundary': {
    title: 'An earlier event can still be unavailable at prediction time',
    description: 'The Monday-noon cutoff divides knowable evidence from the future. Payment A occurs Monday at 09:00 and is available at 09:05. Payment B occurs at 10:00 but is unavailable until Wednesday. Its early event timestamp cannot authorize using its later-arriving information.',
    layout: availability,
    caption: 'Payment B happened before noon but arrived on Wednesday. An event-time-only join leaks future availability. The layout encodes the cutoff, not proportional elapsed time.',
  },
  'frontier-training-divergence-zoom': {
    title: 'Find the first bad cell, then zoom into its cause',
    description: 'A fictional four-step, four-rank grid first becomes non-finite at rank 2, step 3. That cell is traced to layer 6 and a gradient tensor. A one-condition replay tests a candidate explanation; recovery from a known-good checkpoint remains a separate task.',
    layout: divergence,
    caption: 'This fictional grid localizes the first failure; it does not prove its cause. Change one condition in replay. Recovery and diagnosis remain separate.',
  },
  'model-not-learning-diagnostic-path': {
    title: 'Separate a broken training path from a full-data generalization gap',
    description: 'Three illustrative loss plots share a zero-to-two vertical scale. One tiny fixed batch reaches an attainable minimum; another stays stuck and requires training-path checks. A full-data example improves training while validation diverges. The gap is diagnostic evidence, not proof of a particular cause.',
    layout: losses,
    caption: 'First check whether the tiny batch has an attainable minimum. A widening full-data gap suggests hypotheses, not a unique cause. The curves are illustrative.',
  },
  'offline-online-gap-diagnostic-order': {
    title: 'The missing 100 users change which arm wins',
    description: 'Both illustrative arms contain 1,000 assigned users. A has 800 successes. B routes 900 users to the model, with 738 successes, and 100 to fallback, with 40 successes. The 900-to-100 strip is proportional. B therefore has 778 successes, 77.8 percent overall, despite 82 percent among model-delivered users.',
    layout: delivered,
    caption: 'The denominator is everyone assigned, including fallback users. B has 738 + 40 successes, so 77.8% loses to 80%. These counts are illustrative.',
  },
  'training-loop-persistent-state': {
    title: 'Gradients and train/eval mode persist independently',
    description: 'Aligned uncorrected and corrected checkpoints distinguish gradient storage from module mode. Without clearing, batch two adds g2 to stored g1. Explicit zero_grad clears that state first. Separately, eval mode persists until train is called. Clearing gradients does not restore training mode.',
    layout: gradientMode,
    caption: 'zero_grad clears gradient storage; train restores module mode. They repair different persistent states. The corrected batch-two gradient excludes stale g1.',
  },
  'work-prioritization-counterfactual-portfolio': {
    title: 'Every attractive project spends the same remaining capacity',
    description: 'A toy 100-unit capacity ledger reserves 20 mandatory units, leaving 80. Project A costs all 80 and has assumed value 100. Projects B and C cost 45 and 35 and have assumed combined value 115. The two portfolios occupy equal total capacity. Choosing B plus C displaces A, not mandatory work.',
    layout: portfolio,
    caption: 'Reserve obligations before comparing portfolios. Under these toy assumptions, B + C beats A at equal capacity. Revisit when evidence changes the assumptions.',
  },
  'policy-label-observation-loop': {
    legacyFragmentIds: ['policy-label-visual-title'],
    title: 'Pending, censored, and policy-unobserved are not negative labels',
    description: 'Aligned schematic timelines share an evaluation cutoff. They distinguish an early known positive, a selected action whose outcome window remains open, a selected action whose observation ends early, and an action never selected by the policy. A task-specific maturity rule controls eligibility; unknown outcomes do not become negatives.',
    layout: labelHorizons,
    caption: 'These schematic rows distinguish four observation states at one cutoff. An early positive may be known. Other eligibility depends on the declared maturity rule.',
  },
  'agent-safety-authority-loop': {
    title: 'An impact limit must stop the next capability before execution',
    description: 'An illustrative impact ledger already contains 60 units and receives a proposed 50-unit action. A 100-unit policy limit would be exceeded because 60 plus 50 equals 110. The failing branch freezes future capabilities before dispatch. The proposed action is not counted as an already committed effect.',
    layout: () => gate({ example: 'Illustrative impact policy; limit 100', input: 'Committed impact 60\nproposed action +50', test: '60 + 50 = 110 > 100\naggregate limit is exceeded', yesLabel: 'WITHIN LIMIT', noLabel: 'THIS REQUEST', yes: 'recheck authority\nissue bounded capability', no: 'FREEZE authority\nno capability\nno dispatch' }),
    caption: 'The proposed action would cross the illustrative limit. Freeze later capabilities before execution. A monitor alert alone cannot enforce the stop.',
  },
  'enterprise-agent-three-plane-boundary': {
    title: 'Persist permission before dispatch; reconcile a lost reply instead of paying twice',
    description: 'One illustrative 50-unit credit preserves credit-123 and downstream key k-456 across control, durable evidence, dispatch, a committed effect with a lost reply, and authoritative reconciliation. Denied actions or failed durable writes stop before dispatch. Unknown transport outcome does not permit a second logical credit.',
    layout: enterprise,
    caption: 'The durable authorized record precedes dispatch. Denial or a failed write stops there. A lost reply triggers same-key reconciliation, not another logical credit.',
  },
  'distributed-recovery-consistency-gate': {
    title: 'A live rank is not proof of a coherent training state',
    description: 'An explicitly illustrative comparison shows a mixed state with rank zero at step 100 and rank one at step 101. Restarting one rank alone leaves that mismatch. The contrasting rollback restores model, optimizer, RNG, and data position from committed checkpoint 98 across the consistency domain. Supported narrow recovery remains valid only when it proves the same state contract.',
    layout: () => comparison({ example: 'Illustrative mixed state: rank 0 = 100; rank 1 = 101', left: 'RESTART ONE', right: 'RESTORE DOMAIN', rows: [
      { label: 'Rank 0 model / optimizer step', left: '100 / 100', right: '98 / 98' },
      { label: 'Rank 1 model / optimizer step', left: '101 / 101', leftTone: 'warning', right: '98 / 98' },
      { label: 'RNG + data position + integrity', left: 'not proved coherent', leftTone: 'warning', right: 'same committed 98' },
    ] }),
    caption: 'A single-rank restart leaves the pictured mismatch. Restore and validate the consistency domain. Narrow recovery is valid only when its supported protocol proves coherence.',
  },
  'feature-store-two-time-contract': {
    title: 'The historical join and the live lookup answer different time questions',
    description: 'For illustrative entity E, value 3 occurs at 11:00 and is available at 11:05. Value 7 occurs at 11:50 but is available at 12:10. A noon historical query must return 3. A live 12:15 lookup can return 7. Event time alone is insufficient for the historical contract.',
    layout: featureVersions,
    caption: 'At noon, version B is not available, despite its earlier event time. The later live lookup legitimately returns another version. Values are illustrative.',
  },
  'foundation-data-deletion-blast-radius': {
    title: 'Removing a source from tomorrow\'s shards does not erase yesterday\'s model influence',
    description: 'An illustrative lineage connects source D7, shard S2, training R3, checkpoint C4, and released model M5. A new build can exclude D7 while M5 still has unresolved influence. Model intervention and its validation are separate from rewriting future data artifacts.',
    layout: deletion,
    caption: 'Follow D7 through the historical lineage before excluding it from new shards. Existing model influence remains unresolved until separately addressed and validated.',
  },
  'adaptive-red-team-release-loop': {
    title: 'One blocked path leaves untested paths open questions',
    description: 'A fictional private read-only canary scenario compares three access paths before and after a mitigation. The reproduced direct-request failure becomes blocked. Quoted-text and tool-output paths remain question marks, not assumed successes. Adaptive coverage and residual-risk evidence still govern release.',
    layout: redTeam,
    caption: 'This fictional grid distinguishes a reproduced repair from missing coverage. Question marks are unknown outcomes. One blocked path cannot certify the whole system.',
  },
  'ml-ab-test-launch-gates': {
    title: 'Statistically positive, worthwhile, and safe are three different tests',
    description: 'Under an illustrative precommitted one-point worthwhile threshold, interval 0.2 to 0.8 is positive but insufficient. Interval 1.1 to 1.8 is worthwhile but its two-percent error rate exceeds a one-percent guardrail. Both hold. If assignment validity fails, neither effect is interpretable.',
    layout: experiment,
    caption: 'A positive interval can miss the worthwhile threshold. A worthwhile effect can fail safety. If assignment is invalid, do not interpret either effect.',
  },
  'ml-monitoring-observability-layers': {
    title: 'Freshness can fail now while outcome quality is still unknown',
    description: 'At noon, an 11:40 snapshot is twenty minutes old and fails a two-minute freshness contract immediately. Outcome evidence matures after a seven-day window and requires request identity and join coverage. A changed score distribution does not establish changed quality or causal product impact.',
    layout: monitoring,
    caption: 'The stale snapshot breaches a contract immediately. Outcome quality needs mature labels and adequate joins. Score drift alone proves neither quality nor causal impact.',
  },
  'fixed-budget-ranking-funnel': {
    title: 'Spend expensive scoring only after shrinking the candidate set',
    description: 'Explicitly illustrative candidate counts shrink from 10,000 to 1,000 to 100. Widths communicate ordered narrowing rather than a linear count scale. The separate request-level target remains p95 below 80 milliseconds and cost at most 0.08 dollars per 1,000 requests. Stage p95s are not added as if they were request p95.',
    layout: rankingBudget,
    caption: 'The counts are illustrative, and funnel widths are schematic. Measure latency and cost for the whole request. Under overload, reduce work or use fallback.',
  },
  'multi-team-platform-contract-boundary': {
    title: 'The first valid commit wins; a late worker cannot overwrite it',
    description: 'Attempts A and B share run R at expected version zero. A commits artifact A and advances the run to version one. B then fails its stale version-zero compare-and-swap and retains only attempt evidence. Deployment key K retries the same D1, while writer epochs remain separately checked.',
    layout: commitRace,
    caption: 'A advances the version before B arrives. B cannot replace the winning artifact. Deployment idempotency and writer authority remain separate contracts.',
  },
  'llm-inference-two-budget-scheduler': {
    title: 'Memory stays reserved while iteration work starts fresh',
    description: 'Toy KV memory has twelve pages: A reserves five, B reserves four, and three remain free. A separate eight-slot iteration protects three decode slots and uses five for bounded prefill. When A stops, its five pages are released, leaving eight free. These units are illustrative, not performance measurements.',
    layout: scheduler,
    caption: 'KV pages persist across iterations; token-work slots refresh each iteration. Stopping A releases its reservation for later admission. All units are illustrative.',
  },
  'multimodal-stale-frame-gate': {
    title: 'Recent arrival cannot make an old screen fresh',
    description: 'On one mapped clock the screen is captured at 10.00 seconds, arrives at 14.20, and is considered at 14.30. Actual age is 4.30 seconds; arrival age is only 0.10. Their bars use a shared seconds-per-unit scale. A replacement snapshot belongs to a separate later decision with its own 0.068-second age bound and action revalidation.',
    layout: frameAge,
    caption: 'Capture time determines the old frame\'s age. Arrival time hides the stall. A replacement snapshot needs its own clock bound, decision, and authorization checks.',
  },
  'reasoning-fixed-budget-reclaim': {
    title: 'A quarterly allocation is not the same as nodes available right now',
    description: 'The usable quarterly envelope is 248,832 accelerator-hours, split into 65 percent protected and 35 percent reclaimable offline claims. A separately labeled instantaneous eight-node toy example moves two offline nodes back to serving after checkpoint and pause. The toy node ratio is not a conversion of the quarterly percentages.',
    layout: reclaim,
    caption: 'The upper bar allocates quarter-hours. The separate toy node strip shows reclamation at one instant. Protected recovery is not discretionary experiment capacity.',
  },
  'ecosystem-experiment-interference': {
    title: 'Treatment and control can change the same future creator supply',
    description: 'Separate viewer treatment and control cells allocate opportunity in one shared creator pool. Creator response over weeks changes future supply that returns to both cells. The cross-cell dependence is explicit, but the direction of benefit or harm is not assumed.',
    layout: interference,
    caption: 'Both cells influence the same creator pool. Its later response reaches both future cells. A short viewer experiment does not isolate an untouched ecosystem.',
  },
  'spotify-two-level-shelf-composition': {
    title: 'Two strong shelves can make a weaker page when they repeat the same song',
    description: 'Toy shelves A, B, and C contain song IDs one-two, one-three, and four-five, with local scores six, five, and four. Repeated song one creates the declared four-unit overlap penalty for A plus B, giving seven. A plus C gives ten and B plus C nine. Membership does not determine display order.',
    layout: shelves,
    caption: 'Song 1 appears in both A and B. The declared overlap penalty makes A + C the best pair. Shelf order remains unresolved.',
  },
  'youtube-recommender-selective-feedback-loop': {
    title: 'Ranking 900 candidates does not reveal 900 outcomes',
    description: 'An illustrative candidate pool contains 1,000 unique candidates, 900 eligible for ranking, and ten displayed positions. The other 890 eligible candidates remain unshown with unknown outcomes. Only exposure-linked mature outcomes enter this feedback path, with policy and selection metadata retained.',
    layout: exposed,
    caption: 'Ten displayed items produce exposure evidence. The other 890 eligible items remain unknown, not negative. Preserve policy and support when learning from this selection.',
  },
  'disagreement-evidence-commit-loop': {
    title: 'Change the decision when evidence changes the shared constraint',
    description: 'A fictional technical disagreement compares a faster high-memory cache with a smaller cache under a shared eight-gigabyte limit. A nine-gigabyte measured toy footprint rules out the first alternative under that constraint. The decision owner chooses the feasible option, records a memory-based revisit trigger, and retains required safety review.',
    layout: () => gate({ example: 'Fictional cache discussion; shared limit 8 GB', input: 'A: faster cache, 9 GB\nB: smaller cache, 6 GB', test: 'Does the candidate meet the shared memory constraint?', yesLabel: 'B: 6 <= 8', noLabel: 'A: 9 > 8', yes: 'owner selects B\nrevisit if A fits\nrequired reviews stay', no: 'reject A for now\nnot a judgment\nof its advocate' }),
    caption: 'The shared constraint, not seniority, selects the feasible option in this fictional case. Record what evidence would reopen the choice. Required reviews remain.',
  },
  'encoder-once-decoder-loop': {
    title: 'Three growing prefixes share one fixed encoder memory',
    description: 'One encoder-memory block feeds three successive decoder iterations. Prefixes grow from start to start-A to start-A-B. Cross-attention source K/V remain fixed while the causal decoder prefix and its cache grow. The shared input lines distinguish reuse from re-encoding.',
    layout: encoderMemory,
    caption: 'Each decode step reads the same encoded source. Only the output prefix and decoder-side cache grow. Fixed source K/V do not mean fixed decoder state.',
  },
  'agent-eval-diagnostic-flywheel': {
    title: 'A passing test step and a failing task outcome can coexist',
    description: 'A fictional coding-agent trial edits the intended file but then writes an unauthorized file. The intended test passes, the task safety outcome fails, and the trial uses four tool calls. The new unauthorized-write slice returns to the fixed-budget suite; a production mismatch reopens its coverage assumptions.',
    layout: () => comparison({ example: 'Fictional trial T4; same trajectory, three lenses', left: 'WHAT HAPPENED', right: 'WHAT IT PROVES', rows: [
      { label: 'Outcome', left: 'intended test passes', right: 'not sufficient for success', rightTone: 'warning' },
      { label: 'Failure step', left: 'writes forbidden file', leftTone: 'warning', right: 'safety outcome FAIL', rightTone: 'warning' },
      { label: 'Resource ledger', left: '4 tool calls', right: 'compare at fixed budget', rightTone: 'state' },
    ] }),
    caption: 'Add the unauthorized-write slice to the suite. A production mismatch reopens coverage. Passing one test does not erase the same trial\'s safety failure.',
  },
  'search-ranker-evidence-flywheel': {
    title: 'The better average still fails the tail floor',
    description: 'Fixed illustrative head and tail scores are A 0.80 and 0.80, and B 0.90 and 0.70. Shared-scale bars show each matched slice. The 90-to-10 traffic mixture gives averages 0.80 and 0.88, but B fails the declared tail floor 0.78 and must hold.',
    layout: rankerSlices,
    caption: 'Traffic weighting improves B\'s average while its tail breaches the floor. Hold B and diagnose the slice. An aggregate improvement is not release permission.',
  },
  'foundation-data-curation-lineage': {
    title: 'Being listed in a manifest is not proof that a sample was consumed',
    description: 'Illustrative stable IDs separate accepted R17 from rejected R18. R18 keeps its policy-rejection reason and has no shard membership. R17 appears in S1 and M1, then has separate consumption evidence before checkpoint C200. Membership and consumption are different claims.',
    layout: dataLineage,
    caption: 'Rejected R18 retains its reason without entering a shard. Accepted R17 needs a consumption record, not just manifest membership, before linking it to C200.',
  },
  'gan-alternating-update-ownership': {
    title: 'Freezing discriminator parameters must not detach the generator\'s gradient path',
    description: 'The discriminator step detaches generated samples, updating D but not G. The generator step freezes D parameters while a dashed loss-gradient path passes through D back into G. The two panels distinguish no parameter update from no gradient path.',
    layout: ganUpdates,
    caption: 'The discriminator step detaches G\'s output. The generator step instead differentiates through frozen D. No parameter update and no gradient path are different operations.',
  },
  'lora-zero-impact-first-gradient': {
    title: 'A zero initial adapter output does not mean both factors have zero gradient',
    description: 'At initialization A is random and B is zero, so the scaled adapter output sBAx is zero. For upstream gradient g, dL/dA equals s B-transpose g x-transpose and is zero. dL/dB equals s g (Ax)-transpose and is generally nonzero. Frozen W parameters do not block input gradients.',
    layout: loraGradients,
    caption: 'B = 0 makes the initial adapter output and A gradient zero. B can still receive a gradient. Frozen base weights do not detach inputs.',
  },
  'autograd-shared-node-accumulate-then-propagate': {
    title: 'Two branches add at the shared node before its gradient moves upstream',
    description: 'The forward DAG has x equal to 2, shared u equal to 3x or 6, a equal to u squared or 36, b equal to 2u or 12, and L equal to 48. Dashed reverse edges contribute 12 and 2 at the same u. Their total 14 is multiplied by 3 once to yield x gradient 42.',
    layout: autograd,
    caption: 'Solid arrows carry forward values. Dashed return paths contribute 12 and 2 to the same u. Accumulate 14, then propagate 14 x 3 once.',
  },
  'senior-principal-delegated-coherence': {
    title: 'Delegate local choices; escalate changes that cross a shared contract',
    description: 'A fictional runtime, evaluation, and safety ownership example contrasts every routine change waiting for one approver with local changes inside domain contracts. A changed tool-result schema crosses runtime and evaluation ownership and therefore enters joint review. Evidence can reopen the shared doctrine.',
    layout: () => comparison({ example: 'Fictional runtime / evaluation / safety ownership', left: 'ONE APPROVER', right: 'DOMAIN OWNERS', rows: [
      { label: 'Routine runtime change', left: 'waits in central queue', leftTone: 'warning', right: 'runtime owner decides' },
      { label: 'Routine evaluation change', left: 'same central queue', leftTone: 'warning', right: 'evaluation owner decides' },
      { label: 'Tool-result schema change', left: 'central bottleneck', leftTone: 'warning', right: 'joint interface review', rightTone: 'focus' },
    ] }),
    caption: 'Local changes stay with domain owners. A shared-schema change crosses boundaries and requires joint review. New evidence can reopen the common doctrine.',
  },
  'healthcare-llm-clinician-gate': {
    title: 'The correct patient is not enough: the encounter and draft version must also match',
    description: 'Patient P1 has encounter E1 on January 10 without a prior procedure and E2 on February 10 with procedure PX1 on February 9. An E2 draft citing E1 is rejected. A corrected draft version v2 cites E2, and the reviewer receipt binds that exact version. Documentation permission does not grant clinical-action authority.',
    layout: () => comparison({ example: 'Illustrative P1; target encounter E2, February 10', left: 'STALE DRAFT v1', right: 'CORRECTED v2', rows: [
      { label: 'Source encounter', left: 'E1: January 10', leftTone: 'warning', right: 'E2: February 10' },
      { label: 'Procedure claim', left: 'no prior procedure', leftTone: 'warning', right: 'PX1 on February 9' },
      { label: 'Authorized documentation', left: 'REJECT mismatch', leftTone: 'warning', right: 'review receipt binds v2' },
    ] }),
    caption: 'Match patient, encounter, date, and claim. The review receipt binds the corrected draft version. Documentation authority does not permit autonomous clinical action.',
  },
  'llm-security-influence-authority-path': {
    title: 'Untrusted context can propose an action without acquiring authority to execute it',
    description: 'Untrusted context proposes export of B7 from tenant B to an external mock destination. Verified authority A1 permits only a tenant-A internal summary. Exact action, tenant, resource, destination, and purpose do not match, so the proposal ends before credentials or dispatch. Answer, cache, and log disclosure still need separate inspection.',
    layout: () => gate({ example: 'Untrusted proposal != verified authority A1', input: 'Proposal: export B7 / tenant B\nexternal mock destination', test: 'A1 permits tenant A\ninternal summary only\nexact fields do not match', yesLabel: 'MATCH REQUIRED', noLabel: 'THIS PROPOSAL', yes: 'scope still bounded\ninspect answer,\ncache, and logs', no: 'DENY\nno credential\nno dispatch' }),
    caption: 'The proposed action cannot borrow authority from retrieved text. Its fields fail the verified scope. Blocking tools does not settle answer, cache, or log disclosure.',
  },
  'training-recipe-evidence-gates': {
    title: 'A nonzero gradient is not proof that the intended parameters changed',
    description: 'An illustrative known 32-example batch has flat loss 1.10, nonzero gradient norm 0.8, and zero maximum parameter change. The diagnostic checks the optimizer step, learning rate, and optimizer-bound parameter identities, then repeats the same fitting check before escalating to full data. The evidence narrows hypotheses without proving one cause.',
    layout: trainingGate,
    caption: 'The gradient exists, but the intended weights do not move. Inspect update wiring and repeat the same batch. This evidence does not uniquely identify a cause.',
  },
  'personalized-search-request-learning-loop': {
    title: 'Reranking cannot recover an exact result discarded by candidate fusion',
    description: 'Exact-title candidate A has lexical rank one and no semantic rank. Candidate B has lexical rank two and semantic rank one. At RRF offset sixty, their scores are 0.01639 and 0.03252. An unprotected cap of one removes A before later reranking. A protected eligible quota or supported larger merge is a distinct alternative.',
    layout: rrfSurvival,
    caption: 'A disappears before either reranker. An eligible exact-intent quota or supported larger merge must preserve it earlier. Protection does not override eligibility or policy.',
  },
  'personalization-three-feature-clocks': {
    title: 'A fast request can still assemble stale features',
    description: 'A declared illustrative noon request uses an 08:00 batch snapshot with a 24-hour limit, a stream value whose event time is 11:57 and arrival 11:59 with a two-minute event-age limit, and current request context. The batch age is four hours and passes; stream event age is three minutes and fails despite one-minute arrival age. Request execution latency is separate.',
    layout: () => comparison({ example: 'Illustrative request at noon; declared age contracts', left: 'SOURCE CLOCK', right: 'AGE / CONTRACT', rows: [
      { label: 'Batch snapshot', left: '08:00', right: '4 h <= 24 h' },
      { label: 'Stream event / arrival', left: '11:57 / 11:59', right: '3 min > 2 min', rightTone: 'warning' },
      { label: 'Live request context', left: '12:00', right: 'current; age 0' },
    ] }),
    caption: 'The stream value arrived recently but describes an older event. Enforce each feature\'s declared clock contract. Request latency is not source freshness.',
  },
  'inference-cost-measure-test-compound-loop': {
    title: 'Count only savings that survive the same quality gates',
    description: 'Toy fixed-mix cost per successful request starts at ten units. Change one reduces it to eight and passes required quality slices, so eight becomes the new baseline. Change two reaches six but fails a tail-quality gate and is rejected. Accepted reduction is ten over eight, 1.25-fold, not an unsupported tenfold claim.',
    layout: inferenceCost,
    caption: 'Keep change 1 and reject change 2. The accepted baseline is 8 units per successful request. Cost reduction is 1.25x, not a fictional 10x.',
  },
  'ambiguous-problem-scope-loop': {
    title: 'A complaint sample suggests a question; a bounded check decides the next scope',
    description: 'A fictional selected complaint sample contains thirty source, fifteen retrieval, and five synthesis complaints. It is not a representative prevalence estimate. The named missing fact is whether permitted, current source evidence exists for ten sampled cases. A predeclared one-day availability check determines a bounded v1 source repair or a stop, rather than automatically proposing a model change.',
    layout: () => gate({ example: 'Fictional selected sample: 30 source / 15 retrieval / 5 synthesis', input: 'Missing fact: permitted, current evidence exists for 10 sampled cases?', test: 'One-day availability check\npredeclare evidence rule\nselected sample != prevalence', yesLabel: 'EVIDENCE FOUND', noLabel: 'NOT AVAILABLE', yes: 'v1: source repair\nmodel tuning deferred\nowner + review date', no: 'stop this proposal\nor change scope\nrecord missing fact' }),
    caption: 'The selected complaints do not establish population prevalence. Resolve the named availability question within the agreed timebox. Its receipt changes scope or stops the proposal.',
  },
};

export const supplementalRedraws = {
  'ecosystem-serving-stages': {
    slug: 'annotated-ecosystem-strategy-mock',
    title: 'Retrieved is not the same as displayed',
    description: 'An illustrative request retrieves one hundred unique candidates, retains twenty after eligibility and policy checks, and displays five. Only those five positions produce exposure records. Fresh-supply sources join retrieval without bypassing eligibility, and experiment denominators use actual exposure.',
    layout: () => comparison({ example: 'Illustrative one-request counts; policy applies throughout', left: 'CANDIDATE STATE', right: 'WHAT IS OBSERVED', rows: [
      { label: 'Retrieve + merge fresh sources', left: '100 unique candidates', right: 'not exposures', rightTone: 'neutral' },
      { label: 'Eligibility + policy', left: '20 eligible', right: 'still not exposures', rightTone: 'neutral' },
      { label: 'Compose + display', left: '5 displayed positions', right: '5 exposure records' },
    ] }),
    caption: 'Candidate counts cannot replace exposure counts. Only the displayed positions enter this request\'s exposure denominator. The counts are illustrative, not production measurements.',
  },
  'coding-product-mode-boundaries': {
    slug: 'design-ai-coding-product',
    title: 'Shared context does not grant shared execution authority',
    description: 'Suggestion and durable-agent modes share permission-aware context, including dirty-buffer revision. Suggestion mode returns text without tools. Agent mode may use bounded tools in an isolated workspace. Build and test evidence returns a reviewable patch, and neither mode applies the patch without the required user decision.',
    layout: () => comparison({ example: 'Same permitted context; different authority', left: 'SUGGESTION', right: 'DURABLE AGENT', rows: [
      { label: 'Context revision', left: 'dirty buffer v42', right: 'dirty buffer v42', rightTone: 'input' },
      { label: 'Execution', left: 'text only; no tools', right: 'scoped tools\nisolated workspace', rightTone: 'focus' },
      { label: 'Patch application', left: 'user decides', right: 'tests + review\nuser decides' },
    ] }),
    caption: 'The agent\'s scoped tool authority is separate from shared context access. Passing tests returns evidence, not permission to apply a patch automatically.',
  },
  'coding-product-task-lifecycle': {
    slug: 'design-ai-coding-product',
    title: 'A cancellation request is not proof that the worker stopped',
    description: 'A cancellation request removes permission for new dispatch, but the worker may still be stopping. Confirmed worker-stop evidence establishes the terminal cancellation state. Separately, passing verification leaves a patch waiting for the user decision rather than applied automatically.',
    layout: () => comparison({ example: 'Commands and observed terminal states are different', left: 'REQUEST / CHECK', right: 'REQUIRED EVIDENCE', rows: [
      { label: 'Cancel requested', left: 'stop new dispatch', right: 'worker may be stopping', rightTone: 'warning' },
      { label: 'Worker confirms stop', left: 'stop receipt arrives', right: 'cancelled terminal state' },
      { label: 'Verification passes', left: 'tests + diff available', right: 'still awaits user decision', rightTone: 'focus' },
    ] }),
    caption: 'Stop issuing work when cancellation is requested. Wait for worker-stop evidence before declaring cancellation complete. Verification still does not apply the patch.',
  },
  'foundation-data-build-planes': {
    slug: 'design-foundation-model-data-platform',
    title: 'Completed shards are only one input to publication',
    description: 'Illustrative build B8 produces exact shard versions S1 and S2. Publication requires both current use-policy permission and passing validation evidence. Either denial or a failed check holds the build. A successful immutable manifest binds exact shards, transform versions, and authorized use for training.',
    layout: () => gate({ example: 'Illustrative build B8: shards S1 + S2', input: 'Artifacts built\ntransform versions recorded', test: 'Current use policy permits\nAND validation checks pass?', yes: 'publish immutable M8\nexact shard versions\ntraining binds M8', no: 'HOLD B8\nno publication\nretain failure reason' }),
    caption: 'A completed build does not authorize use. Policy permission and checks must both pass before an immutable manifest is published.',
  },
  'multimodal-path-boundaries': {
    slug: 'design-real-time-multimodal-assistant',
    title: 'A proposal is not permission',
    description: 'Toy T9 proposes muting room A. Separate identity, action, room, and active-epoch checks precede dispatch. Only an authoritative device receipt establishes whether mute occurred.',
    layout: () => gate({ example: 'Illustrative T9: mute room A', input: 'Recognized request + active turn epoch\nmodel proposes exact action', test: 'Identity, room, action, and current epoch permitted?', yes: 'scoped dispatch\ndevice receipt\nreport actual state', no: 'no dispatch\nclarify or decline\nretain the decision' }),
    caption: 'Authorize; verify device state.',
  },
  'multimodal-turn-lifecycle': {
    slug: 'design-real-time-multimodal-assistant',
    title: 'Interruption must retire old output',
    description: 'Interruption advances active epoch seven to eight. Epoch-seven output is rejected immediately. Cancellation is requested separately; only playback-stop evidence proves silence. New output carries epoch eight.',
    layout: () => comparison({ example: 'Illustrative interruption: active epoch 7 -> 8', left: 'CONTROL EVENT', right: 'OUTPUT STATE', rows: [
      { label: 'User interrupts', left: 'activate epoch 8', right: 'reject epoch-7 output', rightTone: 'warning' },
      { label: 'Cancel epoch 7 requested', left: 'command sent', right: 'not yet proof of silence', rightTone: 'warning' },
      { label: 'Playback-stop receipt', left: 'old output stopped', right: 'new output uses epoch 8' },
    ] }),
    caption: 'Confirm playback stopped.',
  },
  'reasoning-admission-and-capacity': {
    slug: 'design-reasoning-model-fixed-budget',
    title: 'A desired model route is not an admitted reservation',
    description: 'A declared toy reasoning request needs four compute slots and six KV pages. Six compute slots are free, but only five KV pages are available. The conjunction fails and the request queues within its deadline, reduces its supported bound, or declines without reservation. Passing requests reserve both resources before execution.',
    layout: () => gate({ example: 'Toy request needs compute 4 AND KV pages 6', input: 'Available: compute 6\nKV pages 5', test: '4 <= 6 is true\n6 <= 5 is false\nboth resources required', yesLabel: 'BOTH FIT', noLabel: 'THIS REQUEST', yes: 'reserve compute + KV\nthen execute\nbounded work', no: 'queue within deadline\nreduce or decline\nno phantom reservation' }),
    caption: 'Free compute cannot substitute for missing KV pages. This request is not admitted. Reserve both resources before execution when both bounds fit.',
  },
  'video-ecosystem-delayed-effects': {
    slug: 'design-short-form-video-ecosystem',
    title: 'The immediate viewer signal arrives before the full supply response',
    description: 'A shared schematic horizon separates viewer response observed inside a short experiment from creator retention and production observed weeks later. Eligibility and monetization can also affect supply, but no causal direction or effect size is invented.',
    layout: feedback,
    caption: 'Viewer response and creator supply arrive on different horizons. Eligibility and funding may affect supply too. This drawing assumes neither a benefit nor a harm.',
  },
  'video-publication-evidence': {
    slug: 'design-short-form-video-ecosystem',
    title: 'A release binds compatible versions, and rollback must bind them too',
    description: 'Illustrative release R12 binds model M4, candidate index I8, feature schema F3, and policy P6 after current authorization and evaluation. Rollback restores a validated compatible bundle, not just an old model file. Artifact lineage and actual exposure records remain separate evidence.',
    layout: () => comparison({ example: 'Illustrative versioned release R12', left: 'RELEASE BUNDLE', right: 'ROLLBACK CONTRACT', rows: [
      { label: 'Model + retrieval index', left: 'M4 + I8', right: 'validated compatible pair' },
      { label: 'Feature schema + policy', left: 'F3 + P6', right: 'versions still permitted' },
      { label: 'Evidence', left: 'evaluation + authority', right: 'retain lineage\ncheck actual exposures' },
    ] }),
    caption: 'Deploy and roll back compatible versioned bundles. An older model file alone is insufficient. Exposure outcomes require their own records.',
  },
  'stairs-final-move-partition': {
    slug: 'climbing-stairs',
    title: 'Eight paths split cleanly by their final move',
    description: 'All eight sequences of one-step and two-step moves totaling five are drawn to a shared five-step width. Five end in one and three end in two. Their highlighted final moves form disjoint exhaustive groups, giving the recurrence five plus three equals eight.',
    layout: stairPaths,
    caption: 'Each row totals five steps. The highlighted final move places it in exactly one group. Five ending in +1 plus three ending in +2 gives eight.',
  },
  'course-prerequisite-fanout': {
    slug: 'course-schedule',
    title: 'One finished prerequisite does not make course 3 ready',
    description: 'The diamond graph has edges zero to one and two, and both one and two to three. Course three starts with two unmet prerequisites. Emitting one parent leaves one; only emitting both reduces the count to zero and makes three ready.',
    layout: prerequisite,
    caption: 'Course 3 has two incoming requirements. Its unmet count changes 2 -> 1 -> 0 as the parents finish. Ready is not the same as already completed.',
  },
  'robber-take-skip-boundaries': {
    slug: 'house-robber',
    title: 'Taking house 3 can reuse only the optimum through house 1',
    description: 'The pictured house values are two, seven, nine, and three at indices zero through three. Skipping house three keeps houses zero and two, totaling eleven. Taking house three can add only the best through index one, seven, giving ten. The adjacent houses two and three are never combined.',
    layout: houses,
    caption: 'The take branch stops its reusable prefix two houses back. The skip branch keeps eleven. Their maximum is eleven, without using adjacent houses.',
  },
  'lcs-conditional-parent-cells': {
    slug: 'longest-common-subsequence',
    title: 'A match takes the diagonal; a mismatch compares the two shorter prefixes',
    description: 'The actual LCS table for AB versus ACB contains final value two. Matching B with B uses diagonal value one plus one. The B-versus-C mismatch instead compares up one and left one to retain one. The arrow connects the actual matched diagonal parent to its result.',
    layout: lcsCells,
    caption: 'At B = B, extend the diagonal value to two. At B != C, keep the larger shorter-prefix value, one. The parents solve different prefix questions.',
  },
  'subset-sum-old-snapshot': {
    slug: 'partition-equal-subset-sum',
    title: 'Add the new item to the old snapshot, not to sums just created',
    description: 'The old reachable set zero-one adds the new item five exactly once, creating five-six. Only after that computation does union produce zero-one-five-six. Ten is absent because the newly created five is not fed back into the same item update.',
    layout: () => comparison({ example: 'New item 5; one-use update', left: 'OLD SNAPSHOT', right: 'ADD 5 ONCE', rows: [
      { label: 'Each source sum', left: '0\n1', right: '5\n6' },
      { label: 'Union after computation', left: 'retain {0, 1}', right: '{0, 1, 5, 6}' },
      { label: 'Forbidden feedback', left: 'do not reuse new 5', right: '10 is NOT created', rightTone: 'warning' },
    ] }),
    caption: 'New sums use only the old snapshot. Union happens afterward. Feeding new 5 back into this same update would incorrectly reuse the item.',
  },
  'dsu-edges-versus-parents': {
    slug: 'redundant-connection',
    title: 'Path compression changes the bookkeeping, not the original graph',
    description: 'The accepted problem graph remains the chain one-two-three-four. Before find four, DSU parents include four to three to one. After compression four points directly to one; two and three still point to one. No accepted graph edge is changed.',
    layout: dsuCompression,
    caption: 'The top graph is unchanged. Only the parent pointer for 4 shortens after find. A DSU parent arrow is not an accepted problem-graph edge.',
  },
  'subsets-increasing-index-tree': {
    slug: 'subsets',
    title: 'Increasing indices choose membership without emitting every ordering',
    description: 'All eight subsets of one-two-three appear once in the increasing-index tree. A proposed two-one branch is rejected because it returns to an earlier index. This excludes duplicate orderings without excluding the valid subset one-two.',
    layout: subsetOrder,
    caption: 'Each extension uses a later index. The rejected [2,1] ordering represents membership already emitted as [1,2]. All eight distinct subsets remain.',
  },
};

export function renderRedraw(identity, model = redraws[identity] ?? supplementalRedraws[identity]) {
  assert(model, `Missing illustration model: ${identity}`);
  const { body, height: authoredHeight } = model.layout();
  let height = authoredHeight;
  function reserveText(node) {
    if (node.nodeName === 'text') {
      const attributes = Object.fromEntries((node.attrs ?? []).map((item) => [item.name, item.value]));
      const fontSize = Number(attributes.style?.match(/font-size:([\d.]+)px/)?.[1] ?? 14);
      const lineOffsets = (node.childNodes ?? []).filter((child) => child.nodeName === 'tspan')
        .reduce((sum, child) => sum + Number(child.attrs?.find((attribute) => attribute.name === 'dy')?.value ?? 0), 0);
      height = Math.max(height, Number(attributes.y) + lineOffsets + fontSize * .4 + 12);
    }
    for (const child of node.childNodes ?? []) reserveText(child);
  }
  reserveText(parseFragment(`<svg>${body}</svg>`));
  height = Math.ceil(height);
  const prefix = `${identity}-edition`;
  assert(Number.isFinite(height) && height > 0);
  const supplementalAttribute = Object.hasOwn(supplementalRedraws, identity) ? ` data-supplemental-visual="${identity}"` : '';
  const printPageAttribute = model.printPage ? ' data-illustration-print-page' : '';
  const legacyAnchors = (model.legacyFragmentIds ?? []).map((id) => `<span class="visual-fragment-anchor" id="${escape(id)}" aria-hidden="true"></span>`).join('');
  return `<figure class="learning-figure editorial-illustration"${supplementalAttribute}${printPageAttribute} aria-labelledby="${prefix}-title">${legacyAnchors}<p class="visual-title" id="${prefix}-title">${escape(figureTakeaways[identity] ?? model.title)}</p><svg viewBox="0 0 360 ${height}" role="img" aria-labelledby="${prefix}-svg-title ${prefix}-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="${prefix}-svg-title">${escape(model.title)}</title><desc id="${prefix}-svg-desc">${escape(model.description)}</desc>${body}</svg><figcaption><strong>Read it this way:</strong> ${escape(model.caption)}</figcaption></figure>`;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const projection = parseFragment(`<svg>${pcaProjection().body}</svg>`);
  const samples = [];
  const scores = [];
  let selectedSamples = 0;
  let rightAngles = 0;
  function checkProjection(node) {
    const attributes = Object.fromEntries((node.attrs ?? []).map((attribute) => [attribute.name, attribute.value]));
    if (attributes['data-pca-score'] !== undefined) {
      const score = Number(attributes['data-pca-score']);
      assert(Math.abs((Number(attributes.cx) - 180) / .85 - score) < 1e-10, 'PCA score must use the declared one-dimensional coordinate scale');
      assert.equal(Number(attributes.cy), 400);
      scores.push(score);
    }
    if (attributes['data-pca-selected'] !== undefined) {
      assert.equal(attributes['data-pca-along'], '85');
      assert.equal(attributes['data-pca-residual'], '-20');
      selectedSamples++;
    }
    if (attributes['data-pca-right-angle'] !== undefined) {
      const [firstX, firstY, cornerX, cornerY, lastX, lastY] = attributes.d.match(/-?\d+(?:\.\d+)?/g).map(Number);
      assert(Math.abs((firstX - cornerX) * (lastX - cornerX) + (firstY - cornerY) * (lastY - cornerY)) < 1e-10, 'PCA angle marker must draw a right angle');
      assert(Math.hypot(firstX - cornerX, firstY - cornerY) > 6, 'PCA angle marker must not collapse');
      rightAngles++;
    }
    if (attributes['data-pca-along']) {
      const along = Number(attributes['data-pca-along']);
      const residual = Number(attributes['data-pca-residual']);
      const segment = node.childNodes.find((child) => child.nodeName === 'path');
      const coordinates = segment.attrs.find((attribute) => attribute.name === 'd').value.match(/-?\d+(?:\.\d+)?/g).map(Number);
      const [sampleX, sampleY, projectionX, projectionY] = coordinates;
      const errorX = sampleX - projectionX;
      const errorY = sampleY - projectionY;
      const cosine = Math.sqrt(3) / 2;
      assert(Math.abs(errorX * cosine - errorY / 2) < 1e-10, 'PCA residual must be perpendicular to the retained axis');
      assert(Math.abs((projectionX - 180) / 2 + (projectionY - 218) * cosine) < 1e-10, 'PCA diamond must lie on PC1');
      assert(Math.abs((projectionX - 180) * cosine - (projectionY - 218) / 2 - .85 * along) < 1e-10, 'PCA projection must retain the original coordinate');
      assert(Math.abs((errorX ** 2 + errorY ** 2) / .85 ** 2 - residual ** 2) < 1e-8, 'PCA segment length must draw the stated reconstruction error');
      samples.push([along, residual]);
    }
    for (const child of node.childNodes ?? []) checkProjection(child);
  }
  checkProjection(projection);
  assert.equal(samples.length, 6);
  assert.deepEqual(scores, samples.map(([along]) => along));
  assert.equal(selectedSamples, 1);
  assert.equal(rightAngles, 1);
  assert.equal(samples.reduce((sum, [along, residual]) => sum + along * residual, 0), 0);
  assert.equal(samples.reduce((sum, [, residual]) => sum + residual ** 2, 0), 1200);
  assert(samples.reduce((sum, [along]) => sum + along ** 2, 0) > 1200);
  const attributesOf = (node) => Object.fromEntries((node.attrs ?? []).map((attribute) => [attribute.name, attribute.value]));
  const drawingNodes = (drawing) => {
    const nodes = [];
    function collect(node) {
      nodes.push(node);
      for (const child of node.childNodes ?? []) collect(child);
    }
    collect(parseFragment(`<svg>${drawing.body}</svg>`));
    return nodes;
  };
  const contrastive = drawingNodes(contrastiveMembership());
  const denominator = contrastive.find((node) => attributesOf(node)['data-infonce-denominator'] !== undefined);
  const candidates = contrastive.filter((node) => attributesOf(node)['data-infonce-candidate'] !== undefined);
  assert.equal(candidates.length, 4);
  assert(candidates.every((node) => node.parentNode === denominator), 'Every candidate, including the positive, belongs to the denominator');
  const numerator = candidates.filter((node) => attributesOf(node)['data-infonce-numerator'] !== undefined);
  assert.equal(numerator.length, 1);
  assert.equal(attributesOf(numerator[0])['data-infonce-candidate'], 'positive');
  assert.equal(candidates.reduce((sum, node) => sum + Number(attributesOf(node)['data-weight']), 0), 10);
  for (const candidate of candidates) assert.equal(candidate.childNodes.filter((node) => attributesOf(node)['data-infonce-unit'] !== undefined).length, Number(attributesOf(candidate)['data-weight']));
  const probabilityUnits = contrastive.filter((node) => attributesOf(node)['data-infonce-probability-unit'] !== undefined);
  assert.equal(probabilityUnits.length, 10);
  assert.equal(probabilityUnits.filter((node) => attributesOf(node).style.includes('--viz-input-bg')).length, 4);
  const ring = drawingNodes(concurrentRing());
  const messages = ring.filter((node) => attributesOf(node)['data-ring-chunk'] !== undefined).map(attributesOf);
  assert.deepEqual(messages.map((message) => message['data-ring-chunk']), ['A', 'B', 'C', 'D']);
  for (const [rank, message] of messages.entries()) {
    assert.equal(Number(message['data-sender']), rank);
    assert.equal(Number(message['data-receiver']), (rank + 1) % 4);
    assert.equal(message['data-chunk-bytes'], 'B/4');
  }
  for (const phase of ['reduce-scatter', 'all-gather']) {
    const steps = ring.filter((node) => attributesOf(node)['data-ring-phase'] === phase).map(attributesOf);
    assert.equal(steps.length, 3);
    for (const step of steps) assert.equal(Number(step['data-result']), Number(step['data-incoming']) + Number(step['data-local'] ?? 0));
    assert.equal(Number(steps.at(-1)['data-result']), 10);
  }
  assert.equal(ring.filter((node) => attributesOf(node)['data-ring-sent-chunk'] !== undefined).length, 6);
  const matmul = drawingNodes(missingMatmulProducts());
  const expectedProducts = [[19, 22], [43, 50]];
  for (const phase of ['before', 'after']) {
    const cells = matmul.filter((node) => attributesOf(node)['data-matmul-phase'] === phase).map(attributesOf);
    assert.equal(cells.length, 4);
    assert.equal(new Set(cells.map((cell) => `${cell['data-row']}:${cell['data-column']}`)).size, 4);
    for (const cell of cells) {
      const row = Number(cell['data-row']);
      const column = Number(cell['data-column']);
      assert.equal(cell['data-value'], phase === 'before' && row !== column ? 'missing' : String(expectedProducts[row][column]));
    }
  }
  const factorization = factorizationEntry().body;
  assert(factorization.includes('2 * 1.5 = 3') && factorization.includes('1 * 1 = 1') && factorization.includes('3 + 1 = 4'));
  const lookup = queryBeforeStore().body;
  assert(lookup.includes('9 - 7 = 2') && lookup.includes('Return [1, 2]: 2 + 7 = 9'));
  for (const [identity, model] of Object.entries({ ...redraws, ...supplementalRedraws })) {
    const source = renderRedraw(identity, model);
    assert.equal(source.includes('data-illustration-print-page'), model.printPage === true, `${identity}: print-page scope must follow the drawing model`);
    const ids = new Set();
    const references = [];
    function walk(node) {
      const attributes = Object.fromEntries((node.attrs ?? []).map((item) => [item.name, item.value]));
      if (attributes.id) {
        assert(!ids.has(attributes.id), `${identity}: duplicate ID`);
        ids.add(attributes.id);
      }
      if (attributes['aria-labelledby']) references.push(...attributes['aria-labelledby'].split(' '));
      if (node.nodeName === 'text') assert(!source.slice(node.sourceCodeLocation.startOffset, node.sourceCodeLocation.endOffset).includes('font-size:0'), `${identity}: invisible text`);
      for (const child of node.childNodes ?? []) walk(child);
    }
    walk(parseFragment(source, { sourceCodeLocationInfo: true }));
    for (const reference of references) assert(ids.has(reference), `${identity}: missing accessible label ${reference}`);
    assert.doesNotMatch(source, /undefined|NaNpx|\[object Object\]|data-motion-key/);
  }
  console.log(`Original illustration models: ${Object.keys(redraws).length} registered + ${Object.keys(supplementalRedraws).length} supplemental; accessible references and deterministic SVG generation passed.`);
}