import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { figureTakeaways } from './illustration-takeaways.mjs';
import { codingQuestionVisuals } from './coding-question-visuals.mjs';

const escape = (value) => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const color = (tone) => `var(--${tone === 'text' ? 'c-text' : tone === 'soft' ? 'c-text-soft' : `viz-${tone}-stroke`})`;
const fill = (tone) => `var(--viz-${tone === 'text' || tone === 'soft' ? 'neutral' : tone}-bg)`;
const number = (value) => new Intl.NumberFormat('en', { maximumSignificantDigits: 4, useGrouping: false }).format(value);
const line = (fromX, fromY, toX, toY, tone = 'neutral', dashed = false) => `<path d="M${fromX} ${fromY}L${toX} ${toY}" style="fill:none;stroke:${color(tone)};stroke-width:1.75${dashed ? ';stroke-dasharray:4 4' : ''}"></path>`;
const circle = (horizontal, vertical, radius, tone = 'focus', solid = false) => `<circle cx="${horizontal}" cy="${vertical}" r="${radius}" style="fill:${solid ? color(tone) : fill(tone)};stroke:${color(tone)};stroke-width:1.75"></circle>`;

function lines(value, width, size = 13) {
  const maximum = Math.max(1, Math.floor(width / (Math.max(14, size) * .61)));
  if (typeof value === 'number') value = number(value);
  const result = [''];
  for (const word of String(value).split(/\s+/)) {
    const pieces = Array.from({ length: Math.ceil(word.length / maximum) }, (_, index) => word.slice(index * maximum, (index + 1) * maximum));
    for (const piece of pieces) {
      if (result.at(-1) && `${result.at(-1)} ${piece}`.length > maximum) result.push(piece);
      else result[result.length - 1] += `${result.at(-1) ? ' ' : ''}${piece}`;
    }
  }
  return result;
}

function text(horizontal, vertical, value, { width = 320, size = 13, anchor = 'start', tone = 'text', weight = 400 } = {}) {
  size = Math.max(14, size);
  if (typeof value === 'number') value = number(value);
  const leading = Math.max(18, Math.ceil(size * 1.25));
  return `<text x="${horizontal}" y="${vertical}" style="font-size:${size}px;font-weight:${weight};text-anchor:${anchor};fill:${color(tone === 'neutral' ? 'soft' : tone)}">${lines(value, width, size).map((value, index) => `<tspan x="${horizontal}" dy="${index ? leading : 0}">${escape(value)}</tspan>`).join('')}</text>`;
}

function bars(model, offset = 0) {
  if (model.groups) {
    let vertical = offset;
    let body = '';
    for (const group of model.groups) {
      body += text(20, vertical + 22, group.title, { weight: 700 });
      const drawing = bars({ ...group, compact: true }, vertical + 36);
      body += drawing.body;
      vertical += drawing.height + 50;
    }
    return { body, height: vertical - offset };
  }
  const { rows, unit = '', limit, domain } = model;
  const values = rows.map((row) => row[1]);
  const minimum = Math.min(0, ...values, ...(domain ?? []));
  const maximum = Math.max(0, ...values, limit ?? 0, ...(domain ?? []));
  assert(maximum > minimum);
  const coordinate = (value) => 28 + (value - minimum) / (maximum - minimum) * 304;
  let vertical = 23 + offset;
  let body = '';
  for (const [index, [name, value, tone = index ? 'output' : 'input']] of rows.entries()) {
    const labelLines = lines(name, 310).length;
    body += text(18, vertical, name, { weight: 700 });
    vertical += labelLines * 18 + 5;
    body += line(28, vertical + 18, 332, vertical + 18)
      + `<rect data-proof-value="${value}" data-proof-min="${minimum}" data-proof-max="${maximum}" data-proof-zero="${coordinate(0)}" x="${Math.min(coordinate(0), coordinate(value))}" y="${vertical}" width="${Math.abs(coordinate(value) - coordinate(0))}" height="35" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.75"></rect>`;
    body += circle(coordinate(value), vertical + 17.5, 4, tone, true);
    if (limit !== undefined && value > limit) {
      body += `<path data-proof-overflow="${value - limit}" d="M${coordinate(limit)} ${vertical + 8}H${coordinate(value)}M${coordinate(limit)} ${vertical + 17.5}H${coordinate(value)}M${coordinate(limit)} ${vertical + 27}H${coordinate(value)}" style="fill:none;stroke:${color('warning')};stroke-width:2"></path>`;
    }
    if (limit !== undefined) body += line(coordinate(limit), vertical - 6, coordinate(limit), vertical + 43, 'warning', true);
    const labelOffset = model.compact ? 50 : 60;
    const valueLabel = `${number(value)}${unit}`;
    const endpoint = coordinate(value);
    const labelWidth = Math.min(290, Math.max(70, valueLabel.length * 9));
    const labelX = Math.max(28 + labelWidth / 2, Math.min(332 - labelWidth / 2, endpoint));
    body += line(endpoint, vertical + 23, endpoint, vertical + labelOffset - 20, tone)
      + text(labelX, vertical + labelOffset, valueLabel, { anchor: 'middle', width: labelWidth, weight: 700, tone });
    if (limit !== undefined) body += text(28, vertical + labelOffset + 25, `${model.limitLabel ?? 'limit'} ${limit}${unit}`, { width: 304, tone: 'warning', size: 12 });
    vertical += (model.compact ? 76 : 94) + (limit !== undefined ? 25 : 0);
  }
  body += text(28, vertical, minimum, { size: 12, tone: 'soft' }) + text(332, vertical, maximum, { anchor: 'end', size: 12, tone: 'soft' });
  return { body, height: vertical + 20 - offset };
}

function parts(model) {
  if (model.composition === 'budget-transfer') {
    const changed = model.rows.map((row, index) => [row[0], row[1] + (index === model.transfer.to ? model.transfer.value : index === model.transfer.from ? -model.transfer.value : 0), row[2]]);
    let body = text(20, 28, '100 points; fixed budget', { size: 20, weight: 700 });
    for (const [phase, rows] of [model.rows, changed].entries()) {
      const vertical = phase ? 218 : 70;
      body += text(20, vertical - 12, phase ? 'After transfer' : 'Before transfer', { weight: 700 });
      let horizontal = 20;
      for (const [, value, tone] of rows) {
        const width = value / 100 * 320;
        body += `<rect data-proof-budget="${phase}" data-proof-value="${value}" x="${horizontal}" y="${vertical}" width="${width}" height="44" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.75"></rect>`
          + text(horizontal + width / 2, vertical + 27, value, { width: width - 4, anchor: 'middle', weight: 700, tone });
        horizontal += width;
      }
    }
    body += `<path data-proof-transfer="6" d="M212 116V151H199V187" style="fill:none;stroke:${color('focus')};stroke-width:2"></path>`
      + line(194, 180, 199, 187, 'focus') + line(204, 180, 199, 187, 'focus')
      + text(20, 155, 'Move 6 points', { width: 170, weight: 700, tone: 'focus' });
    let vertical = 310;
    for (const [index, [name, value, tone]] of model.rows.entries()) {
      const protectedValue = model.transfer.protected.includes(index);
      body += text(20, vertical, name, { width: 190, weight: 700, tone })
        + text(340, vertical, protectedValue ? `${value}; protected` : `${value} to ${changed[index][1]}`, { width: 145, anchor: 'end', tone });
      vertical += 40;
    }
    return { body, height: vertical + 8 };
  }
  const total = model.rows.reduce((sum, row) => sum + row[1], 0);
  assert(total > 0);
  const heading = `${number(total)}${model.unit ?? ''}`;
  const top = 38 + lines(heading, 320, 22).length * 28;
  let horizontal = 20;
  let body = text(20, 30, heading, { size: 22, weight: 700 });
  for (const [index, [name, value, tone = index ? 'warning' : 'output']] of model.rows.entries()) {
    const width = value / total * 320;
    body += `<rect data-proof-part="${escape(name)}" data-proof-value="${value}" data-proof-total="${total}" x="${horizontal}" y="${top}" width="${width}" height="44" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.75"></rect>`;
    const units = Number.isInteger(total) && total <= 32 ? total : 0;
    for (let unit = 1; unit < units; unit++) {
      const position = 20 + unit / total * 320;
      if (position > horizontal && position < horizontal + width) body += line(position, top + 4, position, top + 40, tone);
    }
    horizontal += width;
  }
  let vertical = top + 82;
  let preceding = 0;
  for (const [index, [name, value, tone = index ? 'warning' : 'output']] of model.rows.entries()) {
    body += text(20, vertical, name, { width: 230, weight: 700 })
      + text(340, vertical, `${(100 * value / total).toFixed(model.digits ?? 1)}%`, { anchor: 'end', width: 90, size: 18, weight: 700, tone });
    vertical += lines(name, 230).length * 18 + 10;
    body += `<rect x="20" y="${vertical}" width="320" height="12" style="fill:none;stroke:${color('neutral')};stroke-width:1"></rect><rect x="${20 + preceding / total * 320}" y="${vertical}" width="${value / total * 320}" height="12" style="fill:${color(tone)}"></rect>`;
    vertical += 36;
    body += text(20, vertical, `${number(value)} of ${number(total)}${model.unit ?? ''}`, { width: 320, tone: 'soft' });
    vertical += lines(`${number(value)} of ${number(total)}${model.unit ?? ''}`, 320).length * 18 + 24;
    preceding += value;
  }
  return { body, height: vertical };
}

function population(model) {
  const count = model.rows.reduce((sum, row) => sum + row[1], 0);
  assert(count <= 1200 && count > 0 && Number.isInteger(count));
  const columns = count > 400 ? 40 : count > 80 ? 20 : 10;
  const spacing = 320 / columns;
  const radius = Math.min(7, spacing * .3);
  let body = text(20, 24, `Each mark: ${model.unit ?? 'one observation'}`, { tone: 'soft' });
  let vertical = 38 + lines(`Each mark: ${model.unit ?? 'one observation'}`, 320).length * 18;
  for (const [group, [name, total, tone = group ? 'warning' : 'output']] of model.rows.entries()) {
    body += text(20, vertical + 20, total, { size: 26, width: 78, weight: 700, tone })
      + text(108, vertical + 10, name, { width: 230, weight: 700 });
    vertical += Math.max(34, lines(name, 230).length * 18 + 8) + 10;
    for (let index = 0; index < total; index++) {
      const horizontal = 20 + (index % columns + .5) * spacing;
      const position = vertical + (Math.floor(index / columns) + .5) * spacing;
      body += `<g data-proof-population="${escape(name)}">${group === 0 ? circle(horizontal, position, radius, tone, true) : `<rect x="${horizontal - radius}" y="${position - radius}" width="${radius * 2}" height="${radius * 2}" rx="${radius * .3}" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.2"></rect>`}</g>`;
    }
    vertical += Math.ceil(total / columns) * spacing + 28;
  }
  const finding = `${model.rows[0][1]} / ${count} = ${(100 * model.rows[0][1] / count).toFixed(1)}%`;
  body += line(20, vertical - 7, 340, vertical - 7)
    + text(20, vertical + 27, finding, { size: 24, weight: 700, tone: 'output' });
  return { body, height: vertical + 42 + (lines(finding, 320, 24).length - 1) * 30 };
}

function stateMark(horizontal, vertical, value, tone) {
  const word = String(value).toLowerCase();
  const status = /^(yes|true|pass|on)$/.test(word) ? 'present'
    : /^(no|false|off|absent|none)$/.test(word) ? 'absent'
      : /^(stop|reject|skip|fail)$/.test(word) ? 'stopped'
        : /^(\?|unknown|not yet)$/.test(word) ? 'unknown' : 'value';
  let mark = '';
  if (status === 'present') mark = `<path d="M${horizontal - 7} ${vertical}l5 5 10-12" style="fill:none;stroke:${color(tone)};stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round"></path>`;
  if (status === 'absent') mark = circle(horizontal, vertical, 9, tone) + line(horizontal - 5, vertical, horizontal + 5, vertical, tone);
  if (status === 'stopped') mark = `<path d="M${horizontal - 8} ${vertical - 8}l16 16m0-16l-16 16" style="fill:none;stroke:${color(tone)};stroke-width:2.5"></path>`;
  if (status === 'unknown') mark = `<circle cx="${horizontal}" cy="${vertical}" r="10" style="fill:none;stroke:${color(tone)};stroke-width:1.75;stroke-dasharray:2 3"></circle>`;
  return { status, mark };
}

function matrix(model) {
  if (model.composition === 'budget-rows') {
    let body = text(20, 26, 'Fictional activity budgets: 100% each', { width: 320, weight: 700 });
    let vertical = 80;
    for (const [index, values] of model.values.entries()) {
      body += text(20, vertical, model.labels[index], { width: 320, size: 20, weight: 700 });
      vertical += 18;
      let horizontal = 20;
      for (const [column, value] of values.entries()) {
        const width = value / 100 * 320;
        const tone = ['input', 'output', 'state'][column];
        body += `<rect data-proof-activity="${index},${column}" data-proof-value="${value}" x="${horizontal}" y="${vertical}" width="${width}" height="46" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.75"></rect>`
          + text(horizontal + width / 2, vertical + 28, value, { width: width - 4, anchor: 'middle', weight: 700, tone });
        horizontal += width;
      }
      vertical += 95;
    }
    for (const [index, name] of model.columns.entries()) {
      const tone = ['input', 'output', 'state'][index];
      body += circle(27, vertical - 4, 5, tone, true) + text(44, vertical, name, { width: 290 });
      vertical += 32;
    }
    return { body, height: vertical + 6 };
  }
  if (model.composition === 'denominators') {
    const drawing = matrix({ ...model, composition: 'cells' });
    const [[hits, misses], [alarms]] = model.values;
    let vertical = drawing.height + 24;
    let body = drawing.body;
    for (const [label, denominator, tone] of [['Recall: actual positives', hits + misses, 'input'], ['Precision: predicted positives', hits + alarms, 'output']]) {
      body += text(20, vertical, label, { width: 320, weight: 700 });
      vertical += lines(label, 320).length * 18 + 14;
      body += `<rect x="20" y="${vertical}" width="320" height="30" style="fill:${fill('neutral')};stroke:${color('neutral')}"></rect><rect data-proof-denominator="${denominator}" data-proof-numerator="${hits}" x="20" y="${vertical}" width="${320 * hits / denominator}" height="30" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>`;
      vertical += 61;
      body += text(20, vertical, `${hits} / ${denominator} = ${number(100 * hits / denominator)}%`, { size: 22, weight: 700, tone });
      vertical += 48;
    }
    return { body, height: vertical };
  }
  const rows = model.values;
  const columns = rows[0].length;
  assert(rows.every((row) => row.length === columns));
  const labelWidth = Math.max(42, Math.min(132, Math.max(...rows.map((row, index) => Math.max(...String(model.labels?.[index] ?? index).split(/\s+/).map(word => word.length)) * 14 * .61 + 8))));
  const start = labelWidth + 16;
  const width = (340 - start) / columns;
  const headerLines = Math.max(...Array.from({ length: columns }, (_, column) => lines(model.columns?.[column] ?? column, width - 2, 12).length));
  let body = '';
  for (let column = 0; column < columns; column++) body += text(start + (column + .5) * width, 18, model.columns?.[column] ?? column, { anchor: 'middle', width: width - 2, size: 12, tone: 'soft' });
  let vertical = 18 + headerLines * 18;
  for (const [row, values] of rows.entries()) {
    const height = Math.max(64, lines(model.labels?.[row] ?? row, labelWidth, 12).length * 18 + 20, ...values.map(value => lines(value, width - 14, 14).length * 18 + 38));
    body += text(start - 12, vertical + 18, model.labels?.[row] ?? row, { anchor: 'end', width: labelWidth, size: 12, tone: 'soft' });
    for (const [column, value] of values.entries()) {
      const tone = model.blocked?.some(([blockedRow, blockedColumn]) => blockedRow === row && blockedColumn === column) ? 'warning' : model.selected?.some(([selectedRow, selectedColumn]) => selectedRow === row && selectedColumn === column) ? 'focus' : 'neutral';
      const horizontal = start + column * width;
      const center = horizontal + (width - 5) / 2;
      const glyph = stateMark(center, vertical + 18, value, tone === 'neutral' ? 'text' : tone);
      const numeric = typeof value === 'number';
      const size = numeric && String(value).length <= 5 && width > 64 ? 22 : 14;
      const baseline = glyph.status === 'value' ? vertical + (height - lines(value, width - 14, size).length * Math.ceil(size * 1.25)) / 2 + size - 2 : vertical + 46;
      body += `<g data-proof-status="${glyph.status}"><rect data-proof-cell="${row},${column}" x="${horizontal}" y="${vertical}" width="${width - 5}" height="${height - 5}" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:${tone === 'focus' ? 2 : 1}"></rect>${glyph.mark}`
        + text(center, baseline, value, { anchor: 'middle', width: width - 14, size, tone: tone === 'warning' ? 'warning' : 'text', weight: tone === 'focus' || numeric ? 700 : 400 });
      if (tone === 'warning') body += `<path data-proof-warning="${row},${column}" d="M${horizontal + 4} ${vertical + 8}v${height - 21}" style="stroke:${color(tone)};stroke-width:3;stroke-dasharray:3 3"></path>`;
      if (tone === 'focus') body += line(horizontal + 5, vertical + height - 11, horizontal + width - 10, vertical + height - 11, 'focus');
      body += '</g>';
    }
    vertical += height;
  }
  return { body, height: vertical + 12 };
}

function curve(model) {
  const { domain, range, series } = model;
  const left = Math.max(48, Math.ceil(Math.max(...range.map(value => number(value).length)) * 14 * .61 + 12));
  const horizontal = (value) => left + (value - domain[0]) / (domain[1] - domain[0]) * (330 - left);
  const vertical = (value) => 220 - (value - range[0]) / (range[1] - range[0]) * 170;
  let body = '';
  for (let division = 1; division < 4; division++) body += `<g opacity=".35">${line(left, 220 - division * 42.5, 330, 220 - division * 42.5, 'neutral', true)}</g>`;
  body += line(left, 40, left, 220) + line(left, 220, 334, 220)
    + text(left, 245, domain[0], { size: 12, tone: 'soft' }) + text(330, 245, domain[1], { size: 12, anchor: 'end', tone: 'soft' })
    + text(left - 8, 54, range[1], { size: 12, anchor: 'end', width: left - 12, tone: 'soft' }) + text(left - 8, 220, range[0], { size: 12, anchor: 'end', width: left - 12, tone: 'soft' });
  const checkpoint = model.landmark ?? (domain[0] + domain[1]) / 2;
  body += `<path data-proof-landmark="${checkpoint}" d="M${horizontal(checkpoint)} 40V220" style="stroke:${color('focus')};stroke-width:1.5;stroke-dasharray:2 4"></path>`;
  if (horizontal(checkpoint) - left > 46 && 330 - horizontal(checkpoint) > 46) body += text(horizontal(checkpoint), 245, number(checkpoint), { anchor: 'middle', size: 12, width: 60, tone: 'focus' });
  if (model.threshold !== undefined) body += line(horizontal(model.threshold), 40, horizontal(model.threshold), 220, 'warning', true);
  const observations = [];
  for (const [index, [name, points, tone = index ? 'output' : 'input']] of series.entries()) {
    assert(points.every(([horizontalValue, verticalValue]) => Number.isFinite(horizontalValue) && Number.isFinite(verticalValue)));
    body += `<path data-proof-series="${escape(name)}" d="${points.map(([horizontalValue, verticalValue], point) => `${point ? 'L' : 'M'}${horizontal(horizontalValue)} ${vertical(verticalValue)}`).join('')}" style="fill:none;stroke:${color(tone)};stroke-width:3${index ? ';stroke-dasharray:7 4' : ''}"></path>`;
    const before = points.findLast(point => point[0] <= checkpoint);
    const after = points.find(point => point[0] >= checkpoint);
    const value = before && after ? before[0] === after[0] ? before[1] : before[1] + (after[1] - before[1]) * (checkpoint - before[0]) / (after[0] - before[0]) : null;
    observations.push(value);
    if (value !== null) body += `<g data-proof-observation="${index}" data-proof-x="${checkpoint}" data-proof-y="${value}">${circle(horizontal(checkpoint), vertical(value), 5, tone, index === 0)}</g>`;
  }
  body += text(190, 273, model.xLabel, { anchor: 'middle', size: 12, width: 280, tone: 'soft' })
    + text(20, 20, model.yLabel, { size: 12, tone: 'soft' });
  let top = 310;
  for (const [index, [name, , tone = index ? 'output' : 'input']] of series.entries()) {
    body += line(20, top - 4, 48, top - 4, tone, Boolean(index)) + text(60, top, name, { width: 280 });
    top += lines(name, 280).length * 18 + 5;
    if (observations[index] !== null) {
      body += text(60, top, `${number(observations[index])} at ${number(checkpoint)}`, { size: 18, width: 280, weight: 700, tone });
      top += lines(`${number(observations[index])} at ${number(checkpoint)}`, 280, 18).length * 23;
    }
    top += 16;
  }
  if (model.threshold !== undefined) {
    body += line(20, top - 4, 48, top - 4, 'warning', true) + text(60, top, `Decision threshold = ${model.threshold}`, { width: 280, tone: 'warning' });
    top += lines(`Decision threshold = ${model.threshold}`, 280).length * 18 + 12;
  }
  return { body, height: top + 5 };
}

function timeline(model) {
  const { events, domain, unit = 'event order, not elapsed time' } = model;
  const start = domain?.[0] ?? Math.min(...events.map((event) => event[1]));
  const end = domain?.[1] ?? Math.max(...events.map((event) => event[1]));
  assert(end > start);
  const coordinate = (value) => 30 + (value - start) / (end - start) * 300;
  let body = '';
  let vertical = 25;
  for (const [index, [name, value, tone = index === events.length - 1 ? 'output' : 'input']] of events.entries()) {
    body += text(20, vertical, name, { width: 320, weight: 700 });
    vertical += lines(name, 320).length * 18 + 12;
    body += line(30, vertical + 8, 330, vertical + 8, 'neutral', true)
      + circle(coordinate(value), vertical + 8, 7, tone, index === events.length - 1)
      + text(Math.max(53, Math.min(307, coordinate(value))), vertical + 39, Number(value.toFixed(3)), { anchor: 'middle', width: 90, size: 18, weight: 700, tone });
    if (model.boundary !== undefined) body += line(coordinate(model.boundary), vertical - 5, coordinate(model.boundary), vertical + 24, 'warning', true);
    vertical += 67;
  }
  if (model.boundary !== undefined) body = `<path data-proof-boundary="${model.boundary}" d="M${coordinate(model.boundary)} 8V${vertical - 25}" style="fill:none;stroke:${color('warning')};stroke-width:1.25;stroke-dasharray:2 5;opacity:.5"></path>${body}`;
  body += line(30, vertical - 9, 330, vertical - 9) + line(323, vertical - 14, 330, vertical - 9) + line(323, vertical - 4, 330, vertical - 9)
    + text(180, vertical + 17, unit, { anchor: 'middle', width: 320, size: 12, tone: 'soft' });
  return { body, height: vertical + lines(unit, 320, 12).length * 18 + 28 };
}

function connections(model) {
  const { nodes, edges } = model;
  const levels = Math.max(...nodes.map((node) => node[2])) + 1;
  const spacing = Math.max(114, ...nodes.map(node => lines(node[0], 155).length * 18 + 80));
  const positions = nodes.map((node) => ({ horizontal: node[1] === 0 ? 90 : node[1] === 1 ? 270 : 180, vertical: 38 + node[2] * spacing }));
  let body = '';
  for (const [edgeIndex, [from, to, blocked = false]] of edges.entries()) {
    const start = positions[from];
    const end = positions[to];
    const side = end.horizontal < 180 ? -1 : 1;
    const route = side < 0 ? 4 + edgeIndex % 3 * 3 : 356 - edgeIndex % 3 * 3;
    const fromX = start.horizontal + side * 17;
    const direction = end.vertical < start.vertical ? -1 : 1;
    const toY = end.vertical - direction * 18;
    const approach = end.vertical - (direction > 0 ? 28 : -22);
    const tone = blocked ? 'warning' : 'focus';
    const path = start.horizontal === end.horizontal
      ? `M${start.horizontal} ${start.vertical + direction * 18}V${toY}`
      : start.vertical === end.vertical
      ? `M${start.horizontal} ${start.vertical - 17}V${start.vertical - 28}H${end.horizontal}V${toY}`
      : `M${fromX} ${start.vertical}H${route}V${approach}H${end.horizontal}V${toY}`;
    body += `<path data-proof-edge="${from},${to}" d="${path}" style="fill:none;stroke:${color(tone)};stroke-width:1.75${blocked ? ';stroke-dasharray:4 4' : ''}"></path>`
      + line(end.horizontal - 4, toY - direction * 7, end.horizontal, toY, tone)
      + line(end.horizontal + 4, toY - direction * 7, end.horizontal, toY, tone);
    if (blocked) {
      const middleX = start.vertical === end.vertical ? (start.horizontal + end.horizontal) / 2 : route;
      const middleY = start.vertical === end.vertical ? start.vertical - 28 : (start.vertical + approach) / 2;
      body += line(middleX - 6, middleY - 6, middleX + 6, middleY + 6, 'warning') + line(middleX + 6, middleY - 6, middleX - 6, middleY + 6, 'warning');
    }
  }
  for (const [index, [name, , , tone = 'input']] of nodes.entries()) {
    const position = positions[index];
    const shape = tone === 'state' ? `<path d="M${position.horizontal} ${position.vertical - 17}l17 17-17 17-17-17Z" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></path>` : circle(position.horizontal, position.vertical, 16, tone);
    const labelHeight = lines(name, 155).length * 18;
    body += `<g data-proof-node="${index}">${shape}${tone === 'output' ? circle(position.horizontal, position.vertical, 12, tone) : ''}`
      + text(position.horizontal, position.vertical + 4, index + 1, { size: 12, anchor: 'middle', width: 30, weight: 700 })
      + `<rect x="${position.horizontal - 79}" y="${position.vertical + 24}" width="158" height="${labelHeight + 8}" style="fill:var(--c-bg)"></rect>`
      + text(position.horizontal, position.vertical + 39, name, { anchor: 'middle', width: 155, weight: 700 });
    body += '</g>';
  }
  return { body, height: levels * spacing + 6 };
}

function ledger(model) {
  if (model.composition === 'physical-map') {
    const pages = model.rows[1][1].map(Number);
    const locations = [pages[1], pages[2], pages[0]];
    let body = text(20, 25, 'Logical sequence', { weight: 700 });
    for (const [index, page] of pages.entries()) {
      const horizontal = 30 + index * 110;
      const target = 55 + locations.indexOf(page) * 110;
      const tone = ['input', 'state', 'output'][index];
      body += `<rect x="${horizontal}" y="48" width="80" height="52" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>`
        + text(horizontal + 40, 80, index, { anchor: 'middle', size: 22, width: 60, weight: 700, tone })
        + `<path data-proof-page-map="${index},${page}" d="M${horizontal + 40} 100V${126 + index * 22}H${target}V238" style="fill:none;stroke:${color(tone)};stroke-width:2${index === 1 ? ';stroke-dasharray:6 3' : ''}"></path>`
        + line(target - 5, 231, target, 238, tone) + line(target + 5, 231, target, 238, tone);
    }
    body += text(20, 219, 'Physical pages: different order', { weight: 700, width: 320 });
    for (const [index, page] of locations.entries()) {
      const horizontal = 20 + index * 110;
      const tone = ['input', 'state', 'output'][pages.indexOf(page)];
      body += `<rect x="${horizontal}" y="245" width="90" height="70" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>`
        + text(horizontal + 45, 271, `page ${page}`, { anchor: 'middle', width: 80, weight: 700 })
        + text(horizontal + 45, 299, `block ${pages.indexOf(page)}`, { anchor: 'middle', width: 80, tone });
    }
    return { body, height: 335 };
  }
  if (model.composition === 'nested') {
    const candidates = model.rows[1][1];
    const height = 94 + candidates.length * 76;
    let body = text(20, 25, 'Denominator: the whole set', { weight: 700 })
      + `<path data-proof-candidate-enclosure d="M34 48H326V${height - 16}H34Z" style="fill:${fill('input')};stroke:${color('input')};stroke-width:2"></path>`;
    for (const [index, candidate] of candidates.entries()) {
      const vertical = 75 + index * 76;
      const positive = model.rows[0][1].includes(candidate);
      if (positive) body += `<path data-proof-nested-positive d="M48 ${vertical - 10}H312V${vertical + 49}H48Z" style="fill:${fill('output')};stroke:${color('output')};stroke-width:2"></path>`;
      body += circle(72, vertical + 18, 8, positive ? 'output' : 'input', positive)
        + text(94, vertical + 22, candidate, { width: 200, weight: 700 });
      if (positive) body += text(94, vertical + 42, 'also the numerator', { width: 200, tone: 'output' });
    }
    return { body, height };
  }
  let body = '';
  let vertical = 28;
  for (const [index, [name, tokens, tone = index ? 'output' : 'input']] of model.rows.entries()) {
    const rowStart = vertical;
    const inset = model.composition === 'transform' ? 44 : 20;
    if (model.composition === 'transform') body += circle(23, vertical - 4, 10, tone) + text(23, vertical, index + 1, { anchor: 'middle', width: 24, weight: 700 });
    body += text(inset, vertical, name, { width: 340 - inset, weight: 700 });
    vertical += lines(name, 340 - inset).length * 18 + 14;
    const columns = Math.min(tokens.length || 1, model.composition === 'arithmetic' ? 5 : 8);
    const width = (340 - inset) / columns;
    const labelWidth = width - (model.labelPadding ?? 12);
    for (let start = 0; start < tokens.length; start += columns) {
      const row = tokens.slice(start, start + columns);
      const height = Math.max(48, ...row.map(value => lines(value, labelWidth, 14).length * 18 + 20));
      for (const [token, value] of row.entries()) {
        const horizontal = inset + token * width;
        const center = horizontal + (width - 6) / 2;
        const operator = /^(\+|-|x|=|:|\/|->)$/.test(String(value));
        const final = index === model.rows.length - 1 && (model.composition === 'arithmetic' ? token === tokens.length - 1 : tokens.length === 1 || model.composition === 'transform');
        const plain = (operator && !(model.emptySlots && value === '-')) || model.composition === 'arithmetic';
        const empty = /^(out|none|free|spare|\?|-)$/i.test(String(value));
        const size = final && String(value).length < 7 && width > 70 ? 20 : 14;
        body += `<g data-proof-token="${escape(value)}" data-proof-row="${index}" data-proof-position="${start + token}" data-proof-box="${horizontal},${vertical},${width - 6},${height}">`;
        if (!plain) body += `<rect x="${horizontal}" y="${vertical}" width="${width - 6}" height="${height}" rx="${model.composition === 'set' ? Math.min(18, (width - 6) / 2) : 0}" style="fill:${empty ? 'none' : fill(tone)};stroke:${color(tone)};stroke-width:${final ? 2 : 1.5}${empty ? ';stroke-dasharray:3 4' : ''}"></rect>`;
        body += text(center, vertical + (height - lines(value, labelWidth, size).length * Math.ceil(size * 1.25)) / 2 + size - 2, value, { anchor: 'middle', width: labelWidth, size, weight: final || operator ? 700 : 400, tone: empty ? 'soft' : 'text' });
        if (plain && final) body += line(horizontal + 4, vertical + height - 3, horizontal + width - 10, vertical + height - 3, tone);
        body += '</g>';
        if (model.composition === 'sequence' && token < row.length - 1) body += line(horizontal + width - 3, vertical + height / 2, horizontal + width + 1, vertical + height / 2, tone);
      }
      vertical += height + 8;
    }
    if (model.composition === 'transform' && index < model.rows.length - 1) body += line(23, rowStart + 13, 23, vertical + 2, 'neutral', true) + line(19, vertical - 4, 23, vertical + 2) + line(27, vertical - 4, 23, vertical + 2);
    vertical += 28;
  }
  return { body, height: vertical };
}

function areas(model) {
  const scale = Math.min(120, 320 / Math.max(...model.regions.map(region => region.width)), 180 / Math.max(...model.regions.map(region => region.height)));
  let vertical = 25;
  let body = '';
  for (const [index, region] of model.regions.entries()) {
    const tone = index ? 'output' : 'input';
    const volume = region.width * region.height;
    const marks = region.mass / model.unitMass;
    assert(Number.isInteger(marks) && marks > 0);
    const columns = Math.ceil(Math.sqrt(marks));
    const rows = Math.ceil(marks / columns);
    body += text(20, vertical, region.label, { weight: 700 });
    vertical += lines(region.label, 320).length * 18 + 14;
    body += `<g data-proof-region="${index}" data-proof-mass="${region.mass}"><rect data-proof-area="${volume}" data-proof-scale="${scale}" x="20" y="${vertical}" width="${region.width * scale}" height="${region.height * scale}" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.75"></rect>`;
    for (let column = 1; column < columns; column++) body += line(20 + column / columns * region.width * scale, vertical, 20 + column / columns * region.width * scale, vertical + region.height * scale, tone, true);
    for (let row = 1; row < rows; row++) body += line(20, vertical + row / rows * region.height * scale, 20 + region.width * scale, vertical + row / rows * region.height * scale, tone, true);
    for (let mark = 0; mark < marks; mark++) body += circle(20 + (mark % columns + .5) / columns * region.width * scale, vertical + (Math.floor(mark / columns) + .5) / rows * region.height * scale, 7, tone, true);
    body += '</g>';
    vertical += region.height * scale + 26;
    body += text(20, vertical, `Volume ${number(volume)}; density ${number(region.mass / volume)}`, { weight: 700, tone });
    vertical += 54;
  }
  body += text(20, vertical, `Each dot = ${number(model.unitMass)} probability mass`, { tone: 'soft' });
  return { body, height: vertical + 28 };
}

function geometry(model) {
  const [minimum, maximum] = model.domain ?? [-2, 4];
  const coordinate = (value) => 42 + (value - minimum) / (maximum - minimum) * 276;
  const vertical = (value) => 312 - (value - minimum) / (maximum - minimum) * 276;
  const originX = coordinate(0);
  const originY = vertical(0);
  let body = line(32, originY, 330, originY) + line(originX, 26, originX, 322)
    + text(328, originY - 11, 'x', { anchor: 'end', width: 15, size: 12, tone: 'soft' })
    + text(originX + 10, 25, 'y', { width: 15, size: 12, tone: 'soft' });
  for (let tick = Math.ceil(minimum); tick <= Math.floor(maximum); tick++) {
    if (tick === 0) continue;
    body = `<g opacity=".3">${line(coordinate(tick), 36, coordinate(tick), 312, 'neutral', true)}${line(42, vertical(tick), 318, vertical(tick), 'neutral', true)}</g>` + body;
    body += line(coordinate(tick), originY - 3, coordinate(tick), originY + 3)
      + text(coordinate(tick), originY + 20, tick, { anchor: 'middle', width: 32, tone: 'soft' });
  }
  if (model.polygon) body += `<polygon data-proof-polygon points="${model.polygon.map(([horizontalValue, verticalValue]) => `${coordinate(horizontalValue)},${vertical(verticalValue)}`).join(' ')}" style="fill:${fill('state')};stroke:${color('state')};stroke-width:1.75"></polygon>`;
  for (const [center, radius, tone = 'output'] of model.circles ?? []) body += `<circle cx="${coordinate(center[0])}" cy="${vertical(center[1])}" r="${radius / (maximum - minimum) * 276}" style="fill:none;stroke:${color(tone)};stroke-width:1.75;stroke-dasharray:5 3"></circle>`;
  for (const [index, [name, [horizontalValue, verticalValue], tone = index ? 'output' : 'input']] of model.vectors.entries()) {
    const endX = coordinate(horizontalValue);
    const endY = vertical(verticalValue);
    const angle = Math.atan2(endY - originY, endX - originX);
    if (model.composition !== 'points') body += `<path d="M${originX} ${originY}L${endX} ${endY}" style="fill:none;stroke:${color(tone)};stroke-width:${index ? 2.5 : 3.5}${index ? ';stroke-dasharray:7 3' : ''}"></path>`
      + line(endX - 9 * Math.cos(angle - .45), endY - 9 * Math.sin(angle - .45), endX, endY, tone)
      + line(endX - 9 * Math.cos(angle + .45), endY - 9 * Math.sin(angle + .45), endX, endY, tone);
    body += `<g data-proof-vector="${index}" data-proof-x="${horizontalValue}" data-proof-y="${verticalValue}">${index % 2 ? `<rect x="${endX - 4}" y="${endY - 4}" width="8" height="8" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>` : circle(endX, endY, 4, tone, true)}</g>`;
  }
  for (const [[fromX, fromY], [toX, toY]] of model.guides ?? []) body += line(coordinate(fromX), vertical(fromY), coordinate(toX), vertical(toY), 'warning', true);
  if (model.composition === 'projection') {
    const [[, point], [, projected]] = model.vectors;
    const direction = point[1] > projected[1] ? -1 : 1;
    body += `<path data-proof-right-angle d="M${coordinate(projected[0]) - 11} ${vertical(projected[1])}v${direction * 11}h11" style="fill:none;stroke:${color('warning')};stroke-width:1.75"></path>`;
  }
  let top = 353;
  for (const [index, [name, values, tone = index ? 'output' : 'input']] of model.vectors.entries()) {
    const label = `${name}: (${values.map(number).join(', ')})`;
    body += line(20, top - 4, 45, top - 4, tone, Boolean(index)) + (index % 2 ? `<rect x="28" y="${top - 8}" width="8" height="8" style="fill:${fill(tone)};stroke:${color(tone)}"></rect>` : circle(32, top - 4, 4, tone, true)) + text(57, top, label, { width: 285 });
    top += lines(label, 285).length * 18 + 14;
  }
  return { body, height: top + 8 };
}

function intervals(model) {
  const horizontal = (value) => 38 + value * 280;
  let body = text(20, 25, 'New sample, new interval', { weight: 700, size: 18 })
    + text(20, 53, 'One fixed probability = 0.5', { tone: 'soft' })
    + `<path data-proof-fixed-parameter="0.5" d="M${horizontal(.5)} 74V284" style="stroke:${color('focus')};stroke-width:2;stroke-dasharray:3 4"></path>`;
  for (const [index, [lower, upper]] of model.intervals.entries()) {
    const vertical = 86 + index * 19;
    const covers = lower <= .5 && upper >= .5;
    const tone = covers ? 'input' : 'warning';
    body += `<g data-proof-interval="${lower},${upper}" data-proof-covers="${covers}">${line(horizontal(lower), vertical, horizontal(upper), vertical, tone)}${line(horizontal(lower), vertical - 5, horizontal(lower), vertical + 5, tone)}${line(horizontal(upper), vertical - 5, horizontal(upper), vertical + 5, tone)}</g>`;
  }
  body += text(20, 313, 'Same truth; changing data', { width: 320, tone: 'soft' })
    + line(20, 336, 340, 336)
    + text(20, 371, 'One sample, one posterior', { weight: 700, size: 18 })
    + text(20, 402, '3 successes, 2 failures; uniform prior', { width: 320, tone: 'soft' });
  const points = Array.from({ length: 81 }, (_, index) => {
    const value = index / 80;
    return [horizontal(value), 560 - 60 * value ** 3 * (1 - value) ** 2 * 60];
  });
  body += `<path data-proof-posterior="Beta(4,3)" d="M38 560${points.map(([horizontalValue, verticalValue]) => `L${horizontalValue} ${verticalValue}`).join('')}L318 560Z" style="fill:${fill('output')};stroke:${color('output')};stroke-width:2.5"></path>`
    + line(38, 560, 318, 560)
    + text(38, 586, '0', { tone: 'soft' }) + text(318, 586, '1', { anchor: 'end', tone: 'soft' })
    + text(178, 614, 'Candidate probability', { anchor: 'middle', width: 280, tone: 'soft' });
  return { body, height: 637 };
}

function suppression(model) {
  let body = text(20, 25, 'Score order: A > B > C', { size: 18, weight: 700 });
  for (const [index, box] of model.boxes.entries()) {
    const tone = ['input', 'warning', 'output'][index];
    body += `<rect data-proof-box-identity="${box.name}" x="${box.left}" y="70" width="${box.width}" height="104" style="fill:none;stroke:${color(tone)};stroke-width:2.5${index === 1 ? ';stroke-dasharray:5 4' : ''}"></rect>`
      + text(box.left + 18, 59, box.name, { width: 30, size: 22, weight: 700, tone });
  }
  body += text(20, 208, 'A overlaps B. B overlaps C.', { width: 320, weight: 700 })
    + text(20, 235, 'A and C do not overlap.', { width: 320, tone: 'soft' });
  for (const [index, [label, result, tone]] of [['Keep A', 'A removes B', 'input'], ['Skip removed B', 'B cannot remove C', 'warning'], ['Keep C', 'A and C survive', 'output']].entries()) {
    const vertical = 278 + index * 86;
    body += circle(31, vertical - 4, 10, tone) + text(31, vertical, index + 1, { anchor: 'middle', width: 25, weight: 700 })
      + text(54, vertical, label, { width: 270, weight: 700 })
      + text(54, vertical + 27, result, { width: 270, tone });
    if (index < 2) body += line(31, vertical + 10, 31, vertical + 68, 'neutral', true);
  }
  return { body, height: 510 };
}

function topology(model) {
  let body = '';
  for (const [from, to] of model.edges) {
    const start = model.nodes[from];
    const end = model.nodes[to];
    body += `<path data-proof-topology-edge="${from},${to}" d="M${start[1]} ${start[2]}L${end[1]} ${end[2]}" style="stroke:${color('input')};stroke-width:2.5"></path>`;
  }
  for (const [index, [name, horizontal, vertical]] of model.nodes.entries()) body += `<g data-proof-topology-node="${index}">${circle(horizontal, vertical, 19, index === 3 ? 'warning' : 'input')}${text(horizontal, vertical + 5, name, { anchor: 'middle', width: 30, weight: 700 })}</g>`;
  body += text(20, 259, '4 nodes; 3 edges = n - 1', { width: 320, size: 20, weight: 700 })
    + text(20, 298, 'A cycle and an isolated node', { width: 320, tone: 'warning' });
  return { body, height: 326 };
}

const renderers = { bars, parts, population, matrix, curve, timeline, connections, ledger, areas, geometry, intervals, suppression, topology };
const sampledOperations = new WeakMap();
const sample = (operation, minimum, maximum, count = 65) => {
  const points = Array.from({ length: count }, (_, index) => {
    const value = minimum + (maximum - minimum) * index / (count - 1);
    return [value, operation(value)];
  });
  sampledOperations.set(points, operation);
  return points;
};
const gaussian = (value, mean = 0, deviation = 1) => Math.exp(-.5 * ((value - mean) / deviation) ** 2) / (deviation * Math.sqrt(2 * Math.PI));
const divergence = (first, second) => first.reduce((sum, value, index) => sum + value * Math.log(value / second[index]), 0);
export const visualArguments = {};
function define(identity, kind, premise, conclusion, data) {
  assert(!visualArguments[identity], `Duplicate argument ${identity}`);
  assert(figureTakeaways[identity], `Unknown figure ${identity}`);
  visualArguments[identity] = { kind, premise, conclusion, ...data };
}

function coding(slug, kind, premise, conclusion, data) {
  assert(codingQuestionVisuals[slug], `Unknown coding definition ${slug}`);
  define(`${slug}-state`, kind, `Worked example: ${premise}`, conclusion, { ...data, coding: true });
}

define('bayes-positive-test-denominator', 'population', 'A positive result is a new population', 'Among all 1,098 positive tests, only 99 are true positives. Sensitivity is not the answer to this question.', { rows: [['true positives', 99, 'output'], ['false positives', 999, 'warning']], unit: 'one positive test' });
define('ab-test-mde-sample-size', 'bars', 'Double the noise, quadruple the sample', 'With effect size and error targets fixed, sample size scales with variance, not standard deviation.', { rows: [['Noise 1; relative sample size', 1], ['Noise 2; relative sample size', 4]], unit: 'x' });
define('adamw-decoupled-shrinkage', 'bars', 'Identical weights, different gradient histories', 'In this illustrative frozen-preconditioner step, AdamW applies the same direct decay to both weights.', { rows: [['Coupled L2: scale 1, shrinkage', .001, 'input'], ['Coupled L2: scale 10, shrinkage', .0001, 'warning'], ['AdamW: either scale, shrinkage', .001, 'output']] });
define('attention-keys-choose-values-contribute', 'bars', 'Attention weight is not the final contribution', 'A smaller attention weight can carry a larger value. Compare the products, not the weights alone.', { rows: [['Weight .8 x value 1', .8, 'input'], ['Weight .2 x value 10', 2, 'output']] });
define('class-imbalance-sampling-prior', 'population', 'A balanced training sample does not rebalance the world', 'This illustrative deployment population has ten positives for every 990 negatives, regardless of balanced training batches.', { rows: [['deployment positives', 10, 'output'], ['deployment negatives', 990, 'neutral']], unit: 'one deployment case' });
define('confusion-matrix-denominators', 'matrix', 'Eight hits; two different questions', 'Recall divides eight by ten actual positives. Precision divides the same eight by forty predicted positives.', { values: [[8, 2], [32, 58]], columns: ['pred +', 'pred -'], labels: ['actual +', 'actual -'], selected: [[0, 0]], blocked: [[0, 1], [1, 0]] });
define('calibration-reliability-gap', 'population', 'An 80% forecast bin needs 80 positives per 100', 'Here only 60 of 100 cases are positive. The 20-point gap is calibration error, not classification error.', { rows: [['observed positives', 60, 'output'], ['observed negatives', 40, 'neutral']], unit: 'one case forecast at 80%' });
define('contextual-bandit-partial-feedback-support', 'ledger', 'Action C has no logged support', 'An unobserved reward cannot be filled with zero. No importance weight can recover an action whose logging probability is zero.', { rows: [['Logging probabilities', ['A: .6', 'B: .4', 'C: 0'], 'input'], ['Reward information', ['A: seen', 'B: seen', 'C: ?'], 'warning']] });
define('continuous-batching-reuses-finished-slot', 'ledger', 'A finished request frees a slot, not the whole batch', 'The next iteration keeps A and replaces B with C. It does not wait for A to finish.', { rows: [['Iteration t', ['A', 'B'], 'input'], ['B finishes', ['A', 'free'], 'warning'], ['Iteration t + 1', ['A', 'C'], 'output']] });
define('point-in-time-availability-boundary', 'timeline', 'An early event can still be unavailable at prediction', 'The 11:50 event arrives at 12:10. A noon prediction cannot use it, even though its event timestamp is earlier.', { events: [['Event occurs', 11 + 50 / 60], ['Prediction boundary', 12, 'warning'], ['Event becomes available', 12 + 10 / 60, 'output']], boundary: 12, unit: 'hour on the same clock' });
define('decision-threshold-action-regions', 'ledger', 'The same score can lead to different permitted actions', 'Scores do not carry a universal action. These illustrative policy regions require costs, thresholds, and review capacity.', { rows: [['Illustrative score regions', ['0-.2', '.2-.8', '.8-1'], 'input'], ['Declared actions', ['allow', 'review', 'block'], 'output']] });
define('fixed-capacity-portfolio-guardrails', 'parts', 'A fixed budget makes the tradeoff visible', 'Training grows from 50 to 56 by taking six points from post-training, which falls from 20 to 14. Protected serving and recovery stay at 20 and 10; the total remains 100.', { rows: [['training', 50, 'input'], ['post-training', 20, 'focus'], ['serving', 20, 'output'], ['recovery', 10, 'neutral']], unit: ' budget points', transfer: { from: 1, to: 0, value: 6, protected: [2, 3] } });
define('fixed-budget-ranking-funnel', 'bars', 'Spend expensive scoring only after narrowing', 'A true count scale makes the 10,000-to-100 reduction visible. Stage widths are not latency measurements.', { rows: [['Retrieved candidates', 10000, 'input'], ['Pre-ranked candidates', 1000, 'state'], ['Final scoring candidates', 100, 'output']], unit: ' candidates' });
define('agent-safety-authority-loop', 'bars', 'The next proposal crosses a limit already partly spent', '60 committed plus 50 proposed is 110. The ten-unit overflow blocks new authority; it does not undo committed effects.', { rows: [['Already committed', 60, 'input'], ['Committed plus proposed', 110, 'warning']], limit: 100, unit: ' impact units' });
define('disagreement-evidence-commit-loop', 'bars', 'The memory limit rejects an option, not its advocate', 'On one shared scale, faster cache A breaches 8 GB. Cache B fits; reopening A needs new evidence of feasibility.', { rows: [['A: faster cache', 9, 'warning'], ['B: smaller cache', 6, 'output']], limit: 8, unit: ' GB' });
define('multimodal-stale-frame-gate', 'bars', 'Recent arrival can hide stale content', 'Capture age is 4.30 seconds; arrival age is 0.10. Arrival does not refresh the frame.', { rows: [['Question minus capture', 4.3, 'warning'], ['Question minus arrival', .1, 'input']], unit: ' s' });
define('reasoning-admission-and-capacity', 'ledger', 'A compute surplus cannot pay a KV-page deficit', 'Both resource conditions must hold. Six required pages do not fit into five available pages, despite spare compute.', { rows: [['Compute: 4 required, 6 available', ['yes', 'yes', 'yes', 'yes', 'free', 'free'], 'output'], ['KV pages: 6 required, 5 available', ['yes', 'yes', 'yes', 'yes', 'yes', 'NO'], 'warning']] });
define('ecosystem-serving-stages', 'population', 'Eligibility is not exposure', 'Only five displayed candidates create exposure records. Fifteen other eligible candidates are unshown, not negative outcomes.', { rows: [['displayed', 5, 'output'], ['eligible but unshown', 15, 'input'], ['ineligible', 80, 'warning']], unit: 'one retrieved candidate' });
define('video-ecosystem-delayed-effects', 'timeline', 'The fast readout arrives before the supply response', 'This is event order, not a duration estimate. A viewer experiment can end before creator supply has had time to respond.', { events: [['Ranking policy changes', 0], ['Immediate viewer response', 1], ['Short experiment ends', 2, 'warning'], ['Creator response becomes observable', 3, 'output']] });

define('activation-gradient-regions', 'curve', 'A flat forward response has little backward slope', 'ReLU has zero derivative on its negative side. Sigmoid slopes become small in both tails; their forward outputs are not gradients.', { domain: [-5, 5], range: [0, 1], xLabel: 'input', yLabel: 'local derivative', series: [['Sigmoid derivative', sample(value => { const probability = 1 / (1 + Math.exp(-value)); return probability * (1 - probability); }, -5, 5)], ['ReLU derivative (except at zero)', [[-5, 0], [-.001, 0], [.001, 1], [5, 1]], 'output']] });
define('optimizer-sharpness-reparameterization', 'curve', 'A stretched coordinate makes the same basin look sharper', 'These two parameter coordinates represent the same scalar prediction error. Curvature alone is not invariant to reparameterization.', { domain: [-2, 2], range: [0, 4], xLabel: 'parameter coordinate', yLabel: 'loss', series: [['L(a) = a squared', sample(value => value ** 2, -2, 2)], ['L(b) = (2b) squared; a = 2b', sample(value => (2 * value) ** 2, -1, 1), 'warning']] });
define('adversarial-threat-set-boundary', 'geometry', 'A small allowed move can cross a decision boundary', 'The dashed segment reaches across the boundary while staying inside this illustrative horizontal perturbation allowance.', { domain: [-2, 3], vectors: [['Original example', [.3, 1], 'input'], ['Perturbed example', [-.2, 1], 'warning']], guides: [[[.3, 1], [-.2, 1]], [[0, -2], [0, 3]]] });
define('als-alternating-independent-solves', 'matrix', 'Freeze one factor; solve independent rows of the other', 'The highlighted row can change while the other factor stays fixed. Alternating does not mean updating both sides simultaneously.', { values: [['u1', '?', '?'], ['u2', '?', '?'], ['V', 'fixed', 'fixed']], columns: ['row', 'factor 1', 'factor 2'], labels: ['solve', 'solve', 'hold'], selected: [[0, 1], [0, 2], [1, 1], [1, 2]] });
define('backprop-forward-reverse-trace', 'ledger', 'The same upstream gradient meets different local multipliers', 'A cached forward value determines each local derivative. Multiplying by the shared upstream signal gives different parameter gradients.', { rows: [['Forward: y = a x; a = 3, x = 2', ['3', 'x', '2', '=', '6'], 'input'], ['Backward: upstream = 4', ['da: 8', 'dx: 12'], 'output']] });
define('estimator-error-reference-points', 'geometry', 'Bias and spread use different reference points', 'The estimate mean is displaced from the truth. Individual estimates spread around that mean, not around an invented zero-error center.', { domain: [-1, 5], vectors: [['Truth', [2, 0], 'input'], ['Mean estimate', [3, 0], 'warning'], ['One estimate', [4, 1], 'output']], guides: [[[2, 0], [3, 0]], [[3, 0], [4, 1]]] });
define('capacity-regimes-double-descent', 'curve', 'One possible test-error curve has an interpolation peak', 'This schematic illustrates a possible regime, not a universal law. Training error can reach zero while test error rises and later falls.', { domain: [0, 10], range: [0, 1], xLabel: 'model capacity; schematic units', yLabel: 'error; schematic units', series: [['Training error', [[0, .9], [2, .4], [4, 0], [10, 0]]], ['Possible test error', [[0, .8], [2, .35], [3, .4], [4, .9], [5, .5], [7, .3], [10, .2]], 'warning']] });
define('clt-data-vs-sample-means', 'curve', 'Averages concentrate; the underlying observations do not change', 'For independent exponential observations, a sample mean has variance 1/n. Averaging does not turn the original skewed population into a Gaussian.', { domain: [0, 4], range: [0, 1.3], xLabel: 'observed value or sample mean', yLabel: 'probability density', series: [['Original exponential density', sample(value => Math.exp(-value), 0, 4)], ['Mean of 10; normal approximation', sample(value => gaussian(value, 1, 1 / Math.sqrt(10)), 0, 4), 'output']] });
define('cnn-spatial-channel-pyramid', 'ledger', 'Spatial pooling keeps the channel axis', 'Global average pooling reduces every spatial channel map to one number. It does not merge all channels into one scalar.', { rows: [['Before pooling: 3 channels', ['4x4', '4x4', '4x4'], 'input'], ['After global average pooling', ['mean 1', 'mean 2', 'mean 3'], 'output']] });
define('cold-start-signal-handoff', 'curve', 'Three outcomes update a distribution, not a certainty', 'Starting from Beta(1,1), two successes and one failure give Beta(3,2). Its mean is 0.6, but substantial uncertainty remains.', { domain: [0, 1], range: [0, 2], xLabel: 'unknown success probability', yLabel: 'probability density', series: [['Prior: Beta(1,1)', sample(() => 1, 0, 1)], ['Posterior: Beta(3,2)', sample(value => 12 * value ** 2 * (1 - value), 0, 1), 'output']] });
define('cross-entropy-confident-wrong-gradient', 'curve', 'Confidence in the wrong answer need not erase the gradient', 'For a positive label, binary cross-entropy has logit gradient p - 1. The squared-error gradient also contains the small sigmoid derivative.', { domain: [0, 1], range: [0, 1], xLabel: 'predicted positive probability', yLabel: 'absolute logit gradient', series: [['Cross-entropy: 1 - p', sample(value => 1 - value, 0, 1)], ['Squared error: 2p(1-p)^2', sample(value => 2 * value * (1 - value) ** 2, 0, 1), 'warning']] });
define('logistic-gradient-cancellation', 'ledger', 'The sigmoid factors cancel, but the input stays', 'The cancellation gives dL/dz = p-y. A weight still receives the chain-rule factor x.', { rows: [['Loss to logit', ['dL/dz', '=', 'p - y'], 'input'], ['Logit to weight', ['dz/dw', '=', 'x'], 'state'], ['Full weight gradient', ['dL/dw', '=', '(p-y)x'], 'output']] });
define('attention-scaling-variance-trace', 'bars', 'Square-root scaling removes the dimension factor under the assumptions', 'With independent, unit-variance coordinate products, a 64-term dot product has variance 64. Dividing by 8 returns variance 1; dependence can change this result.', { rows: [['Unscaled dot product; d = 64', 64, 'input'], ['Divide dot product by sqrt(64)', 1, 'output']], unit: ' variance units' });
define('determinant-signed-area-collapse', 'geometry', 'Column vectors sweep a signed area', 'Vectors (2,0) and (1,2) span area 4. Swapping their order flips the determinant to -4 without changing the polygon.', { domain: [-1, 4], vectors: [['First column', [2, 0], 'input'], ['Second column', [1, 2], 'output']], polygon: [[0, 0], [2, 0], [3, 2], [1, 2]] });
define('discrete-gradients-forward-backward-paths', 'curve', 'A backward surrogate is a declared choice, not the hard derivative', 'The step output is flat away from its jump. A straight-through rule can pass a nonzero backward signal without being that derivative.', { domain: [-2, 2], range: [0, 1.2], xLabel: 'input', yLabel: 'output or backward multiplier', series: [['Hard threshold output', [[-2, 0], [-.001, 0], [.001, 1], [2, 1]]], ['Chosen straight-through multiplier', [[-2, 1], [2, 1]], 'warning']] });
define('dropout-inverted-expectation', 'ledger', 'Preserve the expectation, not each realization', 'With keep probability 1/2, activation 4 becomes either 0 or 8. Their average is 4; neither individual pass preserves the input.', { rows: [['Input activation', ['4'], 'input'], ['Two equally likely outcomes', ['0', '8'], 'warning'], ['Expected output', ['.5x0', '+', '.5x8', '=', '4'], 'output']] });
define('spectral-theorem-change-basis', 'ledger', 'Separate coordinate changes from eigenvalue scaling', 'For this diagonal example, (3,1) becomes (6,3) under eigenvalues 2 and 3. A general symmetric matrix first uses its orthonormal eigenbasis.', { rows: [['Coordinates in eigenbasis', ['3', '1'], 'input'], ['Scale each eigen-coordinate', ['x 2', 'x 3'], 'state'], ['Coordinates after scaling', ['6', '3'], 'output']] });
define('embedding-normalization-metric-equivalence', 'bars', 'Length wins the raw dot product; direction wins after normalization', 'For q=(1,0), A=(10,10) beats B=(1,0) by raw dot product. Cosine reverses the order because B points exactly along q.', { groups: [{ title: 'Raw dot product: scale 0 to 10', rows: [['A: (10,10)', 10, 'input'], ['B: (1,0)', 1, 'neutral']] }, { title: 'Cosine: scale 0 to 1', rows: [['A: diagonal to the query', Math.SQRT1_2, 'warning'], ['B: exactly along the query', 1, 'output']], domain: [0, 1] }] });
define('mutual-information-noisy-channel-budget', 'parts', 'A noisy shared bit retains information despite fair marginals', 'A binary symmetric channel flips 10% of fair input bits. Its mutual information is 1-H(0.1), about 0.531 bits.', { rows: [['retained mutual information', 1 + .1 * Math.log2(.1) + .9 * Math.log2(.9), 'output'], ['conditional uncertainty', -.1 * Math.log2(.1) - .9 * Math.log2(.9), 'warning']], unit: ' bit', digits: 1 });
define('uncertainty-within-between-models', 'geometry', 'Confident models can still disagree with each other', 'The two predictive means are separated. Total predictive variance includes both within-model variance and variance between these means.', { domain: [-1, 5], vectors: [['Model A mean', [1, 1], 'input'], ['Model B mean', [4, 1], 'warning'], ['Mixture mean', [2.5, 1], 'output']], guides: [[[1, .6], [1, 1.4]], [[4, .6], [4, 1.4]], [[1, 1], [4, 1]]] });
define('covariance-centered-product-signs', 'matrix', 'Centered products determine contribution signs', 'Same-side deviations contribute positively; opposite-side deviations negatively. A value at its mean contributes exactly zero.', { values: [['+', '-'], ['-', '+'], ['0', '0']], columns: ['y below', 'y above'], labels: ['x below', 'x above', 'x mean'], selected: [[0, 0], [1, 1]], blocked: [[0, 1], [1, 0]] });
define('ece-weighted-bin-contributions', 'bars', 'Bin size and calibration gap multiply', 'A 10% bin with a 40-point gap and a 40% bin with a 10-point gap both contribute four percentage points to ECE.', { rows: [['.1 x .4', .04, 'input'], ['.4 x .1', .04, 'output']], unit: ' ECE' });
define('gradient-magnitude-through-depth', 'curve', 'Repeated local multipliers compound', 'Ten multipliers of 0.8 retain about 0.107 of the signal; ten of 1.2 produce about 6.19. Local slopes determine the product, not depth alone.', { domain: [0, 10], range: [0, 6.5], xLabel: 'number of repeated factors', yLabel: 'relative gradient magnitude', series: [['Each factor = 0.8', sample(value => .8 ** value, 0, 10)], ['Each factor = 1.2', sample(value => 1.2 ** value, 0, 10), 'warning']] });
define('exponential-family-sufficient-stat-moment-match', 'ledger', 'Keep the statistic that the likelihood needs', 'For Bernoulli observations, five trials and three successes give maximum-likelihood mean 3/5. Their order does not change this likelihood.', { rows: [['Observed Bernoulli outcomes', ['1', '0', '1', '1', '0'], 'input'], ['Sufficient statistics', ['sum: 3', 'n: 5'], 'state'], ['Moment match', ['3', '/', '5', '=', '.6'], 'output']] });
define('factor-model-covariance-decomposition', 'matrix', 'Shared exposure creates off-diagonal covariance', 'For one unit-variance factor with loadings (1,2), the shared covariance is [[1,2],[2,4]]. Independent noise adds diagonal terms only.', { values: [[1, 2], [2, 4]], columns: ['asset A', 'asset B'], labels: ['asset A', 'asset B'], selected: [[0, 1], [1, 0]] });
define('fm-shared-embeddings-unseen-pair', 'ledger', 'An unseen pair can reuse embeddings learned elsewhere', 'The pair weight is the dot product of shared embeddings. This exact pair needs no independently observed coefficient.', { rows: [['Feature A embedding', ['1', '2'], 'input'], ['Feature B embedding', ['3', '1'], 'state'], ['Interaction from shared coordinates', ['1x3', '+', '2x1', '=', '5'], 'output']] });
define('fp16-bf16-bit-budget', 'parts', 'The same 16 bits cannot maximize range and precision', 'BF16 allocates eight exponent bits and seven fraction bits; FP16 uses five and ten. Both reserve one sign bit.', { rows: [['BF16 exponent bits', 8, 'input'], ['BF16 fraction bits', 7, 'output'], ['sign bit', 1, 'neutral']], unit: ' bits' });
define('gmm-soft-responsibility', 'parts', 'One observation can update both components', 'Responsibilities 0.7 and 0.3 sum to one. The point contributes fractionally to both component updates, rather than being duplicated as two full observations.', { rows: [['component A responsibility', .7, 'input'], ['component B responsibility', .3, 'output']], unit: ' observation' });
define('gpu-roofline-batching-ridge', 'bars', 'Less traffic raises a ceiling, not a measured speedup', 'At one TB/s, doubling arithmetic intensity doubles the memory ceiling. Other bottlenecks may prevent reaching either ceiling.', { rows: [['Before traffic reduction', 1, 'input'], ['After halving HBM traffic', 2, 'output']], unit: ' TFLOP/s ceiling' });
define('gradient-accumulation-state-boundary', 'ledger', 'Four backwards can share one unchanged parameter state', 'Accumulate gradients while parameters remain fixed; one optimizer step follows the accumulation boundary.', { rows: [['Four microbatch backwards', ['g1', '+g2', '+g3', '+g4'], 'input'], ['Parameter version during all four', ['v0', 'v0', 'v0', 'v0'], 'state'], ['One optimizer step', ['sum g', '->', 'v1'], 'output']] });
define('gradient-boosting-residual-update', 'ledger', 'Fit what remains, not the same target from scratch', 'For squared error, target 10 minus current prediction 7 leaves residual 3. A shrinkage of 0.1 adds 0.3, not 3 or 10.', { rows: [['Current residual', ['10', '-', '7', '=', '3'], 'input'], ['Shrunk tree contribution', ['.1', 'x', '3', '=', '.3'], 'state'], ['Updated prediction', ['7', '+', '.3', '=', '7.3'], 'output']] });
define('global-norm-clipping-radial-projection', 'geometry', 'Shorten the vector without turning it', 'The vector (3,4) has norm 5. Clipping to norm 2 multiplies every component by 2/5, giving (1.2,1.6).', { domain: [-1, 5], vectors: [['Original gradient; norm 5', [3, 4], 'input'], ['Clipped gradient; norm 2', [1.2, 1.6], 'output']] });
define('gcn-neighborhood-to-matrix-row', 'matrix', 'One adjacency row chooses one neighborhood', 'Only B and its selected neighbors enter this schematic B update. An absent edge supplies no message at this layer.', { values: [[1, 1, 0], [1, 1, 1], [0, 1, 1]], columns: ['A', 'B', 'C'], labels: ['A', 'B', 'C'], selected: [[1, 0], [1, 1], [1, 2]], blocked: [[0, 2], [2, 0]] });
define('loss-estimation-vs-decision', 'bars', 'Probability 0.8 can still favor the negative action', 'False-positive cost is 10; false-negative cost is 1. Expected positive-action cost is 0.2x10=2; expected negative-action cost is 0.8x1=0.8.', { rows: [['Act positive: expected cost', 2, 'warning'], ['Act negative: expected cost', .8, 'output']], unit: ' cost units' });
define('logit-odds-probability-map', 'ledger', 'Equal logit steps multiply odds', 'Moving z from 0 to 1 multiplies odds by e. The probability change is nonlinear and becomes smaller near the extremes.', { rows: [['Logit z', ['0', '1', '2'], 'input'], ['Odds exp(z)', ['1', '2.718', '7.389'], 'state'], ['Probability odds/(1+odds)', ['.500', '.731', '.881'], 'output']] });
define('kernel-trick-exact-shortcut', 'ledger', 'The kernel computes the same inner product without expanding features', 'The coordinate dot product is 4; the kernel (4+2)^2 is 36. This is an exact shortcut, not an approximation.', { rows: [['Original coordinate products', ['1x2', '+', '2x1', '=', '4'], 'input'], ['Kernel arithmetic', ['(4+2)', 'squared', '=', '36'], 'output']] });
define('kl-direction-candidate-ranking', 'bars', 'The averaging distribution reverses this candidate ranking', 'Target P=(.9,.1): forward KL prefers Q1=(.7,.3); reverse KL prefers Q2=(.99,.01). Both panels use the same scale in nats; lower is better.', { groups: [{ title: 'Forward KL: average under P', rows: [['Q1=(.7,.3): preferred', divergence([.9, .1], [.7, .3]), 'output'], ['Q2=(.99,.01)', divergence([.9, .1], [.99, .01]), 'warning']], domain: [0, .16], unit: ' nats' }, { title: 'Reverse KL: average under Q', rows: [['Q1=(.7,.3)', divergence([.7, .3], [.9, .1]), 'warning'], ['Q2=(.99,.01): preferred', divergence([.99, .01], [.9, .1]), 'output']], domain: [0, .16], unit: ' nats' }] });
define('label-smoothing-finite-confidence-optimum', 'parts', 'Uniform mixing gives the true class part of the uniform mass too', 'For five classes and smoothing 0.1, the true target is 0.9+0.1/5=0.92. The other four classes receive 0.02 each.', { rows: [['true-class target', .92, 'output'], ['other four classes combined', .08, 'warning']], unit: ' target mass' });
define('warmup-cosine-three-checkpoints', 'curve', 'A nonzero floor lifts the cosine midpoint', 'After warmup, a cosine from 1 to 0.2 reaches 0.6 at its midpoint, not 0.5. The schedule is a chosen control, not a fitted learning curve.', { domain: [0, 10], range: [0, 1.1], xLabel: 'training schedule position', yLabel: 'relative learning rate', series: [['Warmup then cosine to floor 0.2', [[0, 0], [2, 1], ...sample(value => .2 + .4 * (1 + Math.cos(Math.PI * (value - 2) / 8)), 2, 10)]]] });
define('ols-orthogonal-projection', 'geometry', 'The residual is perpendicular to the fitted subspace', 'Projection of y=(3,2) onto the x-axis is (3,0). The residual (0,2) is orthogonal to that fitted column space, even if coefficient representations are nonunique.', { domain: [-1, 4], vectors: [['Observed y', [3, 2], 'input'], ['Fitted projection', [3, 0], 'output'], ['Residual direction', [0, 2], 'warning']], guides: [[[3, 0], [3, 2]]] });
define('lstm-coordinate-gated-state-update', 'ledger', 'Two gates scale two different contributions', 'For one coordinate, forget gate 0.8 retains 1.6 from old state 2; input gate 0.5 admits 1.5 from candidate 3. Their sum is 3.1.', { rows: [['Retained old state', ['.8', 'x', '2', '=', '1.6'], 'input'], ['Admitted candidate', ['.5', 'x', '3', '=', '1.5'], 'state'], ['New cell state', ['1.6', '+', '1.5', '=', '3.1'], 'output']] });
define('markov-chain-stationary-flow-ledger', 'bars', 'Balanced flow preserves the distribution, not the state', 'For transition matrix [[.8,.2],[.4,.6]], stationary probabilities are 2/3 and 1/3. The weighted flows in opposite directions both equal 2/15, even though their transition probabilities differ.', { rows: [['A to B: (2/3) x .2', (2 / 3) * .2, 'input'], ['B to A: (1/3) x .4', (1 / 3) * .4, 'output']], unit: ' probability mass per step' });
define('matrix-columns-define-map', 'geometry', 'Dependent columns can cancel a nonzero input', 'If both matrix columns are (1,1), input coefficients (1,-1) produce zero. A nonzero input need not survive the linear map.', { domain: [-2, 3], vectors: [['First-column contribution', [1, 1], 'input'], ['Second-column contribution', [-1, -1], 'warning'], ['Sum: collapsed output', [0, 0], 'output']] });
define('jacobian-transpose-pullback', 'matrix', 'Each input collects contributions from every affected output', 'Output gradient (2,1) weights the two Jacobian rows. Summing each input column gives gradient (5,8).', { values: [['1x2', '2x2'], ['3x1', '4x1'], [5, 8]], columns: ['input 1', 'input 2'], labels: ['output 1', 'output 2', 'sum'], selected: [[2, 0], [2, 1]] });
define('mf-missing-rating-dot-product', 'ledger', 'A missing rating is a sum of matched coordinate products', 'The example user vector (1,2) and item vector (3,1) predict 1x3+2x1=5. This is a model estimate, not an observed rating.', { rows: [['User factors', ['1', '2'], 'input'], ['Item factors', ['3', '1'], 'state'], ['Predicted missing rating', ['3', '+', '2', '=', '5'], 'output']] });
define('mle-bernoulli-parameter-sweep', 'curve', 'Keep the data fixed; vary the candidate probability', 'For three successes in five independent trials, likelihood is p^3(1-p)^2. Its maximum occurs at p=3/5, not at a changing data sample.', { domain: [0, 1], range: [0, .04], xLabel: 'candidate Bernoulli probability', yLabel: 'likelihood of this ordered sample', series: [['Three successes; two failures', sample(value => value ** 3 * (1 - value) ** 2, 0, 1)]] });
define('bellman-stochastic-backup', 'parts', 'A chance node averages outcomes, not their maximum', 'The probability-weighted contributions are 1.5 and 0.25. Their expectation is 1.75, not the best possible return 2.', { rows: [['0.75 x return 2', 1.5, 'input'], ['0.25 x return 1', .25, 'output']], unit: ' expected return' });
define('importance-weights-effective-sample-size', 'ledger', 'Reweighting does not create observations', 'Five weights (6,1,1,1,1) sum to 10. ESS is 10 squared divided by 40, or 2.5; the actual draw count stays five.', { rows: [['Five actual draws; illustrative weights', ['6', '1', '1', '1', '1'], 'input'], ['Effective sample size', ['100', '/', '40', '=', '2.5'], 'output']] });
define('multitask-gradient-diagnosis', 'geometry', 'Equal lengths can still point in opposite directions', 'The two unit gradients cancel when summed. Equalizing norms cannot resolve their directional conflict.', { domain: [-2, 2], vectors: [['Task A gradient', [1, 0], 'input'], ['Task B gradient', [-1, 0], 'warning'], ['Combined gradient', [0, 0], 'output']] });
define('naive-bayes-evidence-ledger', 'ledger', 'Evidence ratios multiply only under the declared independence model', 'Prior odds 1:4 become 1:1 after likelihood ratio 2 and another ratio 2. Correlated evidence cannot be multiplied as independent evidence without justification.', { rows: [['Prior odds', ['1', ':', '4'], 'input'], ['Independent class-conditional evidence', ['x 2', 'x 2'], 'state'], ['Posterior odds', ['4', ':', '4'], 'output']] });
define('scaling-fixed-compute-allocation', 'curve', 'At fixed compute, parameters compete with training tokens', 'In the simplified C proportional to N times D model, doubling parameters halves the token budget. This is a constraint curve, not an optimality claim.', { domain: [1, 8], range: [0, 8], xLabel: 'relative parameter count N', yLabel: 'relative token budget D', series: [['Fixed N x D = 8', sample(value => 8 / value, 1, 8)]] });
define('flow-local-volume-density', 'areas', 'Expanding volume dilutes density, not probability mass', 'The same four dots carry mass 1 before and after expansion. Doubling the drawn area changes density from 1 to 0.5, not the total probability mass.', { unitMass: .25, regions: [{ label: 'Before expansion: four dots', width: 1, height: 1, mass: 1 }, { label: 'After expansion: the same four dots', width: 2, height: 1, mass: 1 }] });
define('perplexity-tokenizer-denominator', 'ledger', 'A token is not a fixed amount of text', 'Two tokenizations of the same four characters use different denominators. Compare total code length per shared text unit, not raw perplexity per token.', { rows: [['Tokenizer A; four text characters', ['a', 'b', 'c', 'd'], 'input'], ['Tokenizer B; same characters', ['ab', 'cd'], 'state'], ['Shared comparison unit', ['bits', '/', 'character'], 'output']] });
define('policy-gradient-advantage-sign', 'bars', 'A positive return can still be below baseline', 'Return 3 minus baseline 5 gives advantage -2. This example decreases the sampled action probability despite its positive raw return.', { rows: [['Observed return', 3, 'input'], ['Baseline', 5, 'state'], ['Advantage', -2, 'warning']] });
define('position-bias-inverse-examination-weight', 'ledger', 'The weight changes estimated contribution, not the click', 'One click under examination probability 0.2 contributes weight 5 under this estimator. It remains one observed click.', { rows: [['Observed click', ['1'], 'input'], ['Inverse examination contribution', ['1', '/', '.2', '=', '5'], 'output']] });
define('positive-definite-boundedness', 'curve', 'A flat direction escapes every finite radius', 'For f(x,y)=x squared, the points (0,y) all have loss zero. Positive semidefiniteness alone does not bound sublevel sets.', { domain: [-3, 3], range: [0, 9], xLabel: 'coordinate displacement', yLabel: 'quadratic value', series: [['Along x: x squared', sample(value => value ** 2, -3, 3)], ['Along y at x=0: zero', [[-3, 0], [3, 0]], 'warning']] });
define('precision-recall-f1-harmonic-penalty', 'bars', 'One weak side pulls down the harmonic mean', 'Precision 0.9 and recall 0.1 produce F1 0.18, not their arithmetic average 0.5.', { rows: [['Precision', .9, 'input'], ['Recall', .1, 'warning'], ['F1', .18, 'output']], domain: [0, 1] });
define('distribution-observation-unit', 'ledger', 'Count events or measure waiting time', 'A Poisson count has nonnegative integer outcomes. Exponential waiting time is continuous. The same process supplies two different random variables.', { rows: [['Events in a fixed interval; possible counts', ['0', '1', '2', '3', '...'], 'input'], ['Wait to the next event; example seconds', ['.17', '1.42', '3.08'], 'state']] });
define('random-forest-correlation-floor', 'curve', 'Correlated trees leave a variance floor', 'With unit tree variance and pairwise correlation 0.2, ensemble variance is 0.2+0.8/T. More trees cannot push this model below 0.2.', { domain: [1, 100], range: [0, 1], xLabel: 'number of trees T', yLabel: 'ensemble variance', series: [['Correlation 0.2', sample(value => .2 + .8 / value, 1, 100)], ['Independent trees', sample(value => 1 / value, 1, 100), 'output']] });
define('residual-add-gradient-split', 'ledger', 'Addition copies the upstream gradient into both branches', 'With upstream 2.2, the identity branch returns 2.2. A residual branch with derivative 0.1 returns 0.22, so the input receives 2.42.', { rows: [['Upstream at add', ['2.2'], 'input'], ['Identity / residual contributions', ['2.2', '.22'], 'state'], ['Accumulated input gradient', ['2.2', '+', '.22', '=', '2.42'], 'output']] });
define('potential-shaping-cancels-backtracking', 'ledger', 'Potential gains cancel on an undiscounted loop', 'Moving from potential 2 to 5 adds 3; returning adds -3. An entry bonus paid twice would instead accumulate reward.', { rows: [['Potential along a loop', ['2', '5', '2'], 'input'], ['Potential differences', ['+3', '-3'], 'state'], ['Total shaping reward; gamma = 1', ['3', '-', '3', '=', '0'], 'output']] });
define('pca-rank-one-projection', 'geometry', 'Keep one coordinate; discard the perpendicular residual', 'Projecting (3,2) onto the horizontal first principal direction retains (3,0). The two-unit residual is reconstruction error, not another retained coordinate.', { domain: [-1, 4], vectors: [['Original centered point', [3, 2], 'input'], ['Rank-one reconstruction', [3, 0], 'output']], guides: [[[3, 0], [3, 2]]] });
define('bm25-saturation-length-penalty', 'curve', 'More matches help less, and length changes the same match count', 'With k1=1.2 and b=0.75, the term-frequency factor saturates. Doubling relative document length reduces the factor at fixed frequency.', { domain: [0, 10], range: [0, 2.3], xLabel: 'term frequency', yLabel: 'BM25 term-frequency factor', series: [['Document at average length', sample(value => value * 2.2 / (value + 1.2), 0, 10)], ['Document twice average length', sample(value => value * 2.2 / (value + 2.1), 0, 10), 'warning']] });
define('relu-second-moment-through-depth', 'curve', 'The ReLU second moment depends on the initialization scale', 'Under the stated independence and symmetry assumptions, variance 2/fan-in offsets the half-moment loss of ReLU. This is a moment calculation, not universal gradient stability.', { domain: [0, 10], range: [0, 1.1], xLabel: 'layer count', yLabel: 'relative second moment', series: [['Variance 1/fan-in: factor 1/2 per layer', sample(value => .5 ** value, 0, 10)], ['Variance 2/fan-in: factor 1 per layer', [[0, 1], [10, 1]], 'output']] });
define('z-loss-anchors-common-offset', 'ledger', 'Shift every logit: same probabilities, different normalizer', 'Adding 2 to all logits multiplies the softmax denominator by exp(2). Softmax probabilities are unchanged, but log-normalizer-based z-loss changes.', { rows: [['Original logits', ['0', '1'], 'input'], ['Common shift +2', ['2', '3'], 'state'], ['Both softmax distributions', ['.269', '.731'], 'output']] });

define('checkpoint-boundary-recompute-balance', 'curve', 'Saving less requires rebuilding more', 'In this 16-layer model, boundary storage is 16/s and rebuilt segment storage is s. Their sum is minimized at s=4.', { domain: [1, 16], range: [0, 18], xLabel: 'segment length s', yLabel: 'relative stored states', series: [['Saved plus rebuilt: 16/s + s', sample(value => 16 / value + value, 1, 16)], ['Saved boundaries: 16/s', sample(value => 16 / value, 1, 16), 'input'], ['Rebuilt segment: s', [[1, 1], [16, 16]], 'warning']] });
define('actor-critic-two-objective-flow', 'matrix', 'The advantage is evidence for the actor, not an actor gradient route', 'Both losses can update shared features. The actor loss must not differentiate through the advantage target.', { values: [['yes', 'yes'], ['yes', 'no'], ['no', 'yes'], ['STOP', 'target']], columns: ['actor', 'critic'], labels: ['shared', 'policy', 'value', 'adv.'], selected: [[0, 0], [0, 1], [1, 0], [2, 1]], blocked: [[3, 0]] });
define('gae-residual-weight-trace', 'ledger', 'A backward recurrence equals a weighted residual sum', 'With gamma times lambda equal to 0.8, residuals (1,2,-1) give advantage 1.96. The backward scan gives the same answer.', { rows: [['Residuals from current time', ['1', '2', '-1'], 'input'], ['Decaying weights', ['1', '.8', '.64'], 'state'], ['Weighted sum', ['1', '+1.6', '-.64', '=', '1.96'], 'output']] });
define('normalization-shared-statistics', 'matrix', 'The axes reveal which values share statistics', 'BatchNorm uses column means 3 and 5 across these samples. LayerNorm uses row means 2 and 6 within each sample.', { values: [[1, 3, 2], [5, 7, 6], [3, 5, '-']], columns: ['feat 1', 'feat 2', 'LN'], labels: ['sample A', 'sample B', 'BN'], selected: [[0, 2], [1, 2], [2, 0], [2, 1]] });
define('intervals-two-randomness-models', 'intervals', 'Repeat the data, or condition on these data', 'Repeated samples move the interval while the true probability stays fixed. Below, three successes and two failures with a uniform prior give Beta(4,3), conditioned on one observed sample. The ten intervals are illustrative, not a measured coverage rate.', { intervals: [[.15, .65], [.3, .72], [.23, .6], [.42, .84], [.55, .91], [.2, .56], [.33, .77], [.14, .52], [.38, .82], [.26, .67]] });
define('belief-propagation-branch-summary', 'ledger', 'Sum out the hidden branch before combining messages', 'Hidden states contribute 0.3x2 and 0.7x0.5. Their message sum is 0.95; normalization follows evidence combination.', { rows: [['Hidden-state weighted terms', ['.3x2', '.7x.5'], 'input'], ['Marginal message', ['.6', '+', '.35', '=', '.95'], 'output']] });
define('bert-mlm-selection-and-corruption', 'population', 'Selected for prediction does not always mean replaced by MASK', 'For 1,000 tokens, 150 are prediction targets. The 80/10/10 corruption rule gives 120 masks, 15 random replacements, and 15 unchanged targets.', { rows: [['masked targets', 120, 'focus'], ['randomly replaced targets', 15, 'warning'], ['unchanged targets', 15, 'output'], ['unselected tokens', 850, 'neutral']], unit: 'one input token' });
define('bootstrap-resampling-unit', 'ledger', 'Repeated users bring their paired events with them', 'Sampling users preserves within-user event pairing. Resampling isolated rows would answer a different uncertainty question.', { rows: [['Original user clusters', ['A: a1,a2', 'B: b1,b2', 'C: c1,c2'], 'input'], ['One bootstrap user sample', ['A: a1,a2', 'C: c1,c2', 'A: a1,a2'], 'output']] });
define('bptt-detach-cuts-gradient-not-state', 'matrix', 'Detach cuts the backward path while preserving the forward value', 'The next segment receives the same hidden value. Its gradient cannot cross the detach boundary into earlier computation.', { values: [['h=2', 'h=2'], ['yes', 'yes'], ['yes', 'STOP']], columns: ['before', 'after'], labels: ['value', 'forward', 'backward'], selected: [[0, 1], [1, 1]], blocked: [[2, 1]] });
define('causal-confounding-dag', 'connections', 'A common cause can create an association without a treatment effect', 'The common cause points to both messages and churn. Their association alone does not establish a causal arrow between them.', { nodes: [['Prior engagement', 2, 0, 'state'], ['Messages received', 0, 1, 'input'], ['Churn outcome', 1, 1, 'output']], edges: [[0, 1], [0, 2]] });
define('crf-global-path-scoring', 'bars', 'Transition penalties can defeat the locally best labels', 'Local emissions favor AA, but its transition penalty is -5. Globally, AB or BA scores 3 and beats AA at -1.', { rows: [['AA: emissions 4; transition -5', -1, 'warning'], ['AB: emissions 3; transition 0', 3, 'output'], ['BB: emissions 2; transition 0', 2, 'neutral']] });
define('ctc-blank-preserves-repeat', 'ledger', 'The blank keeps two equal letters from merging', 'First merge consecutive identical emissions, then remove blanks. A blank between the a emissions preserves the repeated output aa.', { rows: [['Alignment path', ['a', 'a', '_', 'a'], 'input'], ['Merge consecutive repeats', ['a', '_', 'a'], 'state'], ['Remove blank', ['a', 'a'], 'output']] });
define('content-profile-scores-new-item', 'ledger', 'Shared features connect a new item to known interests', 'User interests (2,1) and new-item features (1,3) score 5 by dot product. The item needs no interaction history for this content score.', { rows: [['User interest coordinates', ['2', '1'], 'input'], ['New item content coordinates', ['1', '3'], 'state'], ['Content match score', ['2x1', '+', '1x3', '=', '5'], 'output']] });
define('ring-attention-fixed-query-kv-cycle', 'ledger', 'Merge numerators and normalizers, not local answers', 'Local averages 2 and 10 cannot be averaged equally when their normalizers differ. Combined numerator 84 over normalizer 10 gives 8.4.', { rows: [['KV block A: weighted sum / normalizer', ['4', '/', '2', '=', '2'], 'input'], ['KV block B: weighted sum / normalizer', ['80', '/', '8', '=', '10'], 'state'], ['Merged result; same query', ['84', '/', '10', '=', '8.4'], 'output']] });
define('infonce-positive-candidate-set', 'ledger', 'The positive participates in its own denominator', 'The loss compares the declared positive against the whole candidate set. A relevant but undeclared item remains a negative under this labeling.', { rows: [['Numerator', ['positive'], 'output'], ['Denominator candidates', ['positive', 'negative 1', 'negative 2'], 'input']] });
define('im2col-patches-become-columns', 'matrix', 'Each patch becomes one matrix column', 'Two adjacent 2x2 patches become these columns. Input rows are (1,2,3) and (4,5,6); shared pixels appear in both patches.', { values: [[1, 2], [2, 3], [4, 5], [5, 6]], columns: ['patch 0', 'patch 1'], labels: ['top L', 'top R', 'low L', 'low R'], selected: [[0, 0], [1, 0], [2, 0], [3, 0]] });
define('decoding-fixed-count-adaptive-mass', 'parts', 'A probability threshold chooses a count only after seeing the distribution', 'Top-k with k=2 keeps mass 0.70 here. Top-p with p=0.8 needs the third token too, reaching mass 0.85.', { rows: [['first token', .45, 'input'], ['second token', .25, 'input'], ['third token: needed for p=.8', .15, 'output'], ['remaining two tokens', .15, 'neutral']], unit: ' probability mass' });
define('diffusion-training-sampling-asymmetry', 'ledger', 'Train at one sampled noise level; sample through a chain', 'A training example can use only its sampled time t. Generation depends on successive reverse transitions; it cannot parallelize away those dependencies.', { rows: [['One training example', ['x0', '+noise', 'xt', 'loss at t'], 'input'], ['One generation chain', ['z3', 'z2', 'z1', 'x0'], 'output']] });
define('domain-adaptation-shift-assumptions', 'matrix', 'Matching inputs can hide a reversed label rule', 'Both domains can show x=0 and x=1 equally often. Their opposite labels make this concept shift, not merely covariate shift.', { values: [[0, 1], [1, 0]], columns: ['x=0', 'x=1'], labels: ['domain A', 'domain B'], selected: [[0, 0], [0, 1]], blocked: [[1, 0], [1, 1]] });
define('em-lower-bound-climb', 'curve', 'Maximize a bound that touches the current objective', 'This illustrative bound touches at 0.5. Its maximum at 0.25 raises the objective from -0.25 to -0.0625.', { domain: [-1, 1], range: [-4, 0], xLabel: 'parameter; current 0.5, next 0.25', yLabel: 'objective or lower bound', series: [['Objective: -theta squared', sample(value => -(value ** 2), -1, 1)], ['Touching bound at theta=.5', sample(value => -(value ** 2) - (value - .5) ** 2, -1, 1), 'warning']] });
define('explain-backprop-one-loss-all-gradients', 'connections', 'One scalar loss sends contributions to every ancestor', 'A reverse traversal accumulates every outgoing contribution before continuing through a shared node. Separate full backwards per parameter are unnecessary.', { nodes: [['Parameter A', 0, 0, 'input'], ['Parameter B', 1, 0, 'input'], ['Shared intermediate', 2, 1, 'state'], ['One scalar loss', 2, 2, 'output']], edges: [[3, 2], [2, 0], [2, 1]] });
define('bandit-three-selection-rules', 'ledger', 'The best mean need not win an uncertainty-aware choice', 'The mean favors A. An upper-confidence score or posterior draw can favor B; the table uses illustrative, not empirical, values.', { rows: [['Estimated means', ['A: .65', 'B: .60'], 'input'], ['Mean plus uncertainty bonus', ['A: .67', 'B: .72'], 'state'], ['One posterior draw', ['A: .62', 'B: .75'], 'output']] });
define('gp-conditioning-local-uncertainty', 'curve', 'An exact observation removes uncertainty locally', 'A noiseless observation at zero locally constrains this unit RBF prior. Posterior standard deviation is sqrt(1-exp(-x squared)); uncertainty grows with distance.', { domain: [-3, 3], range: [0, 1.1], xLabel: 'input distance from observed x=0', yLabel: 'predictive standard deviation', series: [['Prior standard deviation', [[-3, 1], [3, 1]]], ['Posterior standard deviation', sample(value => Math.sqrt(1 - Math.exp(-(value ** 2))), -3, 3), 'output']] });
define('d-separation-conditioning-switch', 'matrix', 'Conditioning can close one path and open another', 'A conditioned chain middle blocks its path. Conditioning on a collider can open the path between its parents.', { values: [['open', 'closed'], ['closed', 'open']], columns: ['uncond.', 'cond.'], labels: ['chain', 'collider'], selected: [[0, 1], [1, 0]], blocked: [[0, 0], [1, 1]] });
define('gqa-eight-queries-two-kv-groups', 'ledger', 'Eight queries share two KV groups, not two outputs', 'Each query still produces its own result. Sharing key/value projections changes cache storage, not the number of query outputs.', { rows: [['KV group 1 serves four queries', ['Q1', 'Q2', 'Q3', 'Q4'], 'input'], ['KV group 2 serves four queries', ['Q5', 'Q6', 'Q7', 'Q8'], 'state'], ['Eight distinct query outputs', ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8'], 'output']] });
define('lora-zero-impact-first-gradient', 'ledger', 'Zero adapter output does not imply zero learning signal', 'With A=1, B=0, x=2, upstream gradient 3, and scale 1, the initial adapter output is zero. Its first B gradient is 6 while the A gradient is zero.', { rows: [['Initial adapter factors', ['A: 1', 'B: 0'], 'input'], ['Initial adapter output', ['B A x', '=', '0'], 'state'], ['First backward gradients', ['dA: 0', 'dB: 6'], 'output']] });
define('autograd-shared-node-accumulate-then-propagate', 'ledger', 'Collect both contributions before propagating through a shared node', 'At shared u=6, the two branches contribute 12 and 2. Add them to 14, then multiply by du/dx=3 to obtain 42.', { rows: [['Branch contributions at u', ['12', '2'], 'input'], ['Accumulate once', ['12', '+', '2', '=', '14'], 'state'], ['Propagate through u=3x', ['14', 'x', '3', '=', '42'], 'output']] });
define('streaming-metrics-counts-before-f1', 'matrix', 'Add counts before taking a nonlinear metric', 'The merged counts give F1=18/22, about 0.818. Averaging shard F1 values 0.8 and 1 would incorrectly give 0.9.', { values: [[8, 2, 2], [1, 0, 0], [9, 2, 2]], columns: ['TP', 'FP', 'FN'], labels: ['shard A', 'shard B', 'sum'], selected: [[2, 0], [2, 1], [2, 2]] });
define('kmeans-assign-then-update', 'ledger', 'Assignments stay fixed during a centroid update', 'Points 0 and 2 assigned to the left centroid give mean 1. Points 8 and 10 assigned right give mean 9.', { rows: [['Fixed cluster assignments', ['0, 2', '8, 10'], 'input'], ['Old centroids', ['0', '10'], 'state'], ['Updated assigned means', ['1', '9'], 'output']] });
define('distillation-temperature-dark-knowledge', 'matrix', 'Temperature softens alternatives without changing the winner', 'For logits (2,1,0), temperature 2 increases the probability of both losing classes. The top class remains the same.', { values: [[.665, .245, .09], [.506, .307, .186]], columns: ['class A', 'class B', 'class C'], labels: ['T=1', 'T=2'], selected: [[0, 0], [1, 0]] });
define('kge-antisymmetry-score-swap', 'ledger', 'Complex conjugation makes relation direction matter', 'With h=i, r=i, and t=1, the forward ComplEx score is -1. Swapping head and tail gives +1 for the same relation.', { rows: [['Forward: real(h r conjugate(t))', ['i', 'x i', 'x 1', '=', '-1'], 'input'], ['Reverse: real(t r conjugate(h))', ['1', 'x i', 'x -i', '=', '+1'], 'output']] });
define('l1-l2-first-contact-geometry', 'geometry', 'A corner can put one coefficient exactly at zero', 'For squared distance from (2,0.4), the unit L1 constraint selects (1,0). The unit L2 constraint selects a nonzero second coordinate.', { domain: [-2, 4], vectors: [['L1-constrained optimum', [1, 0], 'input'], ['L2-constrained optimum', [2 / Math.hypot(2, .4), .4 / Math.hypot(2, .4)], 'output']], polygon: [[1, 0], [0, 1], [-1, 0], [0, -1]], circles: [[[0, 0], 1, 'output']] });
define('lambdarank-swap-impact-weight', 'bars', 'The same pair error can carry very different ranking cost', 'With one relevant item and IDCG=1, swapping ranks 1 and 2 changes NDCG far more than swapping 9 and 10.', { rows: [['Relevant item: rank 1 to 2', 1 - 1 / Math.log2(3), 'warning'], ['Relevant item: rank 9 to 10', 1 / Math.log2(10) - 1 / Math.log2(11), 'input']], unit: ' NDCG change' });

define('forward-backward-viterbi-chain-objectives', 'ledger', 'The best state marginal need not belong to the best whole path', 'Consider paths AA=.35, AB=.30, BA=.05, BB=.30. Marginals favor A then B, but the highest-probability complete path is AA.', { rows: [['Complete path probabilities', ['AA .35', 'AB .30', 'BA .05', 'BB .30'], 'input'], ['Independent marginal choices', ['A: .65', 'B: .60'], 'warning'], ['Viterbi whole-path choice', ['AA: .35'], 'output']] });
define('hmm-hidden-chain-local-emissions', 'connections', 'Observations connect through the hidden chain', 'The hidden states transmit dependence through time. Each observation is emitted locally; there is no direct observation-to-observation edge.', { nodes: [['Hidden state 1', 0, 0, 'state'], ['Hidden state 2', 1, 0, 'state'], ['Observation 1', 0, 1, 'input'], ['Observation 2', 1, 1, 'output']], edges: [[0, 1], [0, 2], [1, 3]] });
define('learning-rate-range-to-schedule', 'curve', 'Choose the useful descent region before instability', 'This schematic sweep descends, then diverges. The final tested learning rate is not the recommended schedule maximum.', { domain: [0, 5], range: [0, 5], xLabel: 'log learning rate; schematic units', yLabel: 'observed sweep loss; schematic', series: [['Illustrative range-test trajectory', [[0, 4], [1, 3.8], [2, 2], [3, 1], [4, 1.5], [5, 5]]]] });
define('attention-causal-mask-row', 'matrix', 'A future position contributes exactly zero weight', 'For query position 2, only positions 0 through 2 can contribute. Future logits are masked before the weighted value sum.', { values: [[.2, .3, .5, 0, 0]], columns: ['0', '1', '2', '3', '4'], labels: ['q=2'], selected: [[0, 0], [0, 1], [0, 2]], blocked: [[0, 3], [0, 4]] });
define('batched-top-k-running-invariant', 'ledger', 'A discarded candidate cannot beat the retained frontier later', 'The running top two of (9,3) and (7,2) are 9 and 7. A new 8 replaces 7, while previously discarded 3 remains irrelevant.', { rows: [['Batch-local candidates', ['9', '3', '7', '2'], 'input'], ['Running top two', ['9', '7'], 'state'], ['Merge next candidate 8', ['9', '8'], 'output']] });
define('kv-cache-newest-row-equivalence', 'matrix', 'The newest query still sees the same earlier keys', 'Cached projections replace recomputation, not attention access. Only the newest query row is computed at this decode step.', { values: [['cache', 'cache', 'cache', 'new'], ['yes', 'yes', 'yes', 'yes']], columns: ['K0', 'K1', 'K2', 'K3'], labels: ['keys', 'Q3 uses'], selected: [[1, 0], [1, 1], [1, 2], [1, 3]] });
define('pre-norm-decoder-residual-stream', 'connections', 'The identity stream bypasses the normalized learned branch', 'Normalization feeds the learned branch. Its output adds back to the original residual stream rather than replacing that stream.', { nodes: [['Original residual x', 2, 0, 'input'], ['Normalize then branch F', 0, 1, 'state'], ['Identity: keep x', 1, 1, 'input'], ['x + F(norm(x))', 2, 2, 'output']], edges: [[0, 1], [0, 2], [1, 3], [2, 3]] });
define('kv-cache-one-write-growing-read', 'bars', 'One new cache write does not mean one key read', 'At decode position 8, write one new key/value pair while attention reads eight available pairs. Persistent storage grows with the prefix.', { rows: [['New pair written this step', 1, 'output'], ['Pairs read for current attention', 8, 'input']], unit: ' KV pairs' });
define('linear-attention-two-escape-routes', 'ledger', 'Change the position representation, or reorder feature products', 'Low-rank position compression and kernel feature factorization are distinct routes. Neither is simply exact softmax attention without its quadratic cost.', { rows: [['Position-compression route', ['N keys', '->', 'K summaries'], 'input'], ['Kernel-factorization route', ['sum phi(k)v', 'sum phi(k)'], 'state'], ['Query-time normalized read', ['phi(q) x S', '/', 'phi(q) x z'], 'output']] });
define('fp16-loss-scaling-ledger', 'ledger', 'Scale before rounding can erase a small gradient', 'A tiny gradient may underflow in FP16. Scaling the loss first preserves its intermediate representation; unscaling recovers the intended magnitude before stepping.', { rows: [['Unscaled path', ['tiny g', 'round', '0'], 'warning'], ['Scaled backward path', ['S x g', 'nonzero'], 'input'], ['Before optimizer step', ['divide S', 'recover g'], 'output']] });
define('mixup-cutmix-input-target-coupling', 'parts', 'The realized patch area determines the target mixture', 'A CutMix patch covering 30 of 100 pixels assigns target weight 0.3 to its source. Use realized clipped area, not just the sampled parameter.', { rows: [['retained base-image target', .7, 'input'], ['inserted patch target', .3, 'output']], unit: ' target mass' });
define('shap-all-orderings-credit', 'ledger', 'Credit averages marginal contributions across orderings', 'With two features, A enters first in one ordering and after B in the other. Average both contributions rather than choosing a favored ordering.', { rows: [['Ordering A then B', ['A: v(A)-v(0)', 'B: v(AB)-v(A)'], 'input'], ['Ordering B then A', ['B: v(B)-v(0)', 'A: v(AB)-v(B)'], 'state'], ['Feature A credit', ['mean of', 'both A gains'], 'output']] });
define('multi-head-attention-split-recombine', 'matrix', 'Heads split features, not the token sequence', 'Every head can attend over all tokens. Concatenating head outputs restores the feature dimension before the output projection.', { values: [['all', 'all', 'all'], ['all', 'all', 'all']], columns: ['token 0', 'token 1', 'token 2'], labels: ['head 1', 'head 2'], selected: [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]] });
define('positional-relative-mechanisms', 'ledger', 'Rotating features and shifting scores are different operations', 'RoPE rotates query/key coordinates before their dot product. ALiBi instead subtracts a head-specific distance penalty from the attention score.', { rows: [['RoPE operation', ['rotate q', 'rotate k', 'dot'], 'input'], ['ALiBi distances 0, 1, 2; slope .5', ['0', '-.5', '-1'], 'state'], ['Where the position enters', ['features', 'scores'], 'output']] });
define('ppo-clipped-surrogate-plateaus', 'curve', 'Clipping removes incentive in only one direction', 'Positive advantage plateaus above ratio 1.2; negative advantage plateaus below 0.8. The policy ratio itself is not bounded by this objective.', { domain: [0, 2], range: [-2, 1.5], xLabel: 'new / old action probability ratio', yLabel: 'clipped surrogate objective', series: [['Advantage +1', sample(value => Math.min(value, 1.2), 0, 2)], ['Advantage -1', sample(value => Math.min(-value, -.8), 0, 2), 'warning']] });
define('pruning-shape-versus-zeros', 'matrix', 'Zeros leave the dense shape intact', 'Unstructured zeros still occupy this dense 3x3 matrix. Removing an entire channel changes the shape and can remove a whole row.', { values: [[1, 0, 2], [0, 0, 0], [3, 0, 4]], columns: ['c0', 'c1', 'c2'], labels: ['row 0', 'remove', 'row 2'], selected: [[0, 0], [2, 2]], blocked: [[1, 0], [1, 1], [1, 2]] });
define('q-learning-off-policy-backup', 'ledger', 'The target chooses the best next action, not the sampled one', 'With next-action values 2 and 5, reward 1, and discount 0.9, the Q-learning target is 5.5. Behavior may have sampled the action valued at 2.', { rows: [['Next action values', ['A: 2', 'B: 5'], 'input'], ['Behavior sampled A; backup uses max', ['max(2,5)', '=', '5'], 'state'], ['Target', ['1', '+', '.9x5', '=', '5.5'], 'output']] });
define('quantization-scale-granularity', 'ledger', 'An outlier can erase both small signs under one scale', 'Shared scale 2 rounds -0.5 and +0.5 to zero. A separate small-channel scale preserves their opposite signs.', { rows: [['Real values; last value is another channel', ['-.5', '+.5', '254'], 'input'], ['One shared int8 scale: 2', ['0', '0', '127'], 'warning'], ['Separate channel scales: .5/127 and 2', ['-127', '127', '127'], 'output']] });
define('ranking-metrics-per-rank-credit', 'ledger', 'Reciprocal rank stops; average precision keeps collecting hits', 'Relevant ranks 2 and 4 give reciprocal rank 1/2. Average precision is ((1/2)+(2/4))/2=1/2, but its two contributions come from both hits.', { rows: [['Relevance by rank', ['no', 'yes', 'no', 'yes'], 'input'], ['Reciprocal-rank contribution', ['0', '1/2', '0', '0'], 'state'], ['Average-precision hit terms', ['0', '1/2', '0', '2/4'], 'output']] });
define('reparameterization-gradient-path', 'ledger', 'Hold the noise draw fixed while changing its transformation', 'For z=mu+sigma times epsilon, epsilon is an external fixed draw during differentiation. Gradients reach mu and sigma through this deterministic transformation.', { rows: [['One fixed noise draw', ['epsilon'], 'input'], ['Parameterized transformation', ['mu', '+', 'sigma x epsilon'], 'state'], ['Local derivatives', ['dz/dmu: 1', 'dz/dsigma: epsilon'], 'output']] });
define('resnet-shortcut-shape-contract', 'ledger', 'The shortcut must match both spatial and channel shape', 'An 8x8x16 input cannot add directly to a 4x4x32 branch output. A matching projection makes the residual addition well-defined.', { rows: [['Unchanged identity input', ['8x8x16'], 'input'], ['Learned branch output', ['4x4x32'], 'state'], ['Projected shortcut for addition', ['4x4x32'], 'output']] });
define('grpo-group-relative-signal', 'ledger', 'Equal group rewards provide no relative preference', 'Four passing outputs with equal rewards have zero centered advantage. Group-relative success does not distinguish one output from the others.', { rows: [['Group rewards; every output passes', ['1', '1', '1', '1'], 'input'], ['Subtract group mean 1', ['0', '0', '0', '0'], 'warning']] });
define('rnnt-two-axis-alignment-lattice', 'matrix', 'Time and output position are separate axes', 'A blank edge advances time; a token edge advances output position. Multiple tokens can therefore be emitted without consuming a new audio frame.', { values: [['(0,0)', '(0,1)', '(0,2)'], ['(1,0)', '(1,1)', '(1,2)'], ['(2,0)', '(2,1)', '(2,2)']], columns: ['u=0', 'u=1', 'u=2'], labels: ['t=0', 't=1', 't=2'], selected: [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]] });
define('roc-pr-same-operating-point', 'matrix', 'A roughly 1% false-positive rate can still dominate rare positives', 'Nine hits accompany 99 false alarms. Recall is 9/10, but precision is only 9/108, about 8.3%.', { values: [[9, 1], [99, 9891]], columns: ['pred +', 'pred -'], labels: ['actual +', 'actual -'], selected: [[0, 0]], blocked: [[1, 0]] });
define('rope-relative-rotation', 'geometry', 'Rotate both positions; keep their relative angle', 'A common 45-degree rotation preserves the 90-degree angle between these query/key directions. It does not imply every model output is shift-invariant.', { domain: [-1.5, 1.5], vectors: [['Original query', [1, 0], 'input'], ['Original key', [0, 1], 'state'], ['Both shifted: query', [Math.SQRT1_2, Math.SQRT1_2], 'output'], ['Both shifted: key', [-Math.SQRT1_2, Math.SQRT1_2], 'warning']] });
define('attention-source-routing', 'matrix', 'Queries determine the output rows', 'Two query positions attending to three source positions produce two output rows. The number of source keys does not create an extra output query.', { values: [['K0', 'K1', 'K2'], ['K0', 'K1', 'K2']], columns: ['source 0', 'source 1', 'source 2'], labels: ['query 0', 'query 1'], selected: [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]] });
define('semantic-segmentation-boundary-fusion', 'ledger', 'Upsampling cannot restore an edge that was discarded', 'Two high-resolution patterns can average to the same coarse value. Enlarging that value cannot identify which original boundary was present.', { rows: [['Two possible fine patterns', ['0 0 1 1', '0 1 0 1'], 'input'], ['Both average to the same coarse cell', ['.5', '.5'], 'warning'], ['Missing distinguishing evidence', ['fine-resolution skip features'], 'output']] });
define('packed-sequence-mask-boundaries', 'matrix', 'Earlier position does not permit crossing an example boundary', 'Tokens 0 and 1 belong to A; tokens 2 and 3 belong to B. B cannot attend to A even though A appears earlier in the packed buffer.', { values: [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]], columns: ['A0', 'A1', 'B0', 'B1'], labels: ['A0', 'A1', 'B0', 'B1'], selected: [[0, 0], [1, 0], [1, 1], [2, 2], [3, 2], [3, 3]], blocked: [[2, 0], [2, 1], [3, 0], [3, 1]] });
define('momentum-alternating-components', 'curve', 'Persistent directions accumulate while alternating directions cancel', 'With beta=0.5 and v=beta v+g, constant gradients approach 2. Alternating gradients remain bounded and repeatedly reverse direction.', { domain: [0, 4], range: [-1, 2], xLabel: 'update number', yLabel: 'velocity under the stated recurrence', series: [['Constant gradients +1', [[0, 0], [1, 1], [2, 1.5], [3, 1.75], [4, 1.875]]], ['Alternating gradients +1,-1', [[0, 0], [1, 1], [2, -.5], [3, .75], [4, -.625]], 'warning']] });
define('same-mesh-axis-diagonal-gap', 'matrix', 'Diagonal products cannot manufacture the missing cross-block terms', 'Computing A0B0 and A1B1 omits A0B1 and A1B0. Adding the two available blocks cannot reconstruct terms that were never computed.', { values: [['A0B0', 'absent'], ['absent', 'A1B1']], columns: ['B0', 'B1'], labels: ['A0', 'A1'], selected: [[0, 0], [1, 1]], blocked: [[0, 1], [1, 0]] });
define('softmax-ce-logit-update', 'ledger', 'Descent subtracts the gradient, so the true logit rises', 'For three equal logits and target class A, gradient is (-2/3,1/3,1/3). A learning rate of 0.3 produces logits (0.2,-0.1,-0.1).', { rows: [['Initial logits', ['0', '0', '0'], 'input'], ['Cross-entropy gradient', ['-2/3', '+1/3', '+1/3'], 'warning'], ['After subtracting .3 times gradient', ['+.2', '-.1', '-.1'], 'output']] });
define('sparse-attention-reach-patterns', 'matrix', 'A global-token relay needs two layers to connect the endpoints', 'A reads G, and G reads B. The missing direct A-to-B entry remains absent in one layer; the second layer completes the relay.', { values: [[1, 1, 0], [0, 1, 1], [0, 0, 1]], columns: ['from A', 'from G', 'from B'], labels: ['to A', 'to G', 'to B'], selected: [[0, 1], [1, 2]], blocked: [[0, 2]] });
define('speaker-verification-threshold-tradeoff', 'curve', 'One threshold trades false acceptance for false rejection', 'These illustrative score distributions overlap. Raising the threshold rejects more impostors but also rejects more genuine speakers.', { domain: [-3, 3], range: [0, .65], xLabel: 'verification score', yLabel: 'illustrative score density', threshold: .2, series: [['Impostor scores', sample(value => gaussian(value, -1, .7), -3, 3), 'warning'], ['Genuine scores', sample(value => gaussian(value, 1, .7), -3, 3), 'output']] });
define('svm-support-vectors-set-boundary', 'ledger', 'Zero dual weight means no contribution to the decision sum', 'Only support vectors with nonzero alpha enter the weighted kernel decision. Other training examples still exist but add zero here.', { rows: [['Training-example dual weights', ['.7', '0', '.4', '0'], 'input'], ['Contributions to decision sum', ['.7 y1 K1', '0', '.4 y3 K3', '0'], 'output']] });
define('bpe-corpus-merge-trace', 'ledger', 'Learn merge order once; reuse it during encoding', 'The corpus learns a prioritized merge table. A new word applies that fixed order rather than relearning frequencies from itself.', { rows: [['Learned merge order', ['l+o', 'lo+w', 'e+r'], 'input'], ['New word before merges', ['l', 'o', 'w', 'e', 'r'], 'state'], ['Apply the fixed merges', ['low', 'er'], 'output']] });
define('transformer-two-mixing-axes', 'matrix', 'Attention mixes positions; the feed-forward layer mixes features locally', 'Attention can exchange information across these token rows. Each feed-forward application transforms only its own row at this layer.', { values: [['f1', 'f2', 'f3'], ['f1', 'f2', 'f3'], ['f1', 'f2', 'f3']], columns: ['feature 1', 'feature 2', 'feature 3'], labels: ['token 0', 'token 1', 'token 2'], selected: [[1, 0], [1, 1], [1, 2]] });
define('nonlinear-embedding-local-vs-global', 'ledger', 'A gap on the embedding page is not an original-space distance', 'Two embeddings can preserve similar local neighborhoods while changing the displayed gap between groups. Do not read arbitrary global distances from that gap.', { rows: [['Same local neighbors in two layouts', ['A-B', 'C-D'], 'input'], ['Displayed between-group gap', ['layout 1: small', 'layout 2: large'], 'warning'], ['Supported interpretation', ['local neighborhood evidence'], 'output']] });
define('uat-existence-error-band', 'curve', 'An explicit approximation stays inside an error ceiling', 'Four linear segments approximate x squared with maximum error 1/64, below 0.02. This demonstrates approximation here, not trainability or generalization.', { domain: [0, 1], range: [0, .022], xLabel: 'input x', yLabel: 'absolute approximation error', series: [['Error of four-segment interpolant', sample(value => { const left = Math.min(3, Math.floor(value * 4)) / 4; const right = left + .25; const approximation = left ** 2 + (value - left) * (right + left); return approximation - value ** 2; }, 0, 1)], ['Required tolerance 0.02', [[0, .02], [1, .02]], 'warning']] });
define('value-policy-action-interface', 'parts', 'Values need an action-selection rule', 'A softmax policy for equal values assigns equal probabilities, while a greedy tie-break chooses one action. Values alone do not declare the rule.', { rows: [['action A under equal-value softmax', .5, 'input'], ['action B under equal-value softmax', .5, 'output']], unit: ' action probability' });
define('vision-patches-to-positioned-tokens', 'ledger', 'Four image patches become five positioned tokens', 'The class token adds a fifth sequence position. Patch projection changes features, while positional embeddings identify the sequence locations.', { rows: [['Image patch grid', ['P00', 'P01', 'P10', 'P11'], 'input'], ['After adding class token', ['CLS', 'P00', 'P01', 'P10', 'P11'], 'state'], ['Add one position per token', ['pos 0', 'pos 1', 'pos 2', 'pos 3', 'pos 4'], 'output']] });
define('weight-decay-adaptive-paths', 'bars', 'Adaptive rescaling changes coupled L2, not the direct decay fraction', 'For equal weights, coupled L2 passes through different fixed preconditioners. The illustrative decoupled decay remains 0.001 for either history.', { rows: [['Coupled L2; preconditioner 1', .001, 'input'], ['Coupled L2; preconditioner 10', .0001, 'warning'], ['Decoupled decay; either history', .001, 'output']], unit: ' weight reduction' });
define('why-dropout-shared-subnetworks', 'matrix', 'Different masks reuse the same underlying weights', 'A mask selects active units, not an independently trained parameter set. Variation across masks is not automatically calibrated predictive uncertainty.', { values: [['on', 'off', 'on'], ['off', 'on', 'on'], ['w1', 'w2', 'w3']], columns: ['unit 1', 'unit 2', 'unit 3'], labels: ['mask A', 'mask B', 'shared'], selected: [[0, 0], [0, 2], [1, 1], [1, 2]], blocked: [[0, 1], [1, 0]] });
define('word-analogy-shared-displacement', 'geometry', 'Vector arithmetic creates a point; search chooses a word', 'The displacement (1,1) moves A=(0,1) to B=(1,2). Applying it to C=(2,0) creates query point (3,1); vocabulary lookup is separate.', { domain: [-1, 4], vectors: [['A', [0, 1], 'input'], ['B', [1, 2], 'state'], ['C', [2, 0], 'input'], ['Analogy query point', [3, 1], 'output']], guides: [[[0, 1], [1, 2]], [[2, 0], [3, 1]]] });

define('chatbot-randomization-unit', 'ledger', 'Keep the whole conversation in one experimental cell', 'Turn-by-turn switching mixes model effects inside one outcome. Conversation-level assignment keeps this illustrative three-turn conversation consistently on model A.', { rows: [['One conversation; three turns', ['turn 1', 'turn 2', 'turn 3'], 'input'], ['Conversation-level assignment', ['model A', 'model A', 'model A'], 'output'], ['Contaminated turn-level assignment', ['model A', 'model B', 'model A'], 'warning']] });
define('hierarchical-all-reduce-path', 'ledger', 'Combine local partials, then return the same total everywhere', 'The local sums are 1+3=4 and 2+4=6. Their global sum is 10, which every rank receives.', { rows: [['Four rank-local inputs', ['1', '3', '2', '4'], 'input'], ['Two local sums cross the hierarchy', ['4', '6'], 'state'], ['Every rank after all-reduce', ['10', '10', '10', '10'], 'output']] });
define('quality-speed-proportionate-path', 'matrix', 'A protected slice can veto an apparently successful launch', 'Ordinary checks pass while protected-record handling fails. The failed requirement remains a blocker; speed does not substitute for it.', { values: [['pass', 'pass'], ['pass', 'FAIL']], columns: ['speed', 'record rule'], labels: ['ordinary', 'protected'], selected: [[0, 0], [0, 1], [1, 0]], blocked: [[1, 1]] });
define('agent-review-control-loop', 'timeline', 'The equality case distinguishes two nearly identical patches', 'Expiry equal to now must be rejected under this contract. A patch tested only before and after the boundary can miss it.', { events: [['expiry < now: reject', -1, 'output'], ['expiry = now: must also reject', 0, 'warning'], ['expiry > now: allow', 1, 'input']], boundary: 0, unit: 'expiry minus now; illustrative clock units' });
define('ring-all-reduce-chunk-trace', 'ledger', 'Reduce a chunk, then circulate its completed sum', 'For one illustrative chunk, rank values 1,2,3,4 reduce to 10. All-gather distributes that completed chunk; it does not add the sum again.', { rows: [['Local values of one chunk', ['1', '2', '3', '4'], 'input'], ['Reduction partials along the ring', ['1', '3', '6', '10'], 'state'], ['Completed chunk distributed to ranks', ['10', '10', '10', '10'], 'output']] });
define('nms-greedy-suppression', 'suppression', 'A removed box cannot suppress its neighbor', 'Scores order A before B before C. The equal-height boxes have adjacent IoU 1/3, above the illustrative 0.30 threshold. A removes B; removed B cannot suppress C, so A and C survive.', { boxes: [{ name: 'A', left: 20, width: 160 }, { name: 'B', left: 100, width: 160 }, { name: 'C', left: 180, width: 160 }], threshold: .3 });
define('ecosystem-delayed-feedback-loop', 'timeline', 'An experiment clock can stop before the ecosystem responds', 'The creator response follows changed allocation and incentives. A short-run viewer readout does not yet measure the full supply response.', { events: [['Ranking allocation changes', 0], ['Viewer response observed', 1], ['Short-run analysis completed', 2, 'warning'], ['Creator supply response observed', 3, 'output']] });
define('agent-governed-action-path', 'connections', 'Permission and outcome evidence bracket execution', 'A proposal alone cannot dispatch a tool. Authorization must precede execution, and a receipt is needed before claiming its outcome.', { nodes: [['Proposed action', 2, 0, 'input'], ['Exact authorization', 2, 1, 'state'], ['Tool execution', 2, 2, 'focus'], ['Outcome receipt', 2, 3, 'output']], edges: [[0, 1], [1, 2], [2, 3]] });
define('ivf-pq-query-funnel', 'matrix', 'Accurate distance within the wrong partitions still misses a neighbor', 'The true neighbor in partition C stays unreachable when only A and B are scanned. PQ changes distance representation, not this membership decision.', { values: [['yes', 'yes', 'no'], ['near', 'far', 'TRUE']], columns: ['part A', 'part B', 'part C'], labels: ['scanned', 'neighbor'], selected: [[0, 0], [0, 1]], blocked: [[0, 2], [1, 2]] });
define('ml-role-activity-profiles', 'matrix', 'Shared activity units expose different fictional role budgets', 'Rows are fictional percentages, not hiring benchmarks. RS emphasizes research; MLE emphasizes engineering; the lead example emphasizes coordination.', { values: [[60, 15, 25], [10, 65, 25], [20, 20, 60]], columns: ['research', 'build', 'coordinate'], labels: ['RS', 'MLE', 'lead'], selected: [[0, 0], [1, 1], [2, 2]] });
define('asr-alignment-decoder-choices', 'ledger', 'One audio frame can emit more than one transducer token', 'A transducer token edge advances output position without advancing time. A blank advances the audio frame; these are different transitions.', { rows: [['State before emission', ['t=2', 'u=1'], 'input'], ['Emit another token', ['t=2', 'u=2'], 'state'], ['Emit blank', ['t=3', 'u=2'], 'output']] });
define('autoregressive-diffusion-sampling-steps', 'ledger', 'Append a new position, or revise the whole provisional sample', 'Autoregression extends a committed prefix. Diffusion repeatedly updates all provisional positions; its intermediate tokens are not a finished prefix.', { rows: [['Autoregressive prefix growth', ['A', 'A B', 'A B C'], 'input'], ['Diffusion provisional samples', ['? ? ?', 'a ? c', 'A B C'], 'state']] });
define('coding-assistant-two-latency-paths', 'timeline', 'A fast answer to an old document is still stale', 'The request used document version 7. A version-8 edit before its reply makes that reply inapplicable without revalidation.', { events: [['Request for version 7', 0], ['User edits to version 8', 1, 'warning'], ['Version-7 reply arrives', 2, 'warning'], ['Version-8 context revalidated', 3, 'output']] });
define('cot-signal-is-not-faithfulness', 'matrix', 'Cleaner-looking reasoning is not the same as safer behavior', 'A less alarming explanation can accompany the same prohibited action. Evaluate the action and the explanation with separate evidence.', { values: [['alarm', 'write'], ['calm', 'write']], columns: ['reasoning', 'action'], labels: ['before', 'after'], blocked: [[0, 1], [1, 1]] });
define('fused-moe-triton-sram-roofline', 'ledger', 'Keep the intermediate on chip instead of sending it out and back', 'The unfused route writes and rereads the intermediate in HBM. A fitting fused tile avoids that round trip, not the input and output traffic.', { rows: [['Unfused intermediate route', ['SRAM', 'HBM write', 'HBM read', 'SRAM'], 'warning'], ['Fused fitting intermediate route', ['SRAM', 'reuse', 'SRAM'], 'output']] });
define('product-metric-decision-authority', 'matrix', 'Two passing gates cannot outvote the failed third gate', 'Completion and cost pass, but latency exceeds its declared limit. The release condition is an AND, not a majority vote.', { values: [['4.66', '>=2', 'pass'], ['1.97', '<=2', 'pass'], ['2.1', '<=2', 'FAIL']], columns: ['observed', 'required', 'decision'], labels: ['gain', 'cost', 'latency'], selected: [[0, 2], [1, 2]], blocked: [[2, 2]] });
define('coding-interview-method-state', 'ledger', 'Look up the missing value before storing the current index', 'At value 4 and target 6, query 2. A saved 2 at index 1 returns the distinct pair (1,2).', { rows: [['Current value and needed complement', ['6', '-', '4', '=', '2'], 'input'], ['Earlier map entry', ['2', '->', 'index 1'], 'state'], ['Return original indices', ['1', '2'], 'output']] });
define('content-moderation-governed-decision-loop', 'ledger', 'A score gets meaning from the versioned policy', 'Score 0.62 lies inside this illustrative P7 review region. It is not intrinsically a permission to publish or remove content.', { rows: [['Policy P7 score regions', ['<.4', '.4-.8', '>.8'], 'input'], ['Permitted outcomes', ['allow', 'review', 'block'], 'state'], ['This score', ['.62', '->', 'review'], 'output']] });
define('paper-critique-prioritization-funnel', 'ledger', 'An accuracy gain and its compute cost are different units', 'The reported four-point gain used twice the compute. That comparison alone does not isolate a method benefit at matched resources.', { rows: [['Baseline conditions', ['accuracy A', 'compute 1x'], 'input'], ['Reported new conditions', ['A +4 points', 'compute 2x'], 'state'], ['Missing matched comparison', ['baseline at 2x compute'], 'warning']] });
define('cross-validation-preserve-structure', 'matrix', 'Choose the split that matches the deployment claim', 'Random rows can leak user identity or future information. Group and time boundaries protect different dependencies; neither is interchangeable with random splitting.', { values: [['A', 'A', 'A'], ['train', 'test', 'train'], ['train', 'train', 'train']], columns: ['row 1', 'row 2', 'row 3'], labels: ['one user', 'random', 'group'], selected: [[2, 0], [2, 1], [2, 2]], blocked: [[1, 1]] });
define('dbscan-neighborhood-expansion', 'bars', 'A border point can join without becoming an expansion source', 'With minPts=4 including self, counts are D=5, E=2, and F=1. Only the core neighborhood can expand.', { rows: [['D: core', 5, 'output'], ['E: border reachable from D', 2, 'input'], ['F: isolated noise', 1, 'warning']], limit: 4, limitLabel: 'minPts', unit: ' points' });
define('frontier-training-divergence-zoom', 'matrix', 'Locate the first failing rank and step before blaming later symptoms', 'The first nonfinite value appears at rank 1, step 2. The later failures across all ranks do not identify their original cause.', { values: [['ok', 'ok', 'bad'], ['ok', 'FIRST', 'bad'], ['ok', 'ok', 'bad']], columns: ['step 1', 'step 2', 'step 3'], labels: ['rank 0', 'rank 1', 'rank 2'], selected: [[1, 1]], blocked: [[0, 2], [1, 2], [2, 2]] });
define('model-not-learning-diagnostic-path', 'matrix', 'A gradient can exist while the intended parameters stay unchanged', 'The same tiny batch has nonzero gradient but zero weight change. Diagnose the update path before full-data generalization.', { values: [['0.8', 'yes'], ['0', 'NO'], ['1.10', 'flat']], columns: ['observed', 'signal'], labels: ['grad', 'delta W', 'loss'], selected: [[0, 1]], blocked: [[1, 1], [2, 1]] });
define('offline-online-gap-diagnostic-order', 'population', 'Assigned users still count when the treatment falls back', 'This illustrative cell assigns 100 users but serves the new model to only 60. Dropping 40 fallback users changes the estimand and can bias the comparison.', { rows: [['served assigned treatment', 60, 'output'], ['assigned but fell back', 40, 'warning']], unit: 'one assigned user' });
define('training-loop-persistent-state', 'matrix', 'Clearing gradients does not restore training mode', 'A training iteration needs both the intended model mode and a clean gradient buffer. Each action changes a separate persistent state.', { values: [['train', 'old'], ['eval', 'old'], ['eval', 'zero'], ['train', 'zero']], columns: ['mode', 'gradient'], labels: ['before', 'eval()', 'zero()', 'train()'], selected: [[2, 1], [3, 0], [3, 1]], blocked: [[2, 0]] });
define('work-prioritization-counterfactual-portfolio', 'bars', 'The alternative portfolio is the opportunity cost', 'In this illustrative fixed-capacity choice, A yields 8 while compatible B and C yield 11. Past effort on A does not change the forward comparison.', { rows: [['Choose A alone', 8, 'input'], ['Choose B plus C', 11, 'output']], unit: ' assumed value units' });
define('decision-tree-partitions-feature-space', 'matrix', 'The second split affects only one root branch', 'The root sends every x<=0 point to A. Splitting y on the right branch changes only the x>0 region.', { values: [['A', 'C'], ['A', 'B']], columns: ['x<=0', 'x>0'], labels: ['y>1', 'y<=1'], selected: [[0, 1], [1, 1]] });
define('ethical-decision-process-outcome', 'timeline', 'Do not let later luck rewrite what was known at decision time', 'The original choice used the evidence available before its boundary. A favorable later outcome does not prove that the original process was sound.', { events: [['Evidence available to decision maker', 0, 'input'], ['Decision committed', 1, 'state'], ['Unexpected outcome observed', 2, 'warning']], boundary: 1 });
define('policy-label-observation-loop', 'ledger', 'Missing outcomes are not negative outcomes', 'Only observed mature outcomes supply this label. Pending, censored, and unobserved cases retain their distinct missingness states.', { labelPadding: 6, rows: [['Outcome states', ['mature +', 'mature -', 'pending', 'censored'], 'input'], ['Training labels', ['1', '0', '?', '?'], 'output']] });
define('ablation-rival-hypotheses', 'bars', 'A matched control separates the component from the extra resources', 'The baseline scores 80; the resource-matched control scores 82.8; the claimed method scores 83. The isolated matched gap is 0.2 points, not 3.', { rows: [['Original baseline', 80, 'neutral'], ['Resource-matched control', 82.8, 'input'], ['Claimed method', 83, 'output']], unit: ' score points' });
define('coding-product-authority-boundary', 'matrix', 'Shared context does not imply shared execution authority', 'Inline, chat, and review can share repository context. Only an authorized isolated agent may use the declared edit and execution tools.', { values: [['yes', 'no'], ['yes', 'no'], ['yes', 'no'], ['yes', 'scoped']], columns: ['context', 'execute'], labels: ['inline', 'chat', 'review', 'agent'], selected: [[3, 1]], blocked: [[0, 1], [1, 1], [2, 1]] });
define('enterprise-agent-three-plane-boundary', 'timeline', 'Persist authority before dispatch; reconcile a lost reply', 'The payment commits once under an idempotency key. A lost reply requires reconciliation, not a second payment.', { events: [['Exact payment authorization persisted', 0, 'input'], ['Idempotent dispatch', 1, 'state'], ['Account commits; reply lost', 2, 'warning'], ['Same key returns original receipt', 3, 'output']] });
define('distributed-recovery-consistency-gate', 'matrix', 'Matching model weights alone is not a consistent restart', 'Rank 1 has the right model version but stale optimizer state. Recovery waits until model, optimizer, and data position agree.', { values: [['v4', 'v4', 'p8'], ['v4', 'v3', 'p8']], columns: ['model', 'optimizer', 'data'], labels: ['rank 0', 'rank 1'], selected: [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2]], blocked: [[1, 1]] });
define('feature-store-two-time-contract', 'timeline', 'Historical lookup needs availability time as well as event time', 'Version B has an 11:50 event time but arrives at 12:10. The noon lookup must return available version A, not B.', { events: [['Version A available', 11 + 5 / 60, 'output'], ['Version B event occurs', 11 + 50 / 60, 'input'], ['Historical noon query', 12, 'warning'], ['Version B becomes available', 12 + 10 / 60, 'input']], boundary: 12, unit: 'hour on one shared clock' });
define('foundation-data-deletion-blast-radius', 'matrix', 'Deleting source membership does not prove deletion of model influence', 'A new shard can exclude source D7. Existing checkpoint and model influence require their own intervention and validation.', { values: [['excluded', 'yes'], ['unknown', '?'], ['unknown', '?']], columns: ['D7 state', 'verified'], labels: ['new data', 'old ckpt', 'released'], selected: [[0, 1]], blocked: [[1, 1], [2, 1]] });
define('fraud-policy-label-coverage', 'matrix', 'The policy changes which payment outcomes become observable', 'A declined payment has no observed repayment outcome. Pending and declined cases cannot be silently labeled as legitimate negatives.', { values: [['yes', '0/1'], ['not yet', '?'], ['no', '?']], columns: ['outcome', 'label'], labels: ['approved', 'pending', 'declined'], selected: [[0, 1]], blocked: [[1, 1], [2, 1]] });
define('adaptive-red-team-release-loop', 'matrix', 'Fixing one reproduced attack leaves other paths unknown', 'The direct request is now blocked. Quoted and tool-output paths remain untested, not passed.', { values: [['leak', 'blocked'], ['?', '?'], ['?', '?']], columns: ['before', 'after'], labels: ['direct', 'quote', 'tool'], selected: [[0, 1]], blocked: [[1, 1], [2, 1]] });
define('ml-ab-test-launch-gates', 'matrix', 'Significance, practical value, and safety are separate conditions', 'One candidate has too little practical gain; another fails safety. Neither passes all launch gates.', { values: [['yes', 'no', 'yes'], ['yes', 'yes', 'no']], columns: ['nonzero', 'worth it', 'safe'], labels: ['case A', 'case B'], selected: [[0, 0], [0, 2], [1, 0], [1, 1]], blocked: [[0, 1], [1, 2]] });
define('ml-monitoring-observability-layers', 'timeline', 'A freshness violation can be known before quality labels mature', 'A stale snapshot breaches its current contract immediately. Quality inference still waits for mature outcomes and valid joins.', { events: [['Snapshot becomes stale', 0, 'warning'], ['Request served; freshness failure known', 1, 'warning'], ['Outcome maturation window', 2, 'state'], ['Quality labels can be evaluated', 3, 'output']] });
define('multi-team-platform-contract-boundary', 'matrix', 'Only the first valid compare-and-swap owns the result', 'A commits version 0 to 1. B still expects version 0, so its later write is rejected rather than overwriting A.', { values: [['0', '0', 'v1'], ['0', '1', 'REJECT']], columns: ['expected', 'actual', 'result'], labels: ['attempt A', 'attempt B'], selected: [[0, 2]], blocked: [[1, 2]] });
define('post-training-policy-gate', 'matrix', 'A successful task can still fail the authorization requirement', 'Completion and policy compliance are independent checks. Unauthorized account access vetoes the otherwise successful trial.', { values: [['yes', 'yes', 'pass'], ['yes', 'NO', 'FAIL']], columns: ['task', 'authority', 'trial'], labels: ['allowed', 'violation'], selected: [[0, 2]], blocked: [[1, 1], [1, 2]] });
define('llm-inference-two-budget-scheduler', 'ledger', 'Persistent pages and refreshing work slots are different ledgers', 'A and B reserve nine of twelve KV pages. The eight token-work slots refresh each iteration; free slots do not erase persistent page reservations.', { emptySlots: true, rows: [['Persistent KV pool; twelve pages', ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', '-', '-', '-'], 'input'], ['One iteration; eight work slots', ['D', 'D', 'D', 'P', 'P', 'P', 'P', 'P'], 'state']] });
define('reasoning-fixed-budget-reclaim', 'ledger', 'Quarterly hours do not describe the nodes free right now', 'Six serving nodes and two offline nodes fill this instantaneous eight-node pool. Serving pressure reclaims the two offline nodes; protected reserves remain protected.', { rows: [['Before serving pressure', ['S', 'S', 'S', 'S', 'S', 'S', 'O', 'O'], 'input'], ['After checkpoint and offline pause', ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'], 'output']] });
define('ecosystem-experiment-interference', 'connections', 'Treatment can change supply that later reaches control', 'Both cells share one creator pool. Changed supply can return to both cells, breaking a simple no-interference interpretation.', { nodes: [['Treatment allocation', 0, 0, 'input'], ['Control allocation', 1, 0, 'state'], ['Shared creator supply', 2, 1, 'focus'], ['Future treatment', 0, 2, 'input'], ['Future control', 1, 2, 'warning']], edges: [[0, 2], [1, 2], [2, 3], [2, 4]] });
define('spotify-two-level-shelf-composition', 'bars', 'A lower-scoring shelf can improve the whole page', 'Shelf scores are A=6, B=5, C=4. A+B repeats a song and pays penalty 4; A+C wins at 10.', { rows: [['A + B - duplicate penalty', 7, 'warning'], ['A + C', 10, 'output'], ['B + C', 9, 'input']], unit: ' toy utility' });
define('youtube-recommender-selective-feedback-loop', 'population', 'Only displayed candidates have exposure-linked feedback', 'Of 900 eligible candidates, only ten are displayed. The other 890 are unshown, not negative response observations.', { rows: [['displayed candidates', 10, 'output'], ['eligible but unshown', 890, 'neutral']], unit: 'one eligible candidate' });
define('rag-recall-precision-funnel', 'matrix', 'Reranking cannot recover a passage removed upstream', 'Both required evidence passages must survive retrieval and filtering. A relevant passage dropped before reranking cannot appear in the final answer context.', { values: [['yes', 'yes'], ['yes', 'NO'], ['yes', 'absent']], columns: ['evidence A', 'evidence B'], labels: ['retrieved', 'after cap', 'rerank'], selected: [[0, 0], [0, 1], [1, 0], [2, 0]], blocked: [[1, 1], [2, 1]] });

define('encoder-once-decoder-loop', 'matrix', 'Reuse source memory while the generated prefix grows', 'The encoder source key/value memory stays fixed. Decoder prefixes and their own caches grow across generation steps.', { values: [['fixed', 'BOS'], ['fixed', 'BOS A'], ['fixed', 'BOS A B']], columns: ['source KV', 'prefix'], labels: ['step 1', 'step 2', 'step 3'], selected: [[0, 0], [1, 0], [2, 0]] });
define('coding-assistant-evidence-ladder', 'matrix', 'Passing tests does not establish sustained user value', 'A test receipt establishes only its tested contract. Acceptance, continued use, and durable value require separate observations.', { values: [['yes', '?', '?', '?']], columns: ['tests', 'accept', 'reuse', 'value'], labels: ['evidence'], selected: [[0, 0]], blocked: [[0, 1], [0, 2], [0, 3]] });
define('agent-eval-diagnostic-flywheel', 'matrix', 'A passed test and an unauthorized write can coexist', 'One trial can complete its task while violating the authority boundary. The combined result must fail the declared policy gate.', { values: [['pass', 'NO', 'FAIL']], columns: ['task test', 'authorized', 'joint'], labels: ['trial'], selected: [[0, 0]], blocked: [[0, 1], [0, 2]] });
define('search-ranker-evidence-flywheel', 'matrix', 'A better weighted average can conceal a failing tail', 'At head/tail weights 90%/10%, B averages 0.88 versus A at 0.80. Its tail score 0.70 misses the 0.78 floor.', { values: [[.8, .8, .8], [.9, .7, .88]], columns: ['head', 'tail', 'weighted'], labels: ['A', 'B'], selected: [[1, 0], [1, 2]], blocked: [[1, 1]] });
define('independent-confirmation-vs-adaptive-reuse', 'timeline', 'A benchmark becomes development evidence when it guides selection', 'The first feedback influences the chosen model. Independent confirmation requires new evidence after that selection is locked.', { events: [['Benchmark feedback observed', 0, 'input'], ['Model chosen using that feedback', 1, 'warning'], ['Model selection locked', 2, 'state'], ['Fresh confirmation evidence collected', 3, 'output']], boundary: 2 });
define('collective-failure-commit-boundary', 'matrix', 'Retry from a common committed checkpoint, not mixed local progress', 'Rank 1 failed after rank 0 advanced. Both must restore the committed checkpoint before retrying the collective.', { values: [['v5', 'failed'], ['v4', 'v4'], ['retry', 'retry']], columns: ['rank 0', 'rank 1'], labels: ['failure', 'restore', 'resume'], selected: [[1, 0], [1, 1], [2, 0], [2, 1]], blocked: [[0, 0], [0, 1]] });
define('llm-adaptation-layers', 'matrix', 'Correct formatting is not correct policy execution', 'A response can satisfy the JSON schema while citing stale context or violating policy. Format, knowledge, and behavior need separate checks.', { values: [['pass', 'stale', 'FAIL']], columns: ['schema', 'knowledge', 'policy'], labels: ['response'], selected: [[0, 0]], blocked: [[0, 1], [0, 2]] });
define('fine-tuning-two-decisions', 'matrix', 'Training signal and update scope form two independent axes', 'Supervised or preference signals can update all parameters or a parameter-efficient subset. Choosing one axis does not choose the other.', { values: [['SFT full', 'SFT LoRA'], ['pref full', 'pref LoRA']], columns: ['full scope', 'low-rank'], labels: ['labels', 'pairs'], selected: [[0, 1], [1, 0]] });
define('as-five-cross-cutting-signals', 'matrix', 'One follow-up can reveal several skills and leave others unknown', 'A specific follow-up yields evidence about problem framing and experimentation here. Unknown signals remain unknown rather than being inferred from confidence.', { values: [['seen', 'seen', '?', '?', '?']], columns: ['frame', 'test', 'scale', 'lead', 'impact'], labels: ['probe'], selected: [[0, 0], [0, 1]], blocked: [[0, 2], [0, 3], [0, 4]] });
define('flashattention-sram-tile-stream', 'ledger', 'Keep the running statistics, discard temporary score tiles', 'Each key tile updates the row maximum, normalizer, and weighted-value accumulator. The temporary score tile need not survive in HBM.', { rows: [['Tile 1 contributes', ['m1', 'l1', 'o1'], 'input'], ['Merge tile 2 with stable rescaling', ['m12', 'l12', 'o12'], 'state'], ['Only final normalized result persists', ['o12 / l12'], 'output']] });
define('flashattention-standard-hbm-path', 'ledger', 'Remove quadratic intermediate round trips, not all memory movement', 'The standard route writes and reads the score matrix. Tiled attention still reads inputs and writes outputs while keeping temporary scores on chip.', { rows: [['Standard score-matrix traffic', ['HBM write', 'N x N', 'HBM read'], 'warning'], ['Tiled temporary score traffic', ['SRAM tile', 'merge', 'discard'], 'output'], ['Still required', ['read QKV', 'write output'], 'input']] });
define('foundation-data-curation-lineage', 'matrix', 'Manifest membership and actual consumption need different receipts', 'R17 belongs to the accepted shard and has a training-read receipt. R18 is rejected; its reason remains recorded without membership or consumption.', { values: [['yes', 'yes', 'yes'], ['denied', 'no', 'no']], columns: ['policy', 'manifest', 'read log'], labels: ['R17', 'R18'], selected: [[0, 0], [0, 1], [0, 2]], blocked: [[1, 0], [1, 1], [1, 2]] });
define('interview-evidence-to-drill', 'matrix', 'Known constraints select a drill; unknown permission requires a question', 'An established time budget can determine exercise size. Unknown tool permission is not an implicit authorization to use tools.', { values: [['known', 'choose'], ['unknown', 'ASK']], columns: ['evidence', 'next step'], labels: ['time', 'tools'], selected: [[0, 1]], blocked: [[1, 1]] });
define('proof-of-work-three-checks', 'matrix', 'Patch authorship is not ownership of the team rollout', 'A personal patch proves one contribution. Decision ownership, rollout responsibility, and outcome attribution require their own evidence.', { values: [['yes', '?', '?']], columns: ['patch', 'decision', 'rollout'], labels: ['receipt'], selected: [[0, 0]], blocked: [[0, 1], [0, 2]] });
define('fsdp-wrapped-unit-lifecycle', 'ledger', 'Sharded resting storage still gathers the current unit temporarily', 'Four ranks each retain one shard at rest. The active wrapped unit gathers for computation, then returns to sharded storage.', { rows: [['At rest on each rank', ['1/4', '1/4', '1/4', '1/4'], 'input'], ['Active unit temporarily on each rank', ['full', 'full', 'full', 'full'], 'warning'], ['After unit computation', ['1/4', '1/4', '1/4', '1/4'], 'output']] });
define('gan-alternating-update-ownership', 'matrix', 'Frozen discriminator weights can still transmit generator gradients', 'The discriminator step stops the fake-sample gradient into G. The generator step freezes D parameters but preserves the path through D.', { values: [['no', 'yes', 'STOP G'], ['yes', 'no', 'pass G']], columns: ['update G', 'update D', 'gradient'], labels: ['D step', 'G step'], selected: [[0, 1], [1, 0], [1, 2]], blocked: [[0, 2]] });
define('hallucination-claim-release-gates', 'matrix', 'A supported claim can still be unreleasable', 'Evidence supports the claim, but release permission is a separate condition. Passing factual support does not authorize disclosure.', { values: [['yes', 'yes'], ['yes', 'NO']], columns: ['support', 'release'], labels: ['public', 'private'], selected: [[0, 0], [0, 1], [1, 0]], blocked: [[1, 1]] });
define('llm-eval-calibration-loop', 'population', 'Thirteen joint passes do not clear a fourteen-pass floor', 'Thirteen of twenty trials pass both requirements, or 65%. A 70% floor requires at least fourteen joint passes.', { rows: [['joint passes', 13, 'output'], ['joint failures', 7, 'warning']], unit: 'one evaluation trial' });
define('asr-adaptation-boundaries', 'matrix', 'A lexicon update need not retrain the acoustic model', 'The modular example keeps the acoustic core fixed while changing a separate lexicon. End-to-end systems may place adaptation behind different boundaries.', { values: [['fixed', 'changed'], ['fixed', 'changed']], columns: ['acoustic', 'lexicon'], labels: ['before', 'after'], selected: [[1, 1]] });
define('model-gain-evidence-decision', 'bars', 'A nonzero estimate can remain below the worthwhile threshold', 'A statistically positive half-point gain can still miss a one-point practical threshold. The two questions are distinct.', { rows: [['Illustrative gain estimate', .5, 'input']], limit: 1, limitLabel: 'worthwhile', unit: ' points' });
define('beam-search-live-finished-transition', 'matrix', 'Length normalization can reverse the raw-probability winner', 'Raw probability favors A EOS; log-probability divided by length favors A x EOS. Count EOS, exclude BOS, and stop expanding finished beams.', { values: [[.18, 2, Number((Math.log(.18) / 2).toFixed(4))], [.15, 3, Number((Math.log(.15) / 3).toFixed(4))]], columns: ['prob.', 'length', 'final'], labels: ['A EOS', 'Ax EOS'], selected: [[0, 0], [1, 2]] });
define('knn-index-decision-funnel', 'matrix', 'Scanning fewer vectors and storing fewer bytes are different savings', 'IVF changes which partitions are scanned. PQ compresses their representations; its approximation cannot recover a neighbor excluded by IVF.', { values: [['fewer', 'same'], ['same', 'fewer'], ['fewer', 'fewer']], columns: ['vectors read', 'bytes each'], labels: ['IVF', 'PQ', 'both'], selected: [[0, 0], [1, 1], [2, 0], [2, 1]] });
define('black-box-discriminating-probes', 'matrix', 'Choose an input where rival hypotheses disagree', 'A shared prediction cannot distinguish the two hypotheses. A predicted output switch is informative, but still does not prove the hidden mechanism uniquely.', { values: [['same', 'same'], ['A', 'B']], columns: ['hypothesis 1', 'hypothesis 2'], labels: ['weak probe', 'strong'], selected: [[1, 0], [1, 1]] });
define('project-stop-forward-evidence-ledger', 'bars', 'Past spending does not enter the forward-value comparison', 'The illustrative continue option has expected future value 3 and remaining cost 5. Sunk cost 20 cannot turn its net -2 into a benefit.', { rows: [['Continue: future value minus future cost', -2, 'warning'], ['Stop: additional net value', 0, 'output']], domain: [-3, 1], unit: ' future value units' });
define('senior-principal-delegated-coherence', 'matrix', 'Local ownership ends where a shared interface changes', 'An internal implementation choice can remain delegated. Changing a shared interface requires alignment with affected owners rather than unilateral local approval.', { values: [['local', 'owner'], ['shared', 'align']], columns: ['scope', 'decision'], labels: ['internal', 'interface'], selected: [[0, 1], [1, 1]] });
define('reactive-pretraining-evidence-loop', 'connections', 'A bounded probe branches from a recoverable mainline', 'Keep the hot mainline recoverable while testing the response. A promising probe needs comparative evidence before replacing that mainline.', { nodes: [['Committed mainline', 2, 0, 'input'], ['Recoverable continuation', 0, 1, 'input'], ['Bounded probe branch', 1, 1, 'state'], ['Evidence-based promotion', 1, 2, 'output']], edges: [[0, 1], [0, 2], [2, 3]] });
define('judge-calibration-gates', 'matrix', 'A hard failure vetoes quality; order sensitivity requires review', 'A high quality score cannot override a hard-rule failure. Reversing answer order and changing the judgment exposes a separate calibration problem.', { values: [['.9', 'FAIL', 'hold'], ['.9', 'pass', 'review']], columns: ['quality', 'hard rule', 'decision'], labels: ['violation', 'order flip'], blocked: [[0, 1], [0, 2], [1, 2]] });
define('healthcare-llm-clinician-gate', 'matrix', 'Matching the patient does not match the encounter or approved draft', 'The patient ID agrees, but the encounter and draft versions differ. A clinician must approve the exact current bundle before release.', { values: [['P7', 'E4', 'D2'], ['P7', 'E5', 'D3']], columns: ['patient', 'encounter', 'draft'], labels: ['approved', 'current'], selected: [[0, 0], [1, 0]], blocked: [[1, 1], [1, 2]] });
define('llm-eval-chain-of-trust', 'matrix', 'A citation proxy can improve while confidentiality fails', 'More citations do not establish authorized account handling. Independent policy evidence vetoes an apparently improved proxy score.', { values: [[2, 'yes'], [5, 'NO']], columns: ['citations', 'authorized'], labels: ['baseline', 'new'], selected: [[1, 0]], blocked: [[1, 1]] });
define('llm-cost-compounds-across-agent-calls', 'ledger', 'Caching discounts repeated input, not new output at every call', 'Each agent call can generate billable output. Repeated cached context changes one cost component, not the whole multi-call total.', { rows: [['Three calls with repeated context', ['cached in', 'cached in', 'cached in'], 'input'], ['Fresh generated output per call', ['new out 1', 'new out 2', 'new out 3'], 'warning'], ['Total cost includes', ['all input', '+', 'all output'], 'output']] });
define('llm-security-influence-authority-path', 'connections', 'Untrusted content can suggest an action without authorizing it', 'Retrieved text may contain an export instruction. That instruction cannot cross the authority boundary into an executable export.', { nodes: [['Untrusted document text', 2, 0, 'warning'], ['Suggested export action', 2, 1, 'warning'], ['Authorized tool dispatch', 2, 2, 'output']], edges: [[0, 1], [1, 2, true]] });
define('long-context-three-bottlenecks', 'matrix', 'Paging removes allocation waste, not context or attention growth', 'Logical KV length and attention reads still grow with context. Paging targets physical allocation waste; it does not eliminate the other two dimensions.', { values: [['reduced', 'grows', 'grows']], columns: ['waste', 'KV length', 'attention'], labels: ['paging'], selected: [[0, 0]], blocked: [[0, 1], [0, 2]] });
define('loss-spike-trace-taxonomy', 'matrix', 'An evaluation-only spike does not identify a training failure', 'Training loss remains stable while evaluation loss jumps. Investigate evaluation mode, data, and measurement before claiming an optimizer instability.', { values: [['stable', 'stable'], ['stable', 'SPIKE']], columns: ['train', 'eval'], labels: ['before', 'after'], selected: [[1, 0]], blocked: [[1, 1]] });
define('activation-patching-controlled-contrast', 'matrix', 'Change one activation while keeping the corrupted run otherwise fixed', 'The patched run restores activation A from the clean run. Other corrupted activations stay fixed, isolating this intervention within the declared setup.', { values: [['clean', 'clean'], ['corrupt', 'corrupt'], ['clean', 'corrupt']], columns: ['activation A', 'other state'], labels: ['clean', 'corrupt', 'patched'], selected: [[2, 0]], blocked: [[2, 1]] });
define('microannealing-controlled-fork', 'matrix', 'A domain gain cannot waive a preservation failure', 'The probe improves domain loss but fails the declared preservation check. Promotion requires both the gain and preservation conditions.', { values: [['better', 'FAIL', 'hold']], columns: ['domain', 'preserve', 'promote'], labels: ['probe'], selected: [[0, 0]], blocked: [[0, 1], [0, 2]] });
define('fp16-safe-update-order', 'timeline', 'Unscale before clipping; reject nonfinite updates before stepping', 'Clipping a scaled gradient uses the wrong effective threshold. A nonfinite result skips the parameter update instead of corrupting the model.', { events: [['Scaled backward completes', 0, 'input'], ['Unscale and check finite', 1, 'state'], ['Clip the unscaled gradient', 2, 'focus'], ['Step only when finite', 3, 'output']] });
define('moe-top-two-routing-accounting', 'matrix', 'Two active experts do not mean only two stored experts', 'This token executes E1 and E3. E2 and E4 remain stored even though they perform no work for this token.', { values: [['yes', 'yes', 'yes', 'yes'], ['yes', 'no', 'yes', 'no']], columns: ['E1', 'E2', 'E3', 'E4'], labels: ['stored', 'executes'], selected: [[1, 0], [1, 2]], blocked: [[1, 1], [1, 3]] });
define('ml-lineage-two-way-incident-trace', 'matrix', 'Lineage identifies possible exposure; evidence establishes actual impact', 'A checkpoint descends from the affected shard. A consumption receipt and an outcome check are still needed before claiming actual model harm.', { values: [['yes', '?', '?']], columns: ['descendant', 'consumed', 'harmed'], labels: ['model M'], selected: [[0, 0]], blocked: [[0, 1], [0, 2]] });
define('model-organism-policy-test', 'matrix', 'Suppressing one known trigger leaves generalization unknown', 'The reproduced trigger no longer activates the behavior. Untested paraphrases and contexts remain unknown rather than certified safe.', { values: [['yes', 'no'], ['?', '?']], columns: ['before', 'after'], labels: ['known', 'new'], selected: [[0, 1]], blocked: [[1, 0], [1, 1]] });
define('ambitious-project-three-evidence-test', 'matrix', 'Separate the decision owned from the outcome shared', 'A personal design decision is one piece of evidence. Execution contributions and team-level outcomes need distinct attribution rather than being claimed as one personal result.', { values: [['owned', 'shared', 'team']], columns: ['decision', 'execution', 'outcome'], labels: ['project'], selected: [[0, 0]] });
define('overrated-technique-claim-stress-test', 'matrix', 'A bounded critique declares the result that would overturn it', 'The claim concerns a specific task and matched budget. A contrary matched result would update it; a different task alone would not test the same claim.', { values: [['same', 'same', 'compare'], ['other', 'higher', 'unclear']], columns: ['task', 'budget', 'test'], labels: ['matched', 'unmatched'], selected: [[0, 2]], blocked: [[1, 2]] });
define('multi-agent-ctde-information-boundary', 'matrix', 'Training can see joint information that execution cannot', 'The centralized critic uses joint state during training. Each deployed actor must act from its allowed local observation without that critic.', { values: [['joint', 'local'], ['absent', 'local']], columns: ['critic', 'actor'], labels: ['train', 'execute'], selected: [[0, 0], [0, 1], [1, 1]], blocked: [[1, 0]] });
define('multimodal-architectures-meeting-point', 'matrix', 'The meeting point determines when modalities can interact', 'Early fusion mixes modalities before shared processing. Late fusion keeps representations separate until their scores or decisions meet.', { values: [['joint', 'joint'], ['image', 'text'], ['merge', 'merge']], columns: ['image path', 'text path'], labels: ['early', 'late pre', 'late end'], selected: [[0, 0], [0, 1], [2, 0], [2, 1]] });
define('negative-hardness-label-gate', 'matrix', 'A similar relevant item is not a valid hard negative', 'High similarity establishes hardness, not irrelevance. The label gate must reject a relevant answer before negative training uses it.', { values: [['high', 'yes', 'NO'], ['high', 'no', 'yes']], columns: ['sim. score', 'relevant', 'negative'], labels: ['answer', 'distractor'], selected: [[1, 2]], blocked: [[0, 2]] });
define('training-recipe-evidence-gates', 'matrix', 'A nonzero gradient does not prove that the intended weights changed', 'The recipe produces gradient 0.8 but observed parameter change zero. Check the actual optimizer-bound parameters and step receipt.', { values: [[.8, 0, 'FAIL']], columns: ['gradient', 'delta W', 'update'], labels: ['probe'], selected: [[0, 0]], blocked: [[0, 1], [0, 2]] });
define('detector-candidate-resolution', 'matrix', 'Training assignment and inference suppression solve different problems', 'Training matches predictions to ground-truth supervision. Inference removes redundant predicted boxes without access to ground truth.', { values: [['yes', 'match'], ['no', 'suppress']], columns: ['truth used', 'operation'], labels: ['training', 'inference'], selected: [[0, 1], [1, 1]] });

define('accelerator-critical-path-trace', 'bars', 'Less work saves time only when it shortens the critical path', 'Parallel tasks A=100 and B=60 finish together after 100 units. Halving B still takes 100; reducing A to 80 lowers the finish time.', { rows: [['Baseline: max(100,60)', 100, 'input'], ['Optimize B only: max(100,30)', 100, 'warning'], ['Optimize A: max(80,60)', 80, 'output']], unit: ' toy time units' });
define('paged-attention-block-table-indirection', 'ledger', 'Logical order survives scattered physical placement', 'Logical blocks 0,1,2 map to physical pages 7,2,9. The block table, not physical adjacency, reconstructs the token sequence.', { rows: [['Logical sequence block order', ['0', '1', '2'], 'input'], ['Block table: physical pages', ['7', '2', '9'], 'state'], ['Attention follows logical order', ['page 7', 'page 2', 'page 9'], 'output']] });
define('people-also-bought-three-relations', 'matrix', 'Co-purchase evidence does not establish product compatibility', 'A frequently co-purchased case can still fail the phone fit requirement. Complementarity needs a separate compatibility relation.', { values: [['high', 'NO', 'exclude'], ['lower', 'yes', 'eligible']], columns: ['co-buy', 'fits', 'candidate'], labels: ['case A', 'case B'], selected: [[1, 1], [1, 2]], blocked: [[0, 1], [0, 2]] });
define('personalized-search-request-learning-loop', 'matrix', 'A later ranker cannot recover a missing upstream candidate', 'The relevant candidate survives retrieval but fails an earlier cap. The final ranker can compare only the candidates it receives.', { values: [['yes', 'yes'], ['yes', 'NO'], ['yes', 'absent']], columns: ['candidate A', 'candidate B'], labels: ['retrieve', 'cap', 'rank'], selected: [[0, 0], [0, 1], [1, 0], [2, 0]], blocked: [[1, 1], [2, 1]] });
define('pipeline-fill-drain-bubble', 'matrix', 'Idle stage-slots are unused capacity, not extra elapsed time', 'Three microbatches cross three stages in five time slots. Six idle stage-slots occur within that same elapsed schedule.', { values: [['A', 'B', 'C', '-', '-'], ['-', 'A', 'B', 'C', '-'], ['-', '-', 'A', 'B', 'C']], columns: ['t1', 't2', 't3', 't4', 't5'], labels: ['stage 1', 'stage 2', 'stage 3'], selected: [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [2, 4]] });
define('plan-70b-parallel-accounting', 'ledger', 'Model shards do not multiply distinct training examples', 'Two data replicas, sixteen examples per microbatch, and four accumulation steps produce 128 distinct examples. Tensor and pipeline shards share those examples.', { rows: [['Distinct-example factors: DP, microbatch, accumulation', ['2', 'x16', 'x4', '=128'], 'input'], ['TP=8 and PP=2 partition each replica', ['8x2', 'shards', 'same examples'], 'state']] });
define('reward-model-distribution-shift', 'matrix', 'A tested reward proxy can become untested after optimization', 'The proxy was validated on the original response distribution. Optimization shifts responses, so original accuracy does not certify the new region.', { values: [['tested', 'known'], ['shifted', '?']], columns: ['responses', 'proxy accuracy'], labels: ['before', 'after'], selected: [[0, 1]], blocked: [[1, 1]] });
define('prefill-decode-weight-reuse', 'ledger', 'One weight load can serve more than one token computation', 'Prefill or batching can reuse loaded weights across tokens. Decode still performs token-specific computation and reads its growing cache.', { rows: [['One loaded weight tile', ['W'], 'input'], ['Several token computations reuse W', ['token 1', 'token 2', 'token 3', 'token 4'], 'state'], ['Not removed by weight reuse', ['token math', 'KV reads'], 'output']] });
define('technical-project-decision-spine', 'bars', 'The awaited dependency can dominate a faster model', 'A response waits for parallel dependencies A=100 and model B=60. Halving model time leaves the awaited 100-unit dependency unchanged.', { rows: [['End-to-end before: max(100,60)', 100, 'input'], ['End-to-end after: max(100,30)', 100, 'warning']], unit: ' toy time units' });
define('profiling-exposed-collective-tail', 'bars', 'Only communication beyond useful overlap extends the step', 'A 40-unit collective overlaps 30 units of useful work. Its exposed tail is 10, not the full 40.', { rows: [['Full collective duration', 40, 'input'], ['Covered by useful computation', 30, 'state'], ['Exposed tail', 10, 'warning']], unit: ' toy time units' });
define('legal-rag-claim-lineage', 'matrix', 'A resolvable citation can still fail to support its claim', 'Both citations resolve to documents. Only citation A supports the exact claim; citation B must not pass merely because its link works.', { values: [['yes', 'yes'], ['yes', 'NO']], columns: ['resolves', 'supports'], labels: ['cite A', 'cite B'], selected: [[0, 0], [0, 1], [1, 0]], blocked: [[1, 1]] });
define('rag-index-request-paths', 'ledger', 'Keep the answer connected to one versioned source chunk', 'A stable chunk identity and version connect the retrieved evidence to its source. A citation string alone is not that lineage receipt.', { rows: [['Versioned source and indexed chunk', ['doc D v3', 'chunk C7'], 'input'], ['Request retrieves that same chunk', ['C7', 'v3'], 'state'], ['Answer evidence record', ['claim', 'C7 v3'], 'output']] });
define('personalization-three-feature-clocks', 'timeline', 'A fast request can still use an old feature snapshot', 'The request completes quickly after reading a much earlier snapshot. Request latency and feature freshness use different timestamp differences.', { events: [['Feature snapshot materialized', 0, 'warning'], ['Request starts', 9.9, 'input'], ['Response returns', 10, 'output']], unit: 'shared illustrative clock units' });
define('recsys-llm-bounded-augmentation', 'matrix', 'A generated recommendation must satisfy deadline and catalog gates', 'A timely suggestion for an unknown item fails the catalog gate. A valid but late suggestion also falls back to the bounded baseline.', { values: [['yes', 'NO', 'fallback'], ['NO', 'yes', 'fallback'], ['yes', 'yes', 'eligible']], columns: ['on time', 'catalog', 'result'], labels: ['case A', 'case B', 'case C'], selected: [[2, 2]], blocked: [[0, 1], [1, 0]] });
define('inference-cost-measure-test-compound-loop', 'bars', 'Rejected savings do not become the next baseline', 'The 8-unit change passes quality and replaces the 10-unit baseline. The 6-unit change fails tail quality, so the accepted baseline stays 8.', { rows: [['Original accepted baseline', 10, 'input'], ['Change 1: quality passes', 8, 'output'], ['Change 2: quality fails; reject', 6, 'warning']], unit: ' toy cost units' });
define('regularization-diagnose-and-target', 'matrix', 'An observed gap does not identify a unique cause', 'The same train/validation gap can accompany leakage, distribution shift, or overfitting. A discriminating check precedes the choice of regularizer.', { values: [['gap', '?'], ['gap', '?'], ['gap', '?']], columns: ['observed', 'cause known'], labels: ['leakage', 'shift', 'overfit'], blocked: [[0, 1], [1, 1], [2, 1]] });
define('fair-comparison-evidence-ledger', 'matrix', 'Unequal tuning opportunities confound the method comparison', 'Method B received ten times the search trials. Its best score cannot isolate the method effect without a matched tuning comparison.', { values: [[10, 'best A'], [100, 'best B']], columns: ['trials', 'reported'], labels: ['method A', 'method B'], blocked: [[1, 0]] });
define('environment-grader-contract-handoff', 'matrix', 'A successful final state can fail the required process', 'The environment reaches the goal while violating an allowed-tool constraint. The grader must retain both outcome and process evidence.', { values: [['yes', 'NO', 'FAIL']], columns: ['goal', 'process', 'trial'], labels: ['run'], selected: [[0, 0]], blocked: [[0, 1], [0, 2]] });
define('rlhf-dpo-training-paths', 'matrix', 'DPO reuses preference pairs without a separate reward-model rollout loop', 'Both routes use preference information. DPO directly optimizes on pairs; the illustrated RLHF route adds reward-model fitting and policy rollouts.', { values: [['yes', 'yes', 'yes'], ['yes', 'no', 'no']], columns: ['pairs', 'reward fit', 'rollouts'], labels: ['RLHF', 'DPO'], selected: [[0, 0], [1, 0]] });
define('robotics-policy-dagger-recovery', 'connections', 'Ask the expert about the state the learner actually reached', 'The learner drifts away from the demonstration state. DAgger labels that visited state so the dataset includes corrective decisions there.', { nodes: [['Demonstration state', 0, 0, 'input'], ['Learner-visited drift state', 1, 0, 'warning'], ['Expert corrective action', 1, 1, 'state'], ['Aggregated training data', 1, 2, 'output']], edges: [[1, 2], [2, 3]] });
define('oversight-control-assumption-test', 'matrix', 'Competent evaluation and adversarial containment are separate assumptions', 'A judge can identify harmful behavior yet fail to contain the acting system. Test detection and control with independent evidence.', { values: [['yes', '?'], ['?', 'yes']], columns: ['detects', 'contains'], labels: ['eval test', 'control'], selected: [[0, 0], [1, 1]], blocked: [[0, 1], [1, 0]] });
define('ambiguous-problem-scope-loop', 'matrix', 'A complaint sample proposes a scope; a bounded check tests it', 'The observed complaints concern one slice. A discriminating sample from other slices decides whether the proposed scope should widen.', { values: [['seen', 'known'], ['not yet', '?']], columns: ['evidence', 'affected'], labels: ['complaints', 'other'], selected: [[0, 1]], blocked: [[1, 1]] });
define('speculative-decoding-first-rejection-boundary', 'ledger', 'The first rejected draft invalidates its dependent suffix', 'After accepting A and rejecting draft B, later draft C cannot survive unchanged. Draw the corrected replacement and restart from the valid prefix.', { rows: [['Draft sequence', ['A', 'B', 'C'], 'input'], ['Verification', ['accept', 'REJECT', 'discard'], 'warning'], ['Committed prefix after correction', ['A', 'replacement'], 'output']] });
define('streaming-asr-provisional-to-final', 'ledger', 'Visible partial words remain revisable until commitment', 'The partial hypothesis changes as audio arrives. Only the declared finalization rule turns a provisional token sequence into committed output.', { rows: [['Early partial', ['I', 'scream'], 'input'], ['Revised partial', ['ice', 'cream'], 'state'], ['Finalization receipt', ['ice cream', 'committed'], 'output']] });
define('strong-scaling-wall-time-cost-knee', 'matrix', 'A shorter run can spend more accelerator-hours', 'Eight devices for two hours cost sixteen device-hours. Sixteen devices for 1.5 hours finish sooner but cost twenty-four device-hours.', { values: [[8, 2, 16], [16, 1.5, 24]], columns: ['devices', 'hours', 'dev-hours'], labels: ['run A', 'run B'], selected: [[1, 1]], blocked: [[1, 2]] });
define('synthetic-data-two-independent-evidence-gates', 'matrix', 'Filtering training rows is not independent final evaluation', 'A training-row judge can filter synthetic examples. Reusing the same feedback for final selection does not supply independent confirmation.', { values: [['yes', 'train'], ['fresh', 'confirm']], columns: ['evidence', 'purpose'], labels: ['filter', 'held-out'], selected: [[1, 0], [1, 1]] });
define('tensor-parallel-ffn-handoff', 'ledger', 'First-layer channels partition; second-layer contributions add', 'Two channel shards produce partial outputs 4 and 6 for one coordinate. The full coordinate is their sum 10, not their concatenation.', { rows: [['First-layer channel shards', ['channels A', 'channels B'], 'input'], ['Second-layer partial coordinate', ['4', '6'], 'state'], ['Reduced complete coordinate', ['4', '+', '6', '=', '10'], 'output']] });
define('test-time-compute-verifier-bottleneck', 'matrix', 'Generating a correct answer is insufficient if the verifier rejects it', 'Candidate A is correct but discarded; B is wrong but selected. More generation alone does not repair this selection bottleneck.', { values: [['yes', 'NO'], ['no', 'yes']], columns: ['correct', 'selected'], labels: ['A', 'B'], blocked: [[0, 1], [1, 1]] });
define('train-100b-parallelism-map', 'matrix', 'Different parallelism axes partition different objects', 'Data replicas partition examples; pipeline stages partition layers; tensor shards partition computations. Optimizer-state sharding changes stored ownership rather than example count.', { values: [['examples', 'distinct'], ['layers', 'same data'], ['tensor', 'same data'], ['state', 'ownership']], columns: ['partition', 'effect'], labels: ['DP', 'PP', 'TP', 'ZeRO'], selected: [[0, 1], [3, 1]] });
define('transformer-training-inference-ledgers', 'bars', 'Stored experts and active token computation belong in different ledgers', 'In a four-expert top-two model, all four expert weights remain stored. A single token executes only its two selected experts.', { rows: [['Experts stored', 4, 'input'], ['Experts active for this token', 2, 'output']], unit: ' experts' });
define('two-tower-offline-online-split', 'matrix', 'Item encoding happens before the request; query encoding happens during it', 'Offline item vectors populate the index. The live request encodes its query and searches that versioned index without re-encoding every item.', { values: [['encode', 'build'], ['query', 'search']], columns: ['compute', 'index'], labels: ['offline', 'online'], selected: [[0, 0], [1, 0]] });
define('two-tower-cross-encoder-fanout', 'bars', 'Joint encoding spends work only on the retrieved candidate set', 'A two-tower stage retrieves K=20 from 1,000 items. The cross-encoder scores those twenty pairs; it cannot rescue an item outside them.', { rows: [['Candidate corpus', 1000, 'input'], ['Jointly encoded retrieved pairs', 20, 'output']], unit: ' items or pairs' });
define('vae-training-and-generation-paths', 'matrix', 'Generation has no observed input to encode', 'Training infers z from observed x through the encoder. Generation instead samples z from the prior and decodes it without q(z|x).', { values: [['yes', 'q(z|x)', 'decode'], ['no', 'prior', 'decode']], columns: ['observed x', 'latent source', 'decoder'], labels: ['train', 'generate'], selected: [[1, 1], [1, 2]], blocked: [[1, 0]] });
define('speculative-decoding-weight-read-ledger', 'bars', 'Accepted tokens must repay draft and verification work', 'With target verification cost 1 and three draft costs of 0.1, two accepted tokens cost 0.65 each. One accepted token would cost 1.3, exceeding baseline 1.', { rows: [['Baseline target cost per token', 1, 'input'], ['Two accepted: 1.3 / 2', .65, 'output'], ['Only one accepted: 1.3 / 1', 1.3, 'warning']], unit: ' toy cost per accepted token' });
define('cross-validation-fit-the-question', 'matrix', 'A split is valid only for the dependency the claim must respect', 'Random rows test a different claim from unseen users or future deployment. Select the boundary before interpreting the held-out score.', { values: [['same users', 'mixed time'], ['new users', 'mixed time'], ['past', 'future']], columns: ['train/test users', 'time boundary'], labels: ['random', 'group', 'temporal'], selected: [[1, 0], [2, 1]] });
define('wsd-checkpoint-lineage', 'connections', 'A cooldown branch does not have to replace the hot mainline', 'A checkpoint forks into a continued hot run and a cooldown evaluation branch. Their lineage is shared; promotion remains a separate decision.', { nodes: [['Shared checkpoint', 2, 0, 'input'], ['Hot continuation', 0, 1, 'input'], ['Cooldown branch', 1, 1, 'state'], ['Branch evidence', 1, 2, 'output']], edges: [[0, 1], [0, 2], [2, 3]] });
define('coding-product-mode-boundaries', 'matrix', 'The modes share context, not tool permission', 'Inline, chat, and review remain nonexecuting modes. The isolated agent receives only its explicitly scoped tool authority.', { values: [['yes', 'no'], ['yes', 'no'], ['yes', 'no'], ['yes', 'scoped']], columns: ['context', 'tools'], labels: ['inline', 'chat', 'review', 'agent'], selected: [[3, 1]], blocked: [[0, 1], [1, 1], [2, 1]] });
define('coding-product-task-lifecycle', 'timeline', 'Cancellation withdraws authority before it proves worker termination', 'The request cancels future dispatch immediately. A later worker-stop receipt establishes termination; neither event means the patch was successfully verified.', { events: [['Cancel request; future dispatch ends', 0, 'warning'], ['Worker may still be stopping', 1, 'warning'], ['Worker-stop receipt arrives', 2, 'output']] });
define('foundation-data-build-planes', 'matrix', 'Completed shards still need permission and validation before publication', 'Artifact completion alone does not authorize use. Policy permission and passing validation must both join the completed artifact before release.', { values: [['yes', 'yes', 'NO', 'hold'], ['yes', 'NO', 'yes', 'hold'], ['yes', 'yes', 'yes', 'ready']], columns: ['built', 'policy', 'valid', 'publish'], labels: ['case A', 'case B', 'case C'], selected: [[2, 3]], blocked: [[0, 2], [1, 1]] });
define('multimodal-path-boundaries', 'timeline', 'Permission is not an occurrence receipt', 'Correlated receipt confirms occurrence.', { compact: true, events: [['Exact action authorized', 0, 'input'], ['Command dispatched', 1, 'state'], ['Device receipt confirms effect', 2, 'output']] });
define('multimodal-turn-lifecycle', 'matrix', 'Old-epoch packets stay invalid', 'Retired-epoch packets stay rejected.', { compact: true, values: [[7, 7, 'accept'], [8, 7, 'REJECT'], [8, 8, 'accept']], columns: ['active', 'packet', 'output'], labels: ['before', 'late', 'new'], selected: [[0, 2], [2, 2]], blocked: [[1, 2]] });
define('video-publication-evidence', 'matrix', 'Rollback restores a compatible bundle, not just an old model', 'Model version 1 cannot be paired blindly with version-2 features and policy. Restore the verified compatible model-feature-policy bundle.', { values: [['v1', 'v2', 'v2'], ['v1', 'v1', 'v1']], columns: ['model', 'features', 'policy'], labels: ['unsafe', 'bundle'], blocked: [[0, 1], [0, 2]], selected: [[1, 0], [1, 1], [1, 2]] });
define('stairs-final-move-partition', 'parts', 'Every path belongs to exactly one last-move group', 'Five paths end with one step; three end with two steps. The disjoint groups sum to eight paths up five stairs.', { rows: [['last move is one step', 5, 'input'], ['last move is two steps', 3, 'output']], unit: ' paths' });
define('course-prerequisite-fanout', 'ledger', 'One completed parent still leaves the other prerequisite unmet', 'Course 3 requires courses 1 and 2. Its unmet count falls from two to one to zero, becoming ready only at zero.', { rows: [['Neither prerequisite emitted', ['1: wait', '2: wait', 'count 2'], 'input'], ['Only course 1 emitted', ['1: done', '2: wait', 'count 1'], 'state'], ['Both prerequisites emitted', ['1: done', '2: done', 'READY'], 'output']] });
define('robber-take-skip-boundaries', 'bars', 'Taking the current house excludes its immediate neighbor', 'For values (2,7,9,3), skipping house 3 keeps 11. Taking it adds 3 only to the earlier optimum 7, giving 10.', { rows: [['Skip index 3: houses 0 and 2', 11, 'output'], ['Take index 3: best through 1 plus 3', 10, 'input']], unit: ' value' });
define('lcs-conditional-parent-cells', 'matrix', 'A match extends the diagonal, not an arbitrary neighbor', 'For AB versus ACB, matching B extends diagonal value 1 to 2. At mismatch C, compare the up and left subproblems instead.', { values: [[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 2]], columns: ['empty', 'A', 'C', 'B'], labels: ['empty', 'A', 'B'], selected: [[1, 2], [2, 3]] });
define('subset-sum-old-snapshot', 'ledger', 'The new item can extend the old sums only once', 'Starting from old sums (0,1), item 5 creates only (5,6). Ten is absent because a newly created five cannot reuse that same item.', { rows: [['Old reachable snapshot', ['0', '1'], 'input'], ['Add the current item 5 once', ['5', '6'], 'state'], ['Union only after creating new sums', ['0', '1', '5', '6'], 'output']] });
define('dsu-edges-versus-parents', 'matrix', 'Path compression changes bookkeeping, not the original graph', 'Finding node 4 redirects its parent from 3 to root 1. The accepted problem edges remain unchanged.', { values: [['1-2', '2-3', '3-4'], ['2->1', '3->1', '4->3'], ['2->1', '3->1', '4->1']], columns: ['first', 'second', 'third'], labels: ['edges', 'parents', 'after'], selected: [[2, 2]] });
define('subsets-increasing-index-tree', 'ledger', 'Increasing choice indices erase duplicate orderings', 'Indices 0 then 1 generate subset (1,2). Attempting 1 then 0 goes backward and would duplicate the same membership.', { rows: [['Allowed index path', ['0', '->', '1', '=', '{1,2}'], 'output'], ['Rejected ordering', ['1', '->', '0', '=', '{2,1}'], 'warning'], ['Membership result', ['same subset', 'emit once'], 'state']] });

coding('3sum', 'matrix', 'one comparison rejects a whole row of pairs', 'With fixed -4 and left -2, even partner 2 gives -4. Every smaller partner also fails, so discard that left value.', { values: [[-5, -4], ['reject', 'reject']], columns: ['partner 1', 'partner 2'], labels: ['sum', 'decision'], blocked: [[1, 0], [1, 1]] });
coding('alien-dictionary', 'matrix', 'only the first unequal character creates an ordering edge', 'The words wrt and wrf first differ at their third character. Infer t before f; the shared prefix supplies no new edge.', { values: [['w', 'r', 't'], ['w', 'r', 'f']], columns: ['0', '1', '2'], labels: ['wrt', 'wrf'], selected: [[0, 2], [1, 2]] });
coding('balanced-binary-tree', 'ledger', 'a failed child cannot be repaired by a balanced parent', 'A child returns the unbalanced sentinel, not a usable height. Propagate that failure rather than averaging or comparing it as a valid depth.', { rows: [['Returned child results', ['left: FAIL', 'right: 2'], 'input'], ['Parent result', ['FAIL'], 'warning']] });
coding('best-time-to-buy-and-sell-stock', 'ledger', 'the saved trade survives a worse current selling day', 'Prices (3,1,5,2) produce best profit 4 at price 5. The final price gives current profit 1 but does not erase the saved 4.', { rows: [['Observed prices', ['3', '1', '5', '2'], 'input'], ['Best earlier purchase', ['1'], 'state'], ['Current profit / saved best at final day', ['1', '4'], 'output']] });
coding('binary-precision-and-recall', 'matrix', 'one hit count has two denominators', 'Eight true positives accompany two misses and 32 false alarms. Recall is 8/10; precision is 8/40.', { values: [[8, 2], [32, 58]], columns: ['pred +', 'pred -'], labels: ['actual +', 'actual -'], selected: [[0, 0]], blocked: [[0, 1], [1, 0]] });
coding('binary-search', 'ledger', 'the comparison removes an entire sorted region', 'For target 7 in (1,3,5,7,9), middle value 5 is too small. Every value at or left of that middle is also too small.', { rows: [['Sorted input', ['1', '3', '5', '7', '9'], 'input'], ['Remaining after comparing middle 5', ['out', 'out', 'out', '7', '9'], 'output']] });
coding('binary-tree-level-order-traversal', 'ledger', 'freeze the current level before enqueueing its children', 'The level begins with A and B, so process exactly two nodes. Newly appended children belong to the next level.', { rows: [['Queue at level start; size = 2', ['A', 'B'], 'input'], ['After A enqueues its children', ['B', 'A1', 'A2'], 'state'], ['Do not treat new length 3 as this level', ['process B', 'then stop'], 'output']] });
coding('binary-tree-maximum-path-sum', 'ledger', 'a winning path may branch, but the returned path may not', 'At value 10 with positive child gains 4 and 6, the local complete path scores 20. The upward return is 16, using only one child.', { rows: [['Path allowed to end at both children', ['4', '+10', '+6', '=20'], 'input'], ['One extendable path returned upward', ['10', '+max(4,6)', '=16'], 'output']] });
coding('causal-attention', 'matrix', 'future positions receive zero weight', 'A query at position 1 can use positions 0 and 1 only. Masked future entries contribute exactly zero to the output.', { values: [[1, 0, 0], [.4, .6, 0], [.2, .3, .5]], columns: ['key 0', 'key 1', 'key 2'], labels: ['query 0', 'query 1', 'query 2'], selected: [[1, 0], [1, 1]], blocked: [[0, 1], [0, 2], [1, 2]] });
coding('climbing-stairs', 'parts', 'the final move partitions paths into disjoint groups', 'For five stairs, five paths end with one step and three with two. These groups do not overlap, so their counts add to eight.', { rows: [['paths ending with one step', 5, 'input'], ['paths ending with two steps', 3, 'output']], unit: ' paths' });
coding('clone-graph', 'matrix', 'a cycle reuses the already-created copy', 'When B reaches A again, the original-to-copy map returns A-prime. Linking to original A would corrupt the clone boundary.', { values: [['A-prime', 'reuse'], ['A', 'REJECT']], columns: ['B-prime links to', 'decision'], labels: ['copy map', 'original'], selected: [[0, 1]], blocked: [[1, 1]] });
coding('coin-change', 'ledger', 'an unreachable predecessor supplies no valid solution', 'For coins (2,3) and amount 3, coin 2 points to unreachable amount 1. Coin 3 points to base amount 0 and supplies count 1.', { rows: [['Candidate through coin 2', ['dp[1]=4', '+1', '=5', 'reject'], 'warning'], ['Candidate through coin 3', ['dp[0]=0', '+1', '=1', 'keep'], 'output']] });
coding('combination-sum-iv', 'parts', 'group ordered sequences by their last value', 'With values (1,2) and target 4, group by last value. Three sequences end in 1; two end in 2.', { rows: [['ordered sequences ending in 1', 3, 'input'], ['ordered sequences ending in 2', 2, 'output']], unit: ' ordered sequences' });
coding('combination-sum', 'ledger', 'reuse a value without returning to an earlier choice index', 'For candidates (2,3), path (2,2,3) is allowed. Path (3,2,2) would duplicate the same combination and requires a backward index move.', { rows: [['Allowed nondecreasing choice indices', ['0', '0', '1'], 'output'], ['Same membership in a rejected order', ['1', '0', '0'], 'warning']] });
coding('construct-tree-from-preorder-and-inorder-traversal', 'ledger', 'preorder selects the root; inorder assigns subtree membership', 'Preorder begins with 3. Inorder (9,3,15,20,7) places 9 on its left and the remaining three nodes on its right.', { rows: [['Preorder root selection', ['3', '9', '20', '15', '7'], 'input'], ['Inorder split at that same root', ['9', '|3|', '15', '20', '7'], 'state'], ['Subtree sizes', ['left: 1', 'right: 3'], 'output']] });
coding('container-with-most-water', 'ledger', 'keeping the shorter wall cannot improve its current limit', 'With shorter wall 2 and width 3, area is 6. Moving only the taller wall makes width 2 and height at most 2, so area is at most 4.', { rows: [['Current shorter-height bound', ['2', 'x', '3', '=', '6'], 'input'], ['Move taller wall only', ['<=2', 'x', '2', '<=', '4'], 'warning']] });
coding('contains-duplicate', 'ledger', 'repeated positions collapse to one set entry', 'The input has four positions but only three distinct values. The unequal counts prove a duplicate without needing to sort.', { rows: [['Input positions', ['1', '2', '1', '3'], 'input'], ['Distinct set entries', ['1', '2', '3'], 'output']] });
coding('count-number-of-nice-subarrays', 'parts', 'fix the last endpoint of (2,2,2,1,1,1)', 'Six starts contain at most three odds. Two contain at most two odds. Their difference leaves four starts containing exactly three.', { rows: [['exactly three odd values', 4, 'output'], ['at most two odd values', 2, 'neutral']], unit: ' valid starts' });
coding('counting-bits', 'ledger', 'reuse the halved number and add its low bit', 'Binary 1101 has the count of 110 plus one. Thus count(13)=count(6)+1=3.', { rows: [['13 in binary', ['1', '1', '0', '1'], 'input'], ['13 shifted right; count 2', ['1', '1', '0'], 'state'], ['Reuse and add low bit', ['2', '+', '1', '=', '3'], 'output']] });
coding('course-schedule-ii', 'ledger', 'a course is ready only when its final prerequisite disappears', 'Course 3 has prerequisites 1 and 2. Emitting only 1 leaves its unmet count at one; emitting 2 then makes it ready.', { rows: [['Unmet count before parents', ['2'], 'input'], ['After emitting prerequisite 1', ['1: still blocked'], 'warning'], ['After emitting prerequisite 2', ['0: enqueue'], 'output']] });
coding('course-schedule', 'matrix', 'an empty ready queue is not enough to prove success', 'Both cases have an empty queue. Only the case that emitted all courses succeeds; remaining courses indicate a cycle.', { values: [['empty', 'all', 'yes'], ['empty', 'not all', 'NO']], columns: ['ready', 'emitted', 'success'], labels: ['done', 'cycle'], selected: [[0, 2]], blocked: [[1, 2]] });
coding('cross-entropy-from-logits', 'ledger', 'the target logit matters even when the normalizer is shared', 'For logits (2,0), log-sum-exp is about 2.127. Choosing class 0 gives loss 0.127; choosing class 1 gives 2.127.', { rows: [['Shared logits and log normalizer', ['2', '0', 'logZ=2.127'], 'input'], ['Loss when target is class 0', ['2.127', '-2', '=.127'], 'output'], ['Loss when target is class 1', ['2.127', '-0', '=2.127'], 'warning']] });
coding('daily-temperatures', 'ledger', 'a wait is an index difference, not a temperature difference', 'Index 0 stores 70 degrees; index 3 stores 80. The warmer day arrives three days later, not ten days later.', { rows: [['Temperature and index pair', ['70 at 0', '80 at 3'], 'input'], ['Correct wait', ['3', '-', '0', '=', '3 days'], 'output'], ['Wrong quantity', ['80-70', '=', '10 degrees'], 'warning']] });
coding('decode-string', 'ledger', 'finish the inner repeat before applying its outer multiplier', 'For 2[a3[b]], first decode 3[b] to bbb. Then repeat the completed abbb twice.', { rows: [['Innermost pending expression', ['3[b]', '->', 'bbb'], 'input'], ['Completed outer payload', ['a', '+', 'bbb', '=', 'abbb'], 'state'], ['Apply outer repeat 2', ['abbb', 'abbb'], 'output']] });
coding('decode-ways', 'matrix', 'zero is allowed in 10, not as the leading digit of 01', 'The pair 10 maps to a letter, while 01 is invalid. The two-digit transition must satisfy the actual range 10 through 26.', { values: [['10', 'yes'], ['01', 'NO'], ['27', 'NO']], columns: ['pair', 'valid'], labels: ['case A', 'case B', 'case C'], selected: [[0, 1]], blocked: [[1, 1], [2, 1]] });
coding('design-add-and-search-words', 'matrix', 'a matching prefix is not automatically a stored word', 'The trie contains bad. Prefix ba exists, but its final node is not terminal; wildcard b.d reaches the terminal for bad.', { values: [['yes', 'no', 'false'], ['yes', 'yes', 'true']], columns: ['path', 'terminal', 'search'], labels: ['ba', 'b.d'], selected: [[1, 2]], blocked: [[0, 2]] });
coding('edit-distance', 'ledger', 'a mismatch pays one edit; a match inherits the diagonal', 'For single characters a and b, substitution costs one. Matching a against a adds zero to the empty-prefix diagonal.', { rows: [['Mismatch a versus b', ['dp[0,0]', '+1', '=1'], 'warning'], ['Match a versus a', ['dp[0,0]', '+0', '=0'], 'output']] });
coding('encode-and-decode-strings', 'ledger', 'length, not delimiter appearance, determines the payload boundary', 'Encoding a#b as 3#a#b preserves its internal #. Read exactly three payload characters after the length delimiter.', { rows: [['Length prefix and delimiter', ['3', '#'], 'input'], ['Exactly three payload characters', ['a', '#', 'b'], 'state'], ['Decoded single string', ['a#b'], 'output']] });
coding('find-median-from-data-stream', 'matrix', 'balanced heap sizes still need the ordering invariant', 'Heaps (5) and (1) have equal sizes but reversed ordering. Swapping their tops restores max(left)<=min(right) before reading the median.', { values: [[5, 1, 'NO'], [1, 5, 'yes']], columns: ['left max', 'right min', 'ordered'], labels: ['before', 'after'], selected: [[1, 2]], blocked: [[0, 2]] });
coding('find-minimum-in-rotated-sorted-array', 'ledger', 'the middle remains a candidate when it lies on the low side', 'For (3,4,1,2), first discard middle 4 and its left side. Next, middle 1 is below rightmost 2: set right=mid, keeping that middle as the minimum candidate.', { rows: [['Initial rotated values', ['3', '4', '1', '2'], 'input'], ['Middle 4 > right 2: discard middle 4', ['out', 'out', '1', '2'], 'state'], ['Middle 1 < right 2: retain middle 1', ['out', 'out', '1', 'out'], 'output']] });
coding('graph-valid-tree', 'topology', 'the edge count alone cannot rule out a cycle plus isolation', 'A triangle on three nodes plus one isolated node has n-1 edges for n=4. It is not a tree because connectivity fails.', { nodes: [['A', 60, 65], ['B', 190, 65], ['C', 125, 185], ['D', 300, 150]], edges: [[0, 1], [1, 2], [2, 0]] });
coding('group-anagrams', 'ledger', 'equal lengths do not imply equal character counts', 'Both ab and ba count one a and one b. By contrast, aa counts two a characters.', { rows: [['Words', ['ab', 'ba', 'aa'], 'input'], ['Character-count signatures', ['a1 b1', 'a1 b1', 'a2 b0'], 'state'], ['Groups', ['{ab,ba}', '{aa}'], 'output']] });
coding('house-robber-ii', 'matrix', 'two linear cases prevent taking both circular endpoints', 'For a circle, solve once without the last house and once without the first. Every valid solution belongs to at least one of these cases.', { values: [['yes', 'yes', 'no'], ['no', 'yes', 'yes']], columns: ['first', 'middle', 'last'], labels: ['case A', 'case B'], selected: [[0, 0], [1, 2]], blocked: [[0, 2], [1, 0]] });
coding('house-robber', 'bars', 'taking the current house can reuse only the earlier nonadjacent optimum', 'At index 3 of (2,7,9,3), skip gives 11. Taking the current value gives best-through-index-1 plus 3, or 10.', { rows: [['Skip current: keep 11', 11, 'output'], ['Take current: 7 + 3', 10, 'input']], unit: ' value' });
coding('implement-trie', 'matrix', 'prefix reachability and word membership are separate flags', 'After inserting apple, app is a valid prefix but not yet a word. Inserting app changes its terminal flag without rebuilding the shared path.', { values: [['yes', 'no'], ['yes', 'yes']], columns: ['path app', 'terminal'], labels: ['apple only', 'plus app'], selected: [[1, 1]], blocked: [[0, 1]] });
coding('insert-interval', 'ledger', 'merge the overlap but stop before the first separated interval', 'Inserting (2,5) into (1,3),(6,9) merges only the first interval. The separated (6,9) stays unchanged.', { rows: [['Sorted intervals plus new interval', ['[1,3]', 'new [2,5]', '[6,9]'], 'input'], ['Updated merged result', ['[1,5]', '[6,9]'], 'output']] });
coding('invert-binary-tree', 'matrix', 'swap subtree positions without changing their stored values', 'Root 1 keeps value 1 while its returned left and right children exchange positions. Replacing values alone is not the recursive subtree operation.', { values: [[1, 2, 3], [1, 3, 2]], columns: ['root', 'left', 'right'], labels: ['before', 'after'], selected: [[1, 1], [1, 2]] });
coding('jump-game', 'ledger', 'an unreachable index cannot donate its large jump', 'For (1,0,9), the reachable frontier stops at index 1. The large value at index 2 is never reachable and cannot extend that frontier.', { rows: [['Values by index', ['1', '0', '9'], 'input'], ['Reachability from index 0', ['yes', 'yes', 'NO'], 'warning']] });
coding('koko-eating-bananas', 'bars', 'a passing speed rules out the need for faster candidates', 'For piles (3,6,7,11) and eight hours, speed 3 needs ten hours. Speed 4 needs eight and is feasible; every lower speed fails.', { rows: [['Speed 3: required hours', 10, 'warning'], ['Speed 4: required hours', 8, 'output']], limit: 8, limitLabel: 'deadline', unit: ' hours' });
coding('kth-largest-element', 'ledger', 'the kth largest is the smallest value in the retained top-k heap', 'For k=3 and values (7,2,9,4), retain (4,7,9). Its smallest value 4 is third largest, not the global minimum 2.', { rows: [['All observed values', ['7', '2', '9', '4'], 'input'], ['Largest three retained', ['4', '7', '9'], 'state'], ['Heap minimum is the kth largest', ['4'], 'output']] });
coding('kth-smallest-element-in-a-bst', 'ledger', 'count a node when inorder visits it, not when it enters the stack', 'A root 2 can enter the stack before its left child 1. The visit order is still 1 then 2 then 3.', { rows: [['Possible stack push order', ['2', '1'], 'input'], ['Actual inorder visit order', ['1', '2', '3'], 'output']] });
coding('largest-rectangle-in-histogram', 'ledger', 'a shorter bar closes spans but the zero sentinel remains', 'After heights (2,1,2) and a zero sentinel, positive spans are resolved. The actual stack still retains the zero sentinel; it is not an empty stack.', { rows: [['Heights with appended sentinel', ['2', '1', '2', '0'], 'input'], ['Unresolved positive heights', ['none'], 'state'], ['Actual retained stack entry', ['sentinel 0'], 'output']] });
coding('linked-list-cycle', 'matrix', 'compare node identity, not repeated stored values', 'Two distinct nodes may both store value 7. A cycle witness requires the slow and fast pointers to reach the same node identity after valid moves.', { values: [['node A', 'node B', 'no'], ['node C', 'node C', 'yes']], columns: ['slow', 'fast', 'same node'], labels: ['equal 7s', 'meeting'], selected: [[1, 2]], blocked: [[0, 2]] });
coding('longest-common-subsequence', 'matrix', 'a matching pair extends the diagonal subproblem', 'For AB versus ACB, B matches B and extends diagonal 1 to 2. A mismatch compares the two shorter-prefix answers instead.', { values: [[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 2]], columns: ['empty', 'A', 'C', 'B'], labels: ['empty', 'A', 'B'], selected: [[1, 2], [2, 3]] });
coding('longest-consecutive-sequence', 'ledger', 'start a run only where no predecessor exists', 'For set (1,2,3,8), start at 1 and 8. Starting again at 2 or 3 repeats work inside the already-covered run.', { rows: [['Set membership', ['1', '2', '3', '8'], 'input'], ['Predecessor present?', ['no', 'yes', 'yes', 'no'], 'state'], ['Start a traversal?', ['yes', 'no', 'no', 'yes'], 'output']] });
coding('longest-increasing-path-in-a-matrix', 'ledger', 'a cached suffix can serve more than one starting point', 'Paths starting at 1 and 2 can both enter suffix (3,4). The suffix length is solved once and reused; the path still follows strictly increasing values.', { rows: [['First start', ['1', '3', '4'], 'input'], ['Second start reuses the same suffix', ['2', '3', '4'], 'state'], ['Memoized length at value 3', ['2'], 'output']] });
coding('longest-increasing-subsequence', 'ledger', 'a smaller tail preserves length but is not the saved subsequence', 'Processing (3,5,7,2) changes the length-one tail to 2. Tails (2,5,7) are extension bounds, not a subsequence of that processed input.', { rows: [['Processed values', ['3', '5', '7', '2'], 'input'], ['Tails before final value', ['3', '5', '7'], 'state'], ['Tails after final value', ['2', '5', '7'], 'output']] });
coding('longest-palindromic-substring', 'bars', 'a shorter new palindrome cannot replace the saved best', 'A saved palindrome of length 5 remains best when a later center finds length 3. Update only when the new length is strictly larger under this tie rule.', { rows: [['Saved best length', 5, 'output'], ['Later center palindrome length', 3, 'input']], unit: ' characters' });
coding('longest-repeating-character-replacement', 'ledger', 'the saved best need not describe the final shown window', 'AABA needs one replacement and has length 4. A later ABBA window needs two; it does not invalidate the earlier saved optimum of 4.', { rows: [['Earlier valid witness; k=1', ['AABA', '1 edit', 'length 4'], 'output'], ['Later window', ['ABBA', '2 edits', 'not witness'], 'warning']] });
coding('longest-substring-without-repeating-characters', 'timeline', 'an old duplicate cannot move the left boundary backward', 'In abba, the last a was at index 0 while left already equals 2. Set left=max(2,0+1)=2 rather than reopening the duplicate b.', { events: [['Previous a index', 0, 'input'], ['Candidate last[a]+1', 1, 'warning'], ['Current and retained left bound', 2, 'output']], unit: 'string index on one shared scale' });
coding('lowest-common-ancestor-in-a-bst', 'matrix', 'stop when the targets split or one equals the current node', 'At node 6 with targets 2 and 8, their search directions split. If one target is 6 itself, that same node is already the ancestor.', { values: [['left', 'right', 'STOP'], ['equal', 'right', 'STOP']], columns: ['target 1', 'target 2', 'decision'], labels: ['split', 'equal'], selected: [[0, 2], [1, 2]] });
coding('lru-cache', 'ledger', 'a read changes the victim of the next eviction', 'With key 1 older than key 2, reading 1 moves it to most-recent position. Inserting key 3 then evicts key 2.', { rows: [['Oldest to newest before read', ['1', '2'], 'input'], ['After get(1)', ['2', '1'], 'state'], ['After inserting 3 at capacity 2', ['1', '3'], 'output']] });
coding('maximum-depth-of-binary-tree', 'ledger', 'add this node to the larger returned child depth', 'Child depths 2 and 4 produce parent depth 5. Adding both child depths would count nodes from different root-to-leaf paths.', { rows: [['Returned child depths', ['2', '4'], 'input'], ['Parent root-to-leaf depth', ['1', '+', 'max(2,4)', '=', '5'], 'output']] });
coding('maximum-product-subarray', 'ledger', 'a negative minimum becomes a positive maximum after a sign flip', 'For (-2,3,-4), the previous minimum -6 becomes 24 after multiplying by -4. Retaining only the previous maximum would lose this answer.', { rows: [['After values -2 and 3', ['max: 3', 'min: -6'], 'input'], ['Candidates after multiplying by -4', ['-4', '-12', '24'], 'state'], ['New maximum', ['24'], 'output']] });
coding('maximum-subarray', 'ledger', 'the saved best survives a losing final extension', 'For (-2,3,4,-10), the final current sum is -3. The saved best remains 7 from subarray (3,4).', { rows: [['Input', ['-2', '3', '4', '-10'], 'input'], ['Final current sum / saved optimum', ['-3', '7'], 'output']] });
coding('meeting-rooms-ii', 'bars', 'the peak room demand survives a smaller final active heap', 'Meetings (0,2),(.5,1.5),(3,4) overlap two at a time near t=1. After removing finished meetings, the final heap holds only one. Two rooms are still required for the saved peak.', { rows: [['Peak active meetings near t=1', 2, 'output'], ['Final active heap near t=3.5', 1, 'input']], unit: ' rooms' });
coding('meeting-rooms', 'matrix', 'touching endpoints are not overlapping meeting interiors', 'The test is next_start<previous_end. Equality permits attendance; a strictly earlier start creates a conflict.', { values: [[10, 10, 'yes'], [10, 9, 'NO']], columns: ['prev end', 'next start', 'attend'], labels: ['touch', 'overlap'], selected: [[0, 2]], blocked: [[1, 2]] });
coding('merge-intervals', 'ledger', 'extend the active endpoint before comparing the next interval', 'Intervals (1,4) and (2,6) merge to (1,6). The later (5,7) overlaps that updated endpoint, so all three merge to (1,7).', { rows: [['Sorted intervals', ['[1,4]', '[2,6]', '[5,7]'], 'input'], ['First merge updates the endpoint', ['[1,6]', '[5,7]'], 'state'], ['Compare against updated 6, not old 4', ['[1,7]'], 'output']] });
coding('merge-k-sorted-lists', 'ledger', 'advance the list that supplied the minimum, then compare again', 'Current heads are (1,4,2). After emitting 1 and exposing 9, the next minimum is 2, not the next value from that same list.', { rows: [['Current list heads', ['1', '4', '2'], 'input'], ['Heads after emitting 1', ['9', '4', '2'], 'state'], ['Next emitted value', ['2'], 'output']] });
coding('merge-two-sorted-lists', 'ledger', 'the untouched remainder is already sorted', 'After one input is exhausted, append the other input head directly. Its internal links already represent the correctly ordered suffix.', { rows: [['Remaining input heads', ['none', '4 -> 6 -> 9'], 'input'], ['Splice the surviving head once', ['output tail', '->', '4 -> 6 -> 9'], 'output']] });
coding('min-stack', 'ledger', 'duplicate minima need duplicate historical state', 'Pushing 5,2,2 stores minima 5,2,2. Popping one 2 must leave minimum 2; recording only distinct minima would incorrectly restore 5.', { rows: [['Values on the stack', ['5', '2', '2'], 'input'], ['Minimum stored with each value', ['5', '2', '2'], 'state'], ['After one pop: values and minimum', ['5,2', 'min 2'], 'output']] });
coding('mini-batches', 'ledger', 'bounded half-open slicing preserves a short tail', 'Seven items and batch size 3 yield lengths 3,3,1. The last slice keeps item 6 rather than dropping or duplicating it.', { rows: [['Slice [0:3]', ['0', '1', '2'], 'input'], ['Slice [3:6]', ['3', '4', '5'], 'state'], ['Slice [6:9], clipped at length 7', ['6'], 'output']] });
coding('minimum-window-substring', 'matrix', 'matching all character types is not enough when demand repeats', 'Target AABC needs two copies of A. Candidate ABC contains every character type but misses one required A.', { values: [[2, 1, 'NO'], [1, 1, 'yes'], [1, 1, 'yes']], columns: ['required', 'in ABC', 'enough'], labels: ['A', 'B', 'C'], selected: [[1, 2], [2, 2]], blocked: [[0, 2]] });
coding('missing-number', 'ledger', 'paired XOR terms cancel and leave the absent value', 'For n=3, observations (3,0,1) omit value 2. XORing observations against the full range (0,1,2,3) leaves only that unpaired value.', { rows: [['Expected range', ['0', '1', '2', '3'], 'input'], ['Observed values', ['3', '0', '1'], 'state'], ['Unpaired XOR remainder', ['2'], 'output']] });
coding('network-delay-time', 'matrix', 'a queued estimate is stale when a shorter distance is already known', 'Two entries can name the same node. With best distance 2, a later popped entry carrying distance 6 is stale and must not expand neighbors.', { values: [[2, 2, 'expand'], [2, 6, 'SKIP']], columns: ['best', 'popped', 'action'], labels: ['current', 'stale'], selected: [[0, 2]], blocked: [[1, 2]] });
coding('non-overlapping-intervals', 'matrix', 'the earlier finish preserves more future choices', 'Keeping (1,3) permits the next interval (3,4). Keeping (1,5) blocks it, despite the identical starting point.', { values: [[3, 3, 'yes'], [5, 3, 'NO']], columns: ['kept end', 'next start', 'fits'], labels: ['early', 'late'], selected: [[0, 2]], blocked: [[1, 2]] });
coding('number-of-1-bits', 'ledger', 'each pass clears exactly the lowest remaining one bit', 'Binary 1100 becomes 1000, then 0000 under n&(n-1). Two passes count two set bits, not four bit positions.', { rows: [['n = 12', ['1100'], 'input'], ['12 & 11 = 8', ['1000'], 'state'], ['8 & 7 = 0; two passes total', ['0000'], 'output']] });
coding('number-of-connected-components', 'ledger', 'only a successful union reduces the component count', 'Four nodes begin as four components. Joining (0,1) and (2,3) leaves two; joining (1,0) again does not reduce that count.', { rows: [['Initial count', ['4'], 'input'], ['After two distinct-component unions', ['3', '2'], 'state'], ['After duplicate-component union', ['2'], 'output']] });
coding('number-of-islands', 'matrix', 'mark a discovered cell before another neighbor discovers it again', 'From the top-left cell, two neighbors share the bottom-right cell. Marking on discovery puts that shared cell into the traversal once.', { values: [['start', 'next'], ['next', 'shared']], columns: ['0', '1'], labels: ['0', '1'], selected: [[1, 1]] });
coding('pacific-atlantic-water-flow', 'matrix', 'reverse search walks uphill to find possible downhill sources', 'Starting from an ocean, reverse traversal may move from height 2 to 5. Moving to height 1 reverses the intended reachability rule.', { values: [[2, 5, 'yes'], [2, 1, 'NO']], columns: ['current', 'neighbor', 'reverse step'], labels: ['uphill', 'downhill'], selected: [[0, 2]], blocked: [[1, 2]] });
coding('pad-variable-length-sequences', 'matrix', 'equal token values can have different validity', 'For sequences (0,7) and (4), the first zero is real data. The padded second-row zero is invalid; sequence lengths define the mask.', { values: [['0 / true', '7 / true'], ['4 / true', '0 / false']], columns: ['position 0', 'position 1'], labels: ['length 2', 'length 1'], selected: [[0, 0]], blocked: [[1, 1]] });
coding('pairwise-squared-distances', 'ledger', 'combine row norms with the cross-product, without a 3D difference tensor', 'For x=(1,2) and y=(3,2), squared distance is 5+13-2x7=4. The broadcasted identity produces that same scalar.', { rows: [['Squared row norms', ['5', '13'], 'input'], ['Dot product', ['1x3', '+', '2x2', '=', '7'], 'state'], ['Squared-distance identity', ['5', '+13', '-2x7', '=', '4'], 'output']] });
coding('palindromic-substrings', 'parts', 'count locations, not just distinct palindrome texts', 'String aaa has three length-one, two length-two, and one length-three palindromic substrings. These are six occurrences, not three distinct strings.', { rows: [['length-one occurrences', 3, 'input'], ['length-two occurrences', 2, 'state'], ['length-three occurrence', 1, 'output']], unit: ' occurrences' });
coding('partition-equal-subset-sum', 'matrix', 'descending sums prevent one item from being reused in its own update', 'From reachable sum 0, processing one item of value 2 may add sum 2 only. Forward iteration could incorrectly reuse it to make 4.', { values: [['yes', 'yes', 'NO'], ['yes', 'yes', 'yes']], columns: ['sum 0', 'sum 2', 'sum 4'], labels: ['one use', 'reuse bug'], selected: [[0, 1]], blocked: [[1, 2]] });
coding('permutation-in-string', 'matrix', 'character multiplicities must match, not only membership', 'Patterns aab and window abb share the same character set. Their counts differ, so the window is not a permutation.', { values: [[2, 1], [1, 2]], columns: ['a count', 'b count'], labels: ['aab', 'abb'], blocked: [[1, 0], [1, 1]] });
coding('permutations', 'ledger', 'unused positions remain available regardless of their original order', 'After choosing values (1,3) from (1,2,3), value 2 is still unused. An increasing-start rule would wrongly exclude that valid permutation.', { rows: [['Input positions', ['0:1', '1:2', '2:3'], 'input'], ['Partial permutation', ['1', '3'], 'state'], ['Remaining unused position', ['1:2'], 'output']] });
coding('product-of-array-except-self', 'ledger', 'left and right products exclude exactly the current position', 'For (2,0,4), prefix and suffix products produce (0,8,0). The middle output is nonzero even though its own input is zero.', { rows: [['Input', ['2', '0', '4'], 'input'], ['Products strictly left', ['1', '2', '0'], 'state'], ['Products strictly right', ['0', '4', '1'], 'state'], ['Multiply matching positions', ['0', '8', '0'], 'output']] });
coding('redundant-connection', 'ledger', 'the closing edge connects nodes already in one component', 'After joining (1,2) and (2,3), nodes 1 and 3 share a representative. Their new edge closes a cycle instead of merging components.', { rows: [['After first two unions', ['1', '2', '3', 'one group'], 'input'], ['Candidate edge (1,3)', ['same root', 'reject edge'], 'warning']] });
coding('remove-nth-node-from-end', 'ledger', 'the dummy predecessor makes deleting the head an ordinary splice', 'For (1,2,3) and n=3, the predecessor is the dummy node. Its next link skips node 1 and returns head 2.', { rows: [['Before deleting the third from the end', ['dummy', '1', '2', '3'], 'input'], ['After predecessor.next skips the target', ['dummy', '2', '3'], 'output']] });
coding('reorder-list', 'ledger', 'reverse the detached second half before weaving', 'Split after node 3 and reverse the detached second half. Alternating the detached halves yields (1,5,2,4,3) without a cycle.', { rows: [['First half, terminated after 3', ['1', '2', '3'], 'input'], ['Reversed detached second half', ['5', '4'], 'state'], ['Alternating merge', ['1', '5', '2', '4', '3'], 'output']] });
coding('reverse-bits', 'ledger', 'fixed-width reversal maps position i to position 31-i', 'For a 32-bit input containing only low bit 0, reversal sets high bit 31. The result is 2147483648; leading zeros still count as positions.', { rows: [['Input bit positions', ['31..1: 0', '0: 1'], 'input'], ['Reversed bit positions', ['31: 1', '30..0: 0'], 'output']] });
coding('reverse-linked-list', 'ledger', 'save the remaining chain before reversing the current link', 'Before redirecting node 1 toward null, save its original next node 2. That saved reference is the only route to the unprocessed suffix.', { rows: [['Original remaining chain', ['1', '2', '3'], 'input'], ['Save next before overwriting', ['next = 2'], 'state'], ['After flipping node 1', ['1 -> null', 'saved 2 -> 3'], 'output']] });
coding('rotate-image', 'matrix', 'transpose then reverse each row rotates coordinates clockwise', 'Input rows (1,2) and (3,4) rotate to (3,1) and (4,2). Reversing columns after transposing would rotate the other direction.', { values: [[1, 2], [3, 4], [3, 1], [4, 2]], columns: ['left', 'right'], labels: ['before 0', 'before 1', 'after 0', 'after 1'], selected: [[2, 0], [2, 1], [3, 0], [3, 1]] });
coding('rotting-oranges', 'matrix', 'each frozen queue layer advances time by one minute', 'This example reaches its last fresh orange at minute 4. Cells in the same wave share a time; six infections do not imply six minutes.', { values: [[0, 1, 2], [1, 2, 'wall'], ['wall', 3, 4]], columns: ['0', '1', '2'], labels: ['0', '1', '2'], selected: [[2, 2]], blocked: [[1, 2], [2, 0]] });
coding('same-tree', 'matrix', 'equal values in different child positions are not the same tree', 'Both trees visit values 1 then 2 in preorder without null markers. Their left and right shapes differ, so structural equality is false.', { values: [[1, 2, 'null'], [1, 'null', 2]], columns: ['root', 'left', 'right'], labels: ['tree A', 'tree B'], blocked: [[0, 1], [0, 2], [1, 1], [1, 2]] });
coding('search-in-rotated-sorted-array', 'ledger', 'identify the sorted half before testing whether it contains the target', 'For target 0, the sorted half (4,5,6,7) is ineligible. Keep opposite half (0,1,2).', { rows: [['Rotated input; middle value 7', ['4', '5', '6', '7', '0', '1', '2'], 'input'], ['Retained candidate half', ['0', '1', '2'], 'output']] });
coding('serialize-and-deserialize-binary-tree', 'ledger', 'null markers preserve the child positions that values alone lose', 'A left child and a right child can share value preorder (1,2). Their null-marked encodings differ and reconstruct distinct shapes.', { rows: [['Root 1 with left child 2', ['1', '2', '#', '#', '#'], 'input'], ['Root 1 with right child 2', ['1', '#', '2', '#', '#'], 'output']] });
coding('set-matrix-zeroes', 'matrix', 'new zeros must not create new row or column obligations', 'The only original zero is at row 0, column 1. Final row 1 stays (1,0,1), not all zeros.', { values: [[1, 0, 1], [1, 1, 1], [0, 0, 0], [1, 0, 1]], columns: ['0', '1', '2'], labels: ['before 0', 'before 1', 'after 0', 'after 1'], selected: [[0, 1], [3, 1]] });
coding('spiral-matrix', 'ledger', 'a crossed boundary prevents revisiting a one-row remainder', 'For a single row (1,2,3), the top traversal consumes every cell. The bottom traversal must be skipped after top passes bottom.', { rows: [['One-row input', ['1', '2', '3'], 'input'], ['After traversing top', ['top=1', 'bottom=0'], 'state'], ['Bottom traversal', ['skip'], 'output']] });
coding('split-array-largest-sum', 'bars', 'the capacity predicate becomes feasible at the correct boundary', 'For (7,2,5,10,8), capacity 17 needs three groups; capacity 18 needs two. With k=2, choose the first feasible capacity, 18.', { rows: [['Capacity 17: greedy group count', 3, 'warning'], ['Capacity 18: greedy group count', 2, 'output']], limit: 2, limitLabel: 'allowed', unit: ' groups' });
coding('stable-softmax', 'parts', 'subtracting the maximum preserves probabilities while preventing overflow', 'Logits (1000,1001) shift to (-1,0). Their softmax probabilities are about 0.269 and 0.731 without exponentiating 1000.', { rows: [['class 0 after common shift', 1 / (1 + Math.E), 'input'], ['class 1 after common shift', Math.E / (1 + Math.E), 'output']], unit: ' probability mass' });
coding('subarray-sum-equals-k', 'ledger', 'query the old prefix frequency before inserting the current prefix', 'For input (0) and target 0, the seeded empty prefix contributes one matching subarray. Inserting the current prefix first would incorrectly count two.', { rows: [['Frequency before processing the zero', ['prefix 0: 1'], 'input'], ['Query old frequency', ['answer += 1'], 'state'], ['Then insert this prefix', ['prefix 0: 2'], 'output']] });
coding('subsets', 'matrix', 'each position supplies one independent include-or-exclude choice', 'For values (1,2), four binary choice paths produce four distinct subsets. Excluding a value is a branch, not an omitted decision.', { values: [['no', 'no', '{}'], ['no', 'yes', '{2}'], ['yes', 'no', '{1}'], ['yes', 'yes', '{1,2}']], columns: ['take 1', 'take 2', 'subset'], labels: ['00', '01', '10', '11'], selected: [[0, 2], [1, 2], [2, 2], [3, 2]] });
coding('subtree-of-another-tree', 'matrix', 'equal candidate roots still require a complete local tree match', 'Both candidate roots store 5 and have left child 3. An extra right child 7 means the larger local tree is not equal to the proposed subtree.', { values: [[5, 3, 'null'], [5, 3, 7]], columns: ['root', 'left', 'right'], labels: ['wanted', 'candidate'], blocked: [[1, 2]] });
coding('sum-of-two-integers', 'matrix', 'simultaneous XOR and shifted-AND updates preserve the total', 'For 5+3, provisional sum and carry remain separate until carry is zero. Each row still totals 8 within the fixed-width word.', { values: [[5, 3], [6, 2], [4, 4], [0, 8], [8, 0]], columns: ['sum bits', 'carry bits'], labels: ['input', 'pass 1', 'pass 2', 'pass 3', 'pass 4'], selected: [[4, 0], [4, 1]] });
coding('top-k-frequent-elements', 'ledger', 'evict by frequency, not by the value itself', 'For k=2, frequencies are 9:3, 1:5, and 8:4. Keep values 1 and 8; the smallest retained frequency is 4.', { rows: [['Value:frequency observations', ['9:3', '1:5', '8:4'], 'input'], ['Retained top two by frequency', ['1:5', '8:4'], 'output']] });
coding('top-k-scores', 'ledger', 'partition selects membership; it does not sort the selected candidates', 'Candidate indices (3,1) have correct membership but no score order. Sorting that subset returns indices (1,3).', { rows: [['Original score order', ['.2', '.9', '.4', '.8'], 'input'], ['One permitted unordered top-2 subset', ['3: .8', '1: .9'], 'state'], ['Descending-by-score original indices', ['1', '3'], 'output']] });
coding('two-sum', 'ledger', 'lookup before insertion prevents pairing an item with itself', 'For target 6 and input (3,3), the first 3 cannot match itself. The second 3 finds the earlier index 0 and returns distinct indices.', { rows: [['First item at index 0', ['need 3', 'not seen', 'store 3:0'], 'input'], ['Second item at index 1', ['need 3', 'found at 0', 'return 0,1'], 'output']] });
coding('unique-paths', 'parts', 'the final move partitions paths by their incoming neighbor', 'In a 3-by-3 grid, three paths arrive from above and three from the left. The disjoint final-move groups sum to six.', { rows: [['arrive from above', 3, 'input'], ['arrive from the left', 3, 'output']], unit: ' paths' });
coding('valid-anagram', 'matrix', 'equal length does not guarantee equal multiplicities', 'Strings aab and abb both have length 3. Their character counts differ, so they are not anagrams.', { values: [[2, 1], [1, 2]], columns: ['a count', 'b count'], labels: ['aab', 'abb'], blocked: [[1, 0], [1, 1]] });
coding('valid-palindrome', 'ledger', 'skip non-alphanumeric characters before comparing the ends', 'For A,b:a, normalization gives aba. The punctuation never participates in an equality comparison; the retained letters match from both ends.', { rows: [['Input characters', ['A', ',', 'b', ':', 'a'], 'input'], ['Characters actually compared', ['a', 'b', 'a'], 'output']] });
coding('valid-parentheses', 'ledger', 'a closing bracket must match the most recent opener', 'For ([)], the ) encounters [ on top. A matching ( deeper in the stack does not make the sequence valid.', { rows: [['Openers seen so far; top at right', ['(', '['], 'input'], ['Incoming closer', [')'], 'warning'], ['Required top opener versus actual', ['(', '[', 'REJECT'], 'warning']] });
coding('validate-binary-search-tree', 'matrix', 'a descendant must satisfy every inherited ancestor bound', 'Node 12 fits above parent 5 but violates ancestor 10. Every descendant in that ancestor\'s left subtree must remain below 10.', { values: [[5, 12, 'yes'], [10, 12, 'NO']], columns: ['bound', 'node', 'fits'], labels: ['parent low', 'ancestor high'], selected: [[0, 2]], blocked: [[1, 2]] });
coding('word-break', 'matrix', 'a tempting longest prefix can leave an impossible suffix', 'For cars and dictionary (car,ca,rs), greedy car leaves invalid s. The valid split is ca plus rs; reachability must consider both boundaries.', { values: [['car', 's', 'NO'], ['ca', 'rs', 'yes']], columns: ['prefix', 'suffix', 'valid'], labels: ['greedy', 'reachable'], selected: [[1, 2]], blocked: [[0, 2]] });
coding('word-search-ii', 'matrix', 'a terminal trie node may still have another word below it', 'Emit a at its terminal node, then keep exploring. For adjacent cells a,b and words (a,ab), continuing finds ab.', { values: [['a', 'yes', 'b'], ['ab', 'yes', 'none']], columns: ['prefix', 'emit', 'child'], labels: ['step 1', 'step 2'], selected: [[0, 1], [1, 1]] });
coding('word-search', 'ledger', 'a matching letter cannot reuse a cell already on this path', 'On a two-cell board A,B, target ABA would require cell 0 twice. The final A matches in value but is unavailable on the active path.', { rows: [['Requested cell sequence', ['0:A', '1:B', '0:A'], 'input'], ['Availability within this path', ['new', 'new', 'REJECT'], 'warning']] });

const ledgerCompositions = {
  'contextual-bandit-partial-feedback-support': 'compare',
  'continuous-batching-reuses-finished-slot': 'transform',
  'decision-threshold-action-regions': 'compare',
  'reasoning-admission-and-capacity': 'compare',
  'backprop-forward-reverse-trace': 'arithmetic',
  'cnn-spatial-channel-pyramid': 'transform',
  'logistic-gradient-cancellation': 'arithmetic',
  'dropout-inverted-expectation': 'arithmetic',
  'spectral-theorem-change-basis': 'transform',
  'exponential-family-sufficient-stat-moment-match': 'arithmetic',
  'fm-shared-embeddings-unseen-pair': 'arithmetic',
  'gradient-accumulation-state-boundary': 'transform',
  'gradient-boosting-residual-update': 'arithmetic',
  'logit-odds-probability-map': 'compare',
  'kernel-trick-exact-shortcut': 'arithmetic',
  'lstm-coordinate-gated-state-update': 'arithmetic',
  'mf-missing-rating-dot-product': 'arithmetic',
  'importance-weights-effective-sample-size': 'arithmetic',
  'naive-bayes-evidence-ledger': 'arithmetic',
  'perplexity-tokenizer-denominator': 'sequence',
  'position-bias-inverse-examination-weight': 'arithmetic',
  'distribution-observation-unit': 'compare',
  'residual-add-gradient-split': 'arithmetic',
  'potential-shaping-cancels-backtracking': 'arithmetic',
  'z-loss-anchors-common-offset': 'transform',
  'gae-residual-weight-trace': 'arithmetic',
  'belief-propagation-branch-summary': 'arithmetic',
  'bootstrap-resampling-unit': 'set',
  'ctc-blank-preserves-repeat': 'sequence',
  'content-profile-scores-new-item': 'arithmetic',
  'ring-attention-fixed-query-kv-cycle': 'arithmetic',
  'infonce-positive-candidate-set': 'nested',
  'diffusion-training-sampling-asymmetry': 'compare',
  'bandit-three-selection-rules': 'compare',
  'gqa-eight-queries-two-kv-groups': 'compare',
  'lora-zero-impact-first-gradient': 'arithmetic',
  'autograd-shared-node-accumulate-then-propagate': 'arithmetic',
  'kmeans-assign-then-update': 'transform',
  'kge-antisymmetry-score-swap': 'arithmetic',
  'forward-backward-viterbi-chain-objectives': 'compare',
  'batched-top-k-running-invariant': 'transform',
  'linear-attention-two-escape-routes': 'compare',
  'fp16-loss-scaling-ledger': 'compare',
  'shap-all-orderings-credit': 'compare',
  'positional-relative-mechanisms': 'compare',
  'q-learning-off-policy-backup': 'arithmetic',
  'quantization-scale-granularity': 'transform',
  'ranking-metrics-per-rank-credit': 'compare',
  'reparameterization-gradient-path': 'arithmetic',
  'resnet-shortcut-shape-contract': 'compare',
  'grpo-group-relative-signal': 'transform',
  'semantic-segmentation-boundary-fusion': 'compare',
  'softmax-ce-logit-update': 'transform',
  'svm-support-vectors-set-boundary': 'compare',
  'bpe-corpus-merge-trace': 'sequence',
  'nonlinear-embedding-local-vs-global': 'compare',
  'vision-patches-to-positioned-tokens': 'sequence',
  'chatbot-randomization-unit': 'compare',
  'hierarchical-all-reduce-path': 'transform',
  'ring-all-reduce-chunk-trace': 'transform',
  'asr-alignment-decoder-choices': 'transform',
  'autoregressive-diffusion-sampling-steps': 'compare',
  'fused-moe-triton-sram-roofline': 'compare',
  'coding-interview-method-state': 'arithmetic',
  'content-moderation-governed-decision-loop': 'compare',
  'paper-critique-prioritization-funnel': 'compare',
  'policy-label-observation-loop': 'compare',
  'llm-inference-two-budget-scheduler': 'compare',
  'reasoning-fixed-budget-reclaim': 'transform',
  'flashattention-sram-tile-stream': 'transform',
  'flashattention-standard-hbm-path': 'compare',
  'fsdp-wrapped-unit-lifecycle': 'transform',
  'llm-cost-compounds-across-agent-calls': 'arithmetic',
  'paged-attention-block-table-indirection': 'physical-map',
  'plan-70b-parallel-accounting': 'arithmetic',
  'prefill-decode-weight-reuse': 'compare',
  'rag-index-request-paths': 'transform',
  'speculative-decoding-first-rejection-boundary': 'transform',
  'streaming-asr-provisional-to-final': 'transform',
  'tensor-parallel-ffn-handoff': 'arithmetic',
  'course-prerequisite-fanout': 'transform',
  'subset-sum-old-snapshot': 'transform',
  'subsets-increasing-index-tree': 'compare',
  'balanced-binary-tree-state': 'transform',
  'best-time-to-buy-and-sell-stock-state': 'compare',
  'binary-search-state': 'transform',
  'binary-tree-level-order-traversal-state': 'transform',
  'binary-tree-maximum-path-sum-state': 'compare',
  'coin-change-state': 'compare',
  'combination-sum-state': 'compare',
  'construct-tree-from-preorder-and-inorder-traversal-state': 'compare',
  'container-with-most-water-state': 'arithmetic',
  'contains-duplicate-state': 'set',
  'counting-bits-state': 'arithmetic',
  'course-schedule-ii-state': 'transform',
  'cross-entropy-from-logits-state': 'arithmetic',
  'daily-temperatures-state': 'compare',
  'decode-string-state': 'sequence',
  'edit-distance-state': 'compare',
  'encode-and-decode-strings-state': 'sequence',
  'find-minimum-in-rotated-sorted-array-state': 'transform',
  'group-anagrams-state': 'set',
  'insert-interval-state': 'transform',
  'jump-game-state': 'compare',
  'kth-largest-element-state': 'set',
  'kth-smallest-element-in-a-bst-state': 'compare',
  'largest-rectangle-in-histogram-state': 'compare',
  'longest-consecutive-sequence-state': 'set',
  'longest-increasing-path-in-a-matrix-state': 'compare',
  'longest-increasing-subsequence-state': 'transform',
  'longest-repeating-character-replacement-state': 'compare',
  'lru-cache-state': 'sequence',
  'maximum-depth-of-binary-tree-state': 'arithmetic',
  'maximum-product-subarray-state': 'arithmetic',
  'maximum-subarray-state': 'compare',
  'merge-intervals-state': 'transform',
  'merge-k-sorted-lists-state': 'transform',
  'merge-two-sorted-lists-state': 'sequence',
  'min-stack-state': 'transform',
  'mini-batches-state': 'sequence',
  'missing-number-state': 'set',
  'number-of-1-bits-state': 'transform',
  'number-of-connected-components-state': 'transform',
  'pairwise-squared-distances-state': 'arithmetic',
  'permutations-state': 'set',
  'product-of-array-except-self-state': 'compare',
  'redundant-connection-state': 'set',
  'remove-nth-node-from-end-state': 'sequence',
  'reorder-list-state': 'sequence',
  'reverse-bits-state': 'sequence',
  'reverse-linked-list-state': 'sequence',
  'search-in-rotated-sorted-array-state': 'transform',
  'serialize-and-deserialize-binary-tree-state': 'sequence',
  'spiral-matrix-state': 'transform',
  'subarray-sum-equals-k-state': 'transform',
  'top-k-frequent-elements-state': 'set',
  'top-k-scores-state': 'transform',
  'two-sum-state': 'transform',
  'valid-palindrome-state': 'sequence',
  'valid-parentheses-state': 'compare',
  'word-search-state': 'sequence',
};

const curveLandmarks = {
  'activation-gradient-regions': -3,
  'optimizer-sharpness-reparameterization': .5,
  'capacity-regimes-double-descent': 4,
  'clt-data-vs-sample-means': 1,
  'cold-start-signal-handoff': .6,
  'cross-entropy-confident-wrong-gradient': .05,
  'discrete-gradients-forward-backward-paths': -1,
  'gradient-magnitude-through-depth': 10,
  'warmup-cosine-three-checkpoints': 6,
  'mle-bernoulli-parameter-sweep': .6,
  'scaling-fixed-compute-allocation': 2,
  'positive-definite-boundedness': 2,
  'random-forest-correlation-floor': 100,
  'bm25-saturation-length-penalty': 2,
  'relu-second-moment-through-depth': 8,
  'checkpoint-boundary-recompute-balance': 4,
  'em-lower-bound-climb': .5,
  'gp-conditioning-local-uncertainty': 0,
  'learning-rate-range-to-schedule': 3,
  'ppo-clipped-surrogate-plateaus': 1.5,
  'momentum-alternating-components': 4,
  'speaker-verification-threshold-tradeoff': .2,
  'uat-existence-error-band': .125,
};

const pointCompositions = new Set(['adversarial-threat-set-boundary', 'estimator-error-reference-points', 'uncertainty-within-between-models', 'l1-l2-first-contact-geometry', 'word-analogy-shared-displacement']);
for (const [identity, model] of Object.entries(visualArguments)) {
  if (model.kind === 'ledger') model.composition = ledgerCompositions[identity];
  else if (model.kind === 'curve') {
    model.composition = 'landmark';
    model.landmark = curveLandmarks[identity];
    for (const [, points] of model.series) {
      const operation = sampledOperations.get(points);
      if (operation && model.landmark >= points[0][0] && model.landmark <= points.at(-1)[0] && !points.some(([value]) => Math.abs(value - model.landmark) < 1e-12)) {
        points.push([model.landmark, operation(model.landmark)]);
        points.sort((left, right) => left[0] - right[0]);
      }
    }
  }
  else if (model.kind === 'geometry') model.composition = ['ols-orthogonal-projection', 'pca-rank-one-projection'].includes(identity) ? 'projection' : pointCompositions.has(identity) ? 'points' : 'vectors';
  else if (identity === 'ml-role-activity-profiles') model.composition = 'budget-rows';
  else if (identity === 'fixed-capacity-portfolio-guardrails') model.composition = 'budget-transfer';
  else if (['confusion-matrix-denominators', 'binary-precision-and-recall-state', 'roc-pr-same-operating-point'].includes(identity)) model.composition = 'denominators';
  else model.composition = { bars: 'endpoint', parts: 'whole-and-share', population: 'counted-cohorts', matrix: 'cells', timeline: 'shared-clock', connections: 'directed-routes', areas: 'conserved-mass', intervals: 'repeat-versus-condition', suppression: 'overlapping-candidates', topology: 'counterexample-graph' }[model.kind];
}

export function renderVisualArgument(identity) {
  const model = visualArguments[identity];
  assert(model, `No visual argument for ${identity}`);
  const drawing = renderers[model.kind](model);
  assert(drawing.height > 0 && !/NaN|undefined|Infinity/.test(drawing.body), identity);
  return `<div class="visual-argument" data-visual-argument="${identity}" data-argument-kind="${model.kind}" data-argument-composition="${model.composition}"><svg viewBox="0 0 360 ${drawing.height}" role="img" aria-label="${escape(`${model.premise}. ${model.conclusion}`)}"><title>${escape(model.premise)}</title><desc>${escape(model.conclusion)}</desc>${drawing.body}</svg></div>`;
}

export function renderArgumentCaption(identity) {
  return `<strong>Read it this way:</strong> <span data-proof-caption="${identity}">${escape(visualArguments[identity].conclusion)}</span>`;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const { parseFragment } = await import('parse5');
  assert.deepEqual(Object.keys(visualArguments).sort(), Object.keys(figureTakeaways).sort(), 'Every exact illustration identity needs an authored composition');
  assert.equal(Object.keys(visualArguments).length, 408);
  assert.equal(Object.values(visualArguments).filter(model => model.coding).length, Object.keys(codingQuestionVisuals).length);
  assert.deepEqual(Object.keys(ledgerCompositions).sort(), Object.keys(visualArguments).filter(identity => visualArguments[identity].kind === 'ledger').sort(), 'Every ledger needs an individually selected composition');
  assert.deepEqual(Object.keys(curveLandmarks).sort(), Object.keys(visualArguments).filter(identity => visualArguments[identity].kind === 'curve').sort(), 'Every curve needs an individually selected observation');
  for (const [identity, model] of Object.entries(visualArguments)) {
    assert(model.composition, `${identity}: explicit visual composition required`);
    const rendered = renderVisualArgument(identity);
    assert.equal(rendered, renderVisualArgument(identity), `${identity}: nondeterminism`);
    assert(!/data-frame-key|data-motion-key|<script/.test(rendered));
    const nodes = [];
    function walk(node) { nodes.push(node); for (const child of node.childNodes ?? []) walk(child); }
    walk(parseFragment(rendered));
    assert.equal(nodes.filter((node) => node.nodeName === 'svg').length, 1);
    assert(nodes.some((node) => node.nodeName === 'title'));
    assert(nodes.some((node) => node.nodeName === 'desc'));
    if (identity === 'checkpoint-boundary-recompute-balance') assert.deepEqual(nodes.filter(node => node.attrs?.some(attribute => attribute.name === 'data-proof-observation')).map(node => Number(node.attrs.find(attribute => attribute.name === 'data-proof-y').value)), [8, 4, 4], 'The drawn checkpoint values must be exactly eight total, four saved, and four rebuilt');
    if (model.composition === 'budget-transfer') {
      assert.equal(model.rows.reduce((sum, row) => sum + row[1], 0), 100);
      assert.deepEqual(model.transfer, { from: 1, to: 0, value: 6, protected: [2, 3] });
      assert(model.rows[model.transfer.from][1] >= model.transfer.value);
    }
    if (model.composition === 'budget-rows') for (const row of model.values) assert.equal(row.reduce((sum, value) => sum + value, 0), 100);
    if (model.kind === 'suppression') {
      const overlap = (first, second) => Math.max(0, Math.min(first.left + first.width, second.left + second.width) - Math.max(first.left, second.left));
      const iou = (first, second) => overlap(first, second) / (first.width + second.width - overlap(first, second));
      assert(iou(model.boxes[0], model.boxes[1]) > model.threshold && iou(model.boxes[1], model.boxes[2]) > model.threshold && iou(model.boxes[0], model.boxes[2]) <= model.threshold, 'The actual drawn boxes must realize the non-transitive suppression example');
    }
    if (model.kind === 'intervals') {
      assert.equal(model.intervals.filter(([lower, upper]) => lower <= .5 && upper >= .5).length, 9);
      assert(model.intervals.every(([lower, upper]) => lower >= 0 && upper <= 1 && lower < upper));
      assert.equal(nodes.filter(node => node.attrs?.some(attribute => attribute.name === 'data-proof-interval')).length, 10);
    }
    if (model.kind === 'topology') {
      assert.equal(model.edges.length, model.nodes.length - 1);
      assert(model.edges.every(edge => !edge.includes(3)));
      assert.equal(nodes.filter(node => node.attrs?.some(attribute => attribute.name === 'data-proof-topology-edge')).length, model.edges.length);
    }
    if (model.emptySlots) {
      const empty = nodes.filter(node => node.attrs?.some(attribute => attribute.name === 'data-proof-token' && attribute.value === '-'));
      assert.equal(empty.length, model.rows.flatMap(row => row[1]).filter(value => value === '-').length);
      assert(empty.every(node => node.childNodes.some(child => child.nodeName === 'rect' && child.attrs.some(attribute => attribute.name === 'style' && attribute.value.includes('stroke-dasharray')))), `${identity}: free capacity must retain its visible slot boundaries`);
    }
    if (identity === 'policy-label-observation-loop') {
      const label = nodes.find(node => node.attrs?.some(attribute => attribute.name === 'data-proof-token' && attribute.value === 'censored'));
      assert.equal(label.childNodes.find(node => node.nodeName === 'text').childNodes.length, 1, 'The censored state must remain an intact word');
    }
    for (const node of nodes.filter(node => node.attrs?.some(attribute => attribute.name === 'data-proof-cell'))) {
      const attributes = Object.fromEntries(node.attrs.map(attribute => [attribute.name, attribute.value]));
      const [row, column] = attributes['data-proof-cell'].split(',').map(Number);
      if (typeof model.values[row][column] === 'number') assert.equal(node.parentNode.attrs.find(attribute => attribute.name === 'data-proof-status').value, 'value', `${identity}: a real observed number must never be shown as rejected or missing`);
    }
    for (const node of nodes.filter(node => node.attrs?.some(attribute => attribute.name === 'data-proof-denominator'))) {
      const attributes = Object.fromEntries(node.attrs.map(attribute => [attribute.name, attribute.value]));
      assert.equal(Number(attributes.width), 320 * Number(attributes['data-proof-numerator']) / Number(attributes['data-proof-denominator']), `${identity}: denominator changes the drawn share`);
    }
    if (model.kind === 'population') assert.equal(nodes.filter((node) => node.attrs?.some((attribute) => attribute.name === 'data-proof-population')).length, model.rows.reduce((sum, row) => sum + row[1], 0));
    if (model.kind === 'connections') {
      const edges = nodes.filter(node => node.attrs?.some(attribute => attribute.name === 'data-proof-edge'));
      assert.equal(edges.length, model.edges.length, `${identity}: every declared edge is drawn`);
      assert.deepEqual(edges.map(node => node.attrs.find(attribute => attribute.name === 'data-proof-edge').value), model.edges.map(([from, to]) => `${from},${to}`), `${identity}: exact edge endpoints`);
    }
    if (model.kind === 'bars' || model.kind === 'parts' || model.kind === 'population') assert((model.groups?.flatMap(group => group.rows) ?? model.rows).every(row => Number.isFinite(row[1])), `${identity}: finite quantities`);
    if (identity === 'kl-direction-candidate-ranking') {
      assert(model.groups[0].rows[0][1] < model.groups[0].rows[1][1] && model.groups[1].rows[0][1] > model.groups[1].rows[1][1], 'The drawn KL comparison must actually reverse the candidate ranking');
      assert(Math.abs(model.groups[0].rows[0][1] - .1163217565860046) < 1e-12 && Math.abs(model.groups[1].rows[1][1] - .07133122707634103) < 1e-12, 'Worked KL values match independent arithmetic');
    }
    if (model.kind === 'areas') for (const [index, region] of model.regions.entries()) {
      assert(region.width > 0 && region.height > 0 && Number.isFinite(region.mass), `${identity}: valid area and mass`);
      const group = nodes.find(node => node.attrs?.some(attribute => attribute.name === 'data-proof-region' && attribute.value === String(index)));
      const rectangle = group.childNodes.find(node => node.nodeName === 'rect');
      const attributes = Object.fromEntries(rectangle.attrs.map(attribute => [attribute.name, attribute.value]));
      assert.equal(Number(attributes.width) * Number(attributes.height), region.width * region.height * Number(attributes['data-proof-scale']) ** 2, `${identity}: drawn area uses one equal scale`);
      assert.equal(group.childNodes.filter(node => node.nodeName === 'circle').length * model.unitMass, region.mass, `${identity}: countable marks preserve exact mass`);
    }
    if (model.kind === 'bars') for (const node of nodes.filter(node => node.attrs?.some(attribute => attribute.name === 'data-proof-min'))) {
      const attributes = Object.fromEntries(node.attrs.map(attribute => [attribute.name, attribute.value]));
      const span = Number(attributes['data-proof-max']) - Number(attributes['data-proof-min']);
      assert(Math.abs(Number(attributes.width) - Math.abs(Number(attributes['data-proof-value'])) / span * 304) < 1e-8, `${identity}: drawn bar length matches its declared scale`);
      assert(Math.abs(Number(attributes['data-proof-zero']) - (28 - Number(attributes['data-proof-min']) / span * 304)) < 1e-8, `${identity}: zero origin matches its scale`);
    }
    if (model.kind === 'parts' || model.kind === 'population') assert(model.rows.every(row => row[1] >= 0), `${identity}: nonnegative parts`);
    if (model.kind === 'matrix') for (const [row, column] of [...model.selected ?? [], ...model.blocked ?? []]) assert(row >= 0 && row < model.values.length && column >= 0 && column < model.values[0].length, `${identity}: selected cell exists`);
    if (model.kind === 'curve') for (const [, points] of model.series) assert(points.every(([horizontal, vertical]) => horizontal >= model.domain[0] - 1e-9 && horizontal <= model.domain[1] + 1e-9 && vertical >= model.range[0] - 1e-9 && vertical <= model.range[1] + 1e-9), `${identity}: curve stays inside declared numeric axes`);
  }
  const weights = visualArguments['importance-weights-effective-sample-size'].rows[0][1].map(Number);
  assert.equal(weights.length, 5);
  assert.equal(weights.reduce((sum, value) => sum + value, 0) ** 2 / weights.reduce((sum, value) => sum + value ** 2, 0), Number(visualArguments['importance-weights-effective-sample-size'].rows[1][1].at(-1)));
  const [original, clipped] = visualArguments['global-norm-clipping-radial-projection'].vectors.map(vector => vector[1]);
  assert(Math.abs(Math.hypot(...clipped) - 2) < 1e-12);
  assert(Math.abs(original[0] * clipped[1] - original[1] * clipped[0]) < 1e-12);
  const jacobian = visualArguments['jacobian-transpose-pullback'].values;
  for (let column = 0; column < 2; column++) assert.equal(jacobian[2][column], jacobian.slice(0, 2).reduce((sum, row) => sum + row[column].split('x').map(Number).reduce((product, value) => product * value, 1), 0));
  const hours = speed => [3, 6, 7, 11].reduce((sum, value) => sum + Math.ceil(value / speed), 0);
  assert.deepEqual(visualArguments['koko-eating-bananas-state'].rows.map(row => row[1]), [hours(3), hours(4)]);
  const groups = limit => {
    let count = 1;
    let sum = 0;
    for (const value of [7, 2, 5, 10, 8]) {
      if (sum + value > limit) { count++; sum = 0; }
      sum += value;
    }
    return count;
  };
  assert.deepEqual(visualArguments['split-array-largest-sum-state'].rows.map(row => row[1]), [groups(17), groups(18)]);
  const addition = visualArguments['sum-of-two-integers-state'].values;
  for (let index = 1; index < addition.length; index++) {
    const [sum, carry] = addition[index - 1];
    assert.deepEqual(addition[index], [sum ^ carry, (sum & carry) << 1]);
  }
  const products = visualArguments['product-of-array-except-self-state'].rows.map(row => row[1].map(Number));
  assert.deepEqual(products[3], products[0].map((value, index, input) => input.reduce((product, item, position) => product * (index === position ? 1 : item), 1)));
  assert.deepEqual(products[3], products[1].map((value, index) => value * products[2][index]));
  const subsequences = Array.from({ length: 3 }, () => Array(4).fill(0));
  for (let row = 1; row <= 2; row++) for (let column = 1; column <= 3; column++) subsequences[row][column] = 'AB'[row - 1] === 'ACB'[column - 1] ? subsequences[row - 1][column - 1] + 1 : Math.max(subsequences[row - 1][column], subsequences[row][column - 1]);
  assert.deepEqual(visualArguments['longest-common-subsequence-state'].values, subsequences);
  console.log(`Visual arguments: ${Object.keys(visualArguments).length} original compositions; exact inventory, numeric axes, worked arithmetic, deterministic geometry, population counts, accessible descriptions, and identity isolation passed.`);
}