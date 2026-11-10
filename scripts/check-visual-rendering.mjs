import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';
import process from 'node:process';
import { fromMarkdown } from 'mdast-util-from-markdown';
import { parseFragment, serialize } from 'parse5';
import { renderVisualArgument } from './illustration-arguments.mjs';
import sharp from 'sharp';

const root = resolve(import.meta.dirname, '..');

function svgLabelIsContained(svg, text, epsilon) {
  const box = text.getBBox();
  if (!box.width || !box.height) return true;
  const matrix = svg.getCTM().inverse().multiply(text.getCTM());
  const view = svg.viewBox.baseVal;
  return [
    new DOMPoint(box.x, box.y),
    new DOMPoint(box.x + box.width, box.y),
    new DOMPoint(box.x, box.y + box.height),
    new DOMPoint(box.x + box.width, box.y + box.height),
  ].map(point => point.matrixTransform(matrix)).every(point =>
    point.x >= view.x - epsilon && point.y >= view.y - epsilon &&
    point.x <= view.x + view.width + epsilon && point.y <= view.y + view.height + epsilon
  );
}

function lessonFitsWorkspace(figure, workspace, epsilon) {
  return figure.getBoundingClientRect().bottom <= workspace.getBoundingClientRect().bottom + epsilon;
}

function option(name, fallback = '') {
  const prefix = `--${name}=`;
  return process.argv.find((arg) => arg.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

function browserExecutable() {
  const configured = option('browser', process.env.CHROME_PATH ?? '');
  const candidates = configured
    ? [configured]
    : process.platform === 'win32'
      ? [
          'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
          'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
          'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
        ]
      : ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'];
  return candidates.find((candidate) => existsSync(candidate));
}

const slugs = option('slugs')
  .split(',')
  .map((slug) => slug.trim())
  .filter(Boolean);
if (slugs.length === 0 || new Set(slugs).size !== slugs.length) {
  throw new Error('--slugs needs a non-empty, duplicate-free comma-separated list');
}

const baseUrl = option('base-url', 'http://127.0.0.1:8123').replace(/\/$/, '');
const outputDir = resolve(root, option('output', 'scratch/visual-render-review'));
const allFrames = process.argv.includes('--all-frames');
const proofCaptures = process.argv.includes('--proof-captures');
const supplemental = process.argv.includes('--supplemental');
const lessonMode = process.argv.includes('--lessons');
const bindDist = process.argv.includes('--bind-dist');
const lessonCatalog = lessonMode ? JSON.parse(readFileSync(join(root, 'data/visual-lessons.json'), 'utf8')).lessons : {};
const auditsDir = join(root, 'data', 'visual-audits');
const auditedEntries = slugs.map((slug) => {
  const auditPath = join(auditsDir, `${slug}.json`);
  if (!existsSync(auditPath)) throw new Error(`Missing audit sidecar for ${slug}`);
  const audit = JSON.parse(readFileSync(auditPath, 'utf8'));
  if (!['implemented', 'no-visual'].includes(audit.status)) {
    throw new Error(`${slug} is ${audit.status}, not resolved`);
  }
  if (audit.status === 'no-visual') return { slug, status: audit.status };
  const articlePath = join(root, audit.article);
  const article = readFileSync(articlePath, 'utf8');
  const category = article.match(/^---\n[\s\S]*?^category:\s*["']?([^"'\n]+)["']?\s*$[\s\S]*?^---$/m)?.[1]?.trim();
  if (!category) throw new Error(`Could not read category from ${audit.article}`);
  const supplementalIds = [];
  const supplementalSvgIds = [];
  if (supplemental) {
    const visitHtml = (node) => {
      const id = node.attrs?.find((attribute) => attribute.name === 'data-supplemental-visual')?.value;
      if (id) {
        supplementalIds.push(id);
        if (node.nodeName === 'figure') supplementalSvgIds.push(id);
      }
      for (const child of node.childNodes ?? []) visitHtml(child);
    };
    const visitMarkdown = (node) => {
      if (node.type === 'html') visitHtml(parseFragment(node.value));
      for (const child of node.children ?? []) visitMarkdown(child);
    };
    visitMarkdown(fromMarkdown(article));
    if (supplementalIds.length === 0 || new Set(supplementalIds).size !== supplementalIds.length) {
      throw new Error(`${slug} needs a non-empty, unique supplemental visual list`);
    }
  }
  return {
    slug,
    status: audit.status,
    category,
    medium: supplemental ? supplementalSvgIds.length === supplementalIds.length ? 'svg' : 'mermaid' : audit.medium,
    visualIds: supplemental ? supplementalIds : audit.implementation.visualIds,
    article: basename(audit.article),
    supplemental,
  };
});
const entries = auditedEntries.filter((entry) => entry.status === 'implemented');
const noVisualSlugs = auditedEntries.filter((entry) => entry.status === 'no-visual').map((entry) => entry.slug);
if (entries.length === 0) {
  rmSync(outputDir, { recursive: true, force: true });
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'report.json'), `${JSON.stringify({ renders: [], noVisualSlugs }, null, 2)}\n`);
  console.log(`No implemented visuals to render; verified ${noVisualSlugs.length} no-visual outcome(s)`);
  process.exit(0);
}

const executable = browserExecutable();
if (!executable) throw new Error('Chrome or Edge was not found; set CHROME_PATH or --browser');
if (process.platform !== 'win32' && /^[A-Za-z]:|^\/mnt\/[a-z]\//.test(executable)) {
  throw new Error('Run this script with Windows Node when using a Windows browser so its CDP port is reachable');
}

const allModes = [
  { name: 'desktop-light', width: 1280, height: 900, theme: 'light', media: 'screen' },
  { name: 'desktop-dark', width: 1280, height: 900, theme: 'dark', media: 'screen' },
  { name: 'mobile-light', width: 390, height: 900, theme: 'light', media: 'screen', mobile: true },
  { name: 'mobile-dark', width: 390, height: 900, theme: 'dark', media: 'screen', mobile: true },
  { name: 'print', width: 1280, height: 900, theme: 'light', media: 'print' },
];
const requestedModes = option('modes').split(',').map((name) => name.trim()).filter(Boolean);
if (new Set(requestedModes).size !== requestedModes.length || requestedModes.some((name) => !allModes.some((mode) => mode.name === name))) {
  throw new Error('--modes must contain unique names from desktop-light, desktop-dark, mobile-light, mobile-dark, print');
}
const modes = allModes.filter((mode) => requestedModes.length === 0 || requestedModes.includes(mode.name));

class Cdp {
  constructor(url) {
    this.socket = new WebSocket(url);
    this.nextId = 1;
    this.pending = new Map();
    this.events = new Map();
    this.listeners = new Map();
    this.socket.onmessage = ({ data }) => {
      const message = JSON.parse(data);
      if (message.id) {
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        if (message.error) pending.reject(new Error(`${pending.method}: ${message.error.message}`));
        else pending.resolve(message.result);
        return;
      }
      const listeners = this.events.get(message.method) ?? [];
      this.events.delete(message.method);
      for (const resolveEvent of listeners) resolveEvent(message.params);
      for (const listener of this.listeners.get(message.method) ?? []) listener(message.params);
    };
  }

  async ready() {
    if (this.socket.readyState === WebSocket.OPEN) return;
    await new Promise((resolveReady, rejectReady) => {
      this.socket.onopen = resolveReady;
      this.socket.onerror = rejectReady;
    });
  }

  call(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolveCall, rejectCall) => {
      this.pending.set(id, { method, resolve: resolveCall, reject: rejectCall });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  once(method) {
    return new Promise((resolveEvent) => {
      const listeners = this.events.get(method) ?? [];
      listeners.push(resolveEvent);
      this.events.set(method, listeners);
    });
  }

  on(method, listener) {
    const listeners = this.listeners.get(method) ?? [];
    listeners.push(listener);
    this.listeners.set(method, listeners);
  }

  close() {
    this.socket.close();
  }
}

async function waitForVisualLayout(cdp, visualId, theme, label) {
  const evaluation = await cdp.call('Runtime.evaluate', {
    expression: `(async () => {
      document.documentElement.dataset.theme = ${JSON.stringify(theme)};
      await document.fonts.ready;
      const locateVisual = () => {
        if (${JSON.stringify(supplemental)}) {
          const anchor = document.querySelector(${JSON.stringify(`[data-supplemental-visual="${visualId}"]`)});
          return anchor?.matches('figure.learning-figure') ? anchor : anchor?.nextElementSibling;
        }
        const comments = document.createTreeWalker(document, NodeFilter.SHOW_COMMENT);
        let marker;
        while (comments.nextNode()) {
          if (comments.currentNode.data.trim() === ${JSON.stringify(`visual:${visualId}`)}) {
            marker = comments.currentNode;
            break;
          }
        }
        let visual = marker?.nextElementSibling;
        while (visual && !visual.matches('figure.learning-figure, .mermaid')) visual = visual.nextElementSibling;
        return visual;
      };
      let previous = '';
      let previousSvg = null;
      let stableSince = performance.now();
      for (let attempt = 0; attempt < 40; attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, 50));
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        window.scrollTo({ left: 0, top: 0, behavior: 'instant' });
        const visual = locateVisual();
        if (!visual) throw new Error(${JSON.stringify(`${label}: visual not found`)});
        const svg = visual.querySelector('svg');
        if (visual.matches('.mermaid') && (!svg || !svg.querySelector('text, foreignObject'))) {
          previous = '';
          previousSvg = null;
          stableSince = performance.now();
          continue;
        }
        const rect = visual.getBoundingClientRect();
        const signature = [
          rect.left + scrollX,
          rect.top + scrollY,
          rect.width,
          rect.height,
          document.documentElement.scrollWidth,
          document.documentElement.scrollHeight,
        ].map((value) => Math.round(value * 100) / 100).join('|');
        const now = performance.now();
        if (signature !== previous || svg !== previousSvg) {
          previous = signature;
          previousSvg = svg;
          stableSince = now;
        } else if (now - stableSince >= 500) {
          return true;
        }
      }
      throw new Error('Visual layout did not stabilize');
    })()`,
    returnByValue: true,
    awaitPromise: true,
  });
  if (evaluation.exceptionDetails) {
    throw new Error(evaluation.exceptionDetails.exception?.description ?? evaluation.exceptionDetails.text);
  }
}

const delay = (milliseconds) => new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
let port;
const profileDir = join(tmpdir(), `mlmentorship-visual-review-${process.pid}`);
rmSync(profileDir, { recursive: true, force: true });
mkdirSync(profileDir, { recursive: true });
rmSync(outputDir, { recursive: true, force: true });
mkdirSync(outputDir, { recursive: true });

const browser = spawn(executable, [
  '--headless=new',
  '--disable-gpu',
  '--no-first-run',
  '--no-default-browser-check',
  '--remote-debugging-port=0',
  `--user-data-dir=${profileDir}`,
  'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
let browserErrors = '';
browser.stderr.on('data', (chunk) => { browserErrors += chunk.toString(); });

async function waitForBrowser() {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (browser.exitCode !== null) throw new Error(`Browser exited early: ${browserErrors.slice(-1000)}`);
    try {
      const [ownedPort, endpoint] = readFileSync(join(profileDir, 'DevToolsActivePort'), 'utf8').trim().split(/\r?\n/);
      port = Number(ownedPort);
      if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid owned debugging port');
      const response = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (response.ok && (await response.json()).webSocketDebuggerUrl.endsWith(endpoint)) return;
    } catch {
      // Chrome has not opened its debugging socket yet.
    }
    await delay(100);
  }
  throw new Error(`Browser debugging endpoint did not start: ${browserErrors.slice(-1000)}`);
}

async function stopBrowser() {
  if (process.platform === 'win32' && browser.pid) {
    spawnSync('taskkill', ['/PID', String(browser.pid), '/T', '/F'], { stdio: 'ignore' });
  } else if (browser.exitCode === null) {
    browser.kill('SIGTERM');
  }
  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      rmSync(profileDir, { recursive: true, force: true });
      return;
    } catch (error) {
      if (attempt === 19) {
        console.warn(`Could not remove temporary browser profile ${profileDir}: ${error.message}`);
        return;
      }
      await delay(100);
    }
  }
}

const report = [];
const articleResponses = new Map();
function sourceSnapshot() {
  const listing = spawnSync('git', ['ls-files', '--cached', '--others', '--exclude-standard', '-z'], { cwd: root, encoding: 'utf8' });
  if (listing.status !== 0) throw new Error(`Cannot inventory source: ${listing.stderr}`);
  return Object.fromEntries(listing.stdout.split('\0').filter(filename => /^(?:src\/|public\/|scripts\/|data\/|package(?:-lock)?\.json$|astro\.config\.[^/]+$|tsconfig\.json$)/.test(filename)).sort().map(filename => [filename, createHash('sha256').update(readFileSync(join(root, filename))).digest('hex')]));
}
const frozenSource = lessonMode ? sourceSnapshot() : undefined;
let cdp;
let currentLabel = 'browser startup';
async function checkLesson(entry, mode) {
  const lesson = lessonCatalog[entry.slug];
  if (!lesson) throw new Error(`${entry.slug}: no authored visual lesson`);
  if (mode.name === 'desktop-light' || mode.name === 'mobile-light') {
    await cdp.call('Emulation.setScriptExecutionDisabled', { value: true });
    const articleLoaded = cdp.once('Page.loadEventFired');
    const noJsUrl = `${baseUrl}/${entry.category}/${entry.slug}/?no-js-check=${mode.name}`;
    await cdp.call('Page.navigate', { url: noJsUrl });
    let articleTimer;
    try {
      await Promise.race([articleLoaded, new Promise((_, reject) => { articleTimer = setTimeout(() => reject(new Error('No-JavaScript article did not load')), 20000); })]);
    } finally { clearTimeout(articleTimer); }
    const noJs = await cdp.call('Runtime.evaluate', {
      expression: `(async () => {
        await document.fonts.ready;
        const lead = document.querySelector('[data-article-quick-review]');
        if (!lead || lead.hidden || !lead.getBoundingClientRect().height || lead.querySelector('[data-review-visual-host]').getAttribute('aria-busy') !== 'false') throw new Error('No-JavaScript lead is unavailable');
        const figures = [...document.querySelectorAll('[data-visual-lesson-step]')];
        const order = figures.map(figure => figure.dataset.visualLessonStep);
        if (JSON.stringify(order) !== ${JSON.stringify(JSON.stringify(lesson.steps.map(step => `${step.visualId}:${step.view}`)))}) throw new Error('No-JavaScript lesson order differs');
        if (figures.some(figure => figure.closest('details') || !figure.getClientRects().length)) throw new Error('Essential no-JavaScript figure is hidden');
        if (document.querySelector('[data-coding-enhanced="true"]')) throw new Error('No-JavaScript article ran enhancement');
        if (innerWidth !== ${mode.width} || document.documentElement.scrollWidth > innerWidth + 1) throw new Error('No-JavaScript article overflows its actual viewport');
        return {order, width: innerWidth};
      })()`, returnByValue: true, awaitPromise: true,
    });
    if (noJs.exceptionDetails) throw new Error(noJs.exceptionDetails.exception?.description ?? noJs.exceptionDetails.text);
    const response = articleResponses.get(noJsUrl);
    if (!response || response.status !== 200) throw new Error(`${entry.slug}: no-JavaScript article response is unsuccessful`);
    const delivered = await cdp.call('Network.getResponseBody', { requestId: response.requestId });
    const bytes = Buffer.from(delivered.body, delivered.base64Encoded ? 'base64' : 'utf8');
    if (bindDist && !bytes.equals(readFileSync(join(root, 'dist', entry.category, entry.slug, 'index.html')))) throw new Error(`${entry.slug}: no-JavaScript article differs from built output`);
    appendFileSync(join(outputDir, 'no-js-articles.ndjson'), `${JSON.stringify({slug: entry.slug, mode: mode.name, articleSha256: createHash('sha256').update(bytes).digest('hex'), ...noJs.result.value})}\n`);
    await cdp.call('Emulation.setScriptExecutionDisabled', { value: false });
  }
  const loaded = cdp.once('Page.loadEventFired');
  await cdp.call('Page.navigate', { url: `${baseUrl}/review/?visual-check=${encodeURIComponent(`${entry.slug}-${mode.name}`)}#${entry.slug}` });
  let loadTimer;
  try {
    await Promise.race([loaded, new Promise((_, reject) => { loadTimer = setTimeout(() => reject(new Error('Lesson document did not load')), 20000); })]);
  } finally { clearTimeout(loadTimer); }
  const ready = await cdp.call('Runtime.evaluate', {
    expression: `(async () => {
      document.documentElement.dataset.theme = ${JSON.stringify(mode.theme)};
      await document.fonts.ready;
      const deadline = performance.now() + 20000;
      while (performance.now() < deadline) {
        const host = document.querySelector('[data-review-visual-host]');
        if (host?.getAttribute('aria-busy') === 'false' && host.querySelector('[data-visual-lesson-step]')) return true;
        if (host?.querySelector('.review-error')) throw new Error(host.textContent);
        await new Promise(resolve => requestAnimationFrame(resolve));
      }
      throw new Error('Visual lesson did not become ready');
    })()`, returnByValue: true, awaitPromise: true,
  });
  if (ready.exceptionDetails) throw new Error(ready.exceptionDetails.exception?.description ?? ready.exceptionDetails.text);
  const articleUrl = `${baseUrl}/${entry.category}/${entry.slug}/`;
  const response = articleResponses.get(articleUrl);
  if (!response || response.status !== 200) throw new Error(`${entry.slug}: browser article response is missing or unsuccessful`);
  const delivered = await cdp.call('Network.getResponseBody', { requestId: response.requestId });
  const articleBytes = Buffer.from(delivered.body, delivered.base64Encoded ? 'base64' : 'utf8');
  const articleSha256 = createHash('sha256').update(articleBytes).digest('hex');
  if (bindDist && !articleBytes.equals(readFileSync(join(root, 'dist', entry.category, entry.slug, 'index.html')))) throw new Error(`${entry.slug}: browser-fetched article differs from built output`);
  for (const [index, step] of lesson.steps.entries()) {
    let frameCount = 1;
    let initialFrameHeight;
    for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    currentLabel = `${entry.slug}/${mode.name}/${step.visualId}:${step.view}/frame-${frameIndex}`;
    const inspected = await cdp.call('Runtime.evaluate', {
      expression: `(async () => {
        const host = document.querySelector('[data-review-visual-host]');
        const figures = [...host.querySelectorAll(':scope > [data-visual-lesson-step]')];
        if (${index} > 0 && ${frameIndex} === 0) host.querySelector('[data-visual-lesson-next]').click();
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        const visible = figures.filter(figure => !figure.hidden);
        if (visible.length !== 1) throw new Error('Expected exactly one visible teaching step, got ' + visible.length);
        const figure = visible[0];
        if (${frameIndex} > 0) {
          figure.querySelector('[data-coding-next]').click();
          await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
          figure.getAnimations({subtree: true}).filter(animation => animation.playState === 'running').forEach(animation => animation.finish());
        }
        if (figure.dataset.visualLessonStep !== ${JSON.stringify(`${step.visualId}:${step.view}`)}) throw new Error('Wrong teaching step');
        if (figures.length !== ${lesson.steps.length}) throw new Error('Incomplete lesson');
        if (innerWidth !== ${mode.width}) throw new Error('Actual viewport differs from requested viewport: ' + innerWidth);
        if (document.querySelector('[data-review-explanation]').getAttribute('aria-hidden') !== 'true') throw new Error('Explanation opened without request');
        const ids = [...host.querySelectorAll('[id]')].map(element => element.id);
        if (new Set(ids).size !== ids.length) throw new Error('Duplicate cloned IDs');
        for (const element of host.querySelectorAll('[aria-labelledby], [aria-describedby], [aria-controls]')) {
          for (const attribute of ['aria-labelledby', 'aria-describedby', 'aria-controls']) {
            for (const identity of (element.getAttribute(attribute) ?? '').split(/\\s+/).filter(Boolean)) {
              if (!document.getElementById(identity)) throw new Error('Unresolved accessible reference: ' + identity);
            }
          }
        }
        const failures = [];
        const labels = [];
        const printPanels = [];
        for (const panels of figure.querySelectorAll('.visual-lesson-print-panels')) {
          if (!panels.querySelector('[data-visual-lesson-print-panel="1"]')) continue;
          const source = panels.previousElementSibling.querySelector('[data-visual-lesson-screen]');
          const originalLabels = [...source.querySelectorAll('text')].map(label => label.textContent).filter(text => text.trim()).sort();
          const panelLabels = [...panels.querySelectorAll('text')].map(label => label.textContent).filter(text => text.trim()).sort();
          if (JSON.stringify(originalLabels) !== JSON.stringify(panelLabels)) failures.push('Print panels lost or duplicated drawing labels');
          printPanels.push({ sourceLabels: originalLabels.length, panelLabels: panelLabels.length });
        }
        const labelIsContained = ${svgLabelIsContained.toString()};
        for (const svg of figure.querySelectorAll('svg')) {
          if (!svg.getBoundingClientRect().width || getComputedStyle(svg).display === 'none') continue;
          if (!svg.querySelector('title') && !svg.getAttribute('aria-label')) failures.push('Unnamed SVG');
          for (const text of svg.querySelectorAll('text')) {
            const box = text.getBBox();
            if (!box.width || !box.height) continue;
            if (!labelIsContained(svg, text, 1)) failures.push('Label outside SVG: ' + text.textContent);
            const scale = svg.getScreenCTM();
            const size = parseFloat(getComputedStyle(text).fontSize) * Math.hypot(scale.a, scale.b);
            if (size < 12) failures.push('Unreadable label ' + size.toFixed(2) + 'px: ' + text.textContent);
            labels.push({ text: text.textContent, pixels: size });
          }
        }
        if (document.documentElement.scrollWidth > innerWidth + 1) failures.push('Page horizontal overflow');
        const scrollers = [...figure.querySelectorAll('.visual-lesson-canvas')].map(canvas => {
          const maximum = canvas.scrollWidth - canvas.clientWidth;
          canvas.scrollLeft = maximum;
          const reachable = Math.abs(canvas.scrollLeft - maximum) <= 1;
          canvas.scrollLeft = 0;
          if (!reachable) failures.push('Diagram right edge is unreachable');
          return { maximum, reachable };
        });
        const rect = figure.getBoundingClientRect();
        const fitsWorkspace = ${lessonFitsWorkspace.toString()};
        if (!fitsWorkspace(figure, document.querySelector('[data-review-workspace]'), 1)) failures.push('Lesson extends below its workspace into the following section');
        const pinned = [...document.querySelectorAll('header, nav')].map(element => ({element, style: getComputedStyle(element), rect: element.getBoundingClientRect()})).filter(({style, rect}) => ['fixed', 'sticky'].includes(style.position) && rect.width && rect.height);
        const clearanceTop = 16 + Math.max(0, ...pinned.filter(({element}) => !element.matches('[data-review-controls]')).map(({style, rect}) => Number.isFinite(parseFloat(style.top)) ? parseFloat(style.top) + rect.height : 0));
        const clearanceBottom = 16 + Math.max(0, ...pinned.filter(({element}) => element.matches('[data-review-controls]')).map(({style, rect}) => Number.isFinite(parseFloat(style.bottom)) ? parseFloat(style.bottom) + rect.height : 0));
        const activeFrame = figure.querySelector('[data-coding-frame]:not([hidden])');
        return {failures, labels, printPanels, scrollers, frames: figure.querySelectorAll('[data-frame-key]').length, folds: figure.querySelectorAll('details').length,
          frameIndex: activeFrame ? Number(activeFrame.dataset.codingFrame) : undefined, frameHeight: activeFrame?.getBoundingClientRect().height,
          clearanceTop, clearanceBottom,
          rect: {x: rect.x + scrollX, y: rect.y + scrollY, width: rect.width, height: rect.height}};
      })()`, returnByValue: true, awaitPromise: true,
    });
    if (inspected.exceptionDetails) throw new Error(inspected.exceptionDetails.exception?.description ?? inspected.exceptionDetails.text);
    const details = inspected.result.value;
    if (allFrames && mode.media === 'screen' && step.view === 'trace') {
      frameCount = details.frames;
      if (details.frameIndex !== frameIndex) throw new Error(`${currentLabel}: trace skipped a checkpoint`);
      initialFrameHeight ??= details.frameHeight;
      if (Math.abs(details.frameHeight - initialFrameHeight) > .5) throw new Error(`${currentLabel}: trace stage changed height`);
    }
    const prefix = `${entry.slug}--${mode.name}--lesson-${index + 1}${allFrames && mode.media === 'screen' && step.view === 'trace' ? `--frame-${frameIndex + 1}` : ''}`;
    if (proofCaptures || details.failures.length) {
      const proofHeight = Math.max(mode.height, Math.ceil(details.rect.height + details.clearanceTop + details.clearanceBottom), mode.media === 'print' ? Math.ceil(details.rect.y + details.rect.height + details.clearanceBottom) : 0);
      await cdp.call('Emulation.setDeviceMetricsOverride', { width: mode.width, height: proofHeight, deviceScaleFactor: 1, mobile: Boolean(mode.mobile), screenWidth: mode.width, screenHeight: proofHeight });
      const positioned = await cdp.call('Runtime.evaluate', {
        expression: `(async () => {
          await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
          const figure = document.querySelector('[data-review-visual-host] > [data-visual-lesson-step]:not([hidden])');
          const before = figure.getBoundingClientRect();
          window.scrollTo({top: Math.max(0, before.top + scrollY - ${details.clearanceTop}), left: 0, behavior: 'instant'});
          await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
          const rect = figure.getBoundingClientRect();
          if (Math.abs(rect.width - ${details.rect.width}) > .5 || Math.abs(rect.height - ${details.rect.height}) > .5) throw new Error('Capture viewport changed figure geometry');
          if (rect.top < ${details.clearanceTop} - 1 || rect.bottom > innerHeight - ${details.clearanceBottom} + 1) throw new Error('Pinned controls obstruct the capture: ' + JSON.stringify({top: rect.top, bottom: rect.bottom, height: innerHeight, topClearance: ${details.clearanceTop}, bottomClearance: ${details.clearanceBottom}}));
          return { x: rect.x + scrollX, y: rect.y + scrollY, width: rect.width, height: rect.height };
        })()`, returnByValue: true, awaitPromise: true,
      });
      if (positioned.exceptionDetails) throw new Error(positioned.exceptionDetails.exception?.description ?? positioned.exceptionDetails.text);
      const capture = async suffix => {
        const screenshot = await cdp.call('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false, clip: {...positioned.result.value, scale: 1}});
        const bytes = Buffer.from(screenshot.data, 'base64');
        const metadata = await sharp(bytes).metadata();
        if (Math.abs(metadata.width - details.rect.width) > 1 || Math.abs(metadata.height - details.rect.height) > 1) throw new Error('Capture truncated the figure');
        writeFileSync(join(outputDir, `${prefix}${suffix}.png`), bytes);
      };
      await capture('');
      if (details.scrollers.some(canvas => canvas.maximum > 0)) {
        await cdp.call('Runtime.evaluate', { expression: `document.querySelector('[data-review-visual-host] > [data-visual-lesson-step]:not([hidden])').querySelectorAll('.visual-lesson-canvas').forEach(canvas => { canvas.scrollLeft = canvas.scrollWidth; })` });
        await capture('--right-edge');
      }
      await cdp.call('Emulation.setDeviceMetricsOverride', { width: mode.width, height: mode.height, deviceScaleFactor: 1, mobile: Boolean(mode.mobile), screenWidth: mode.width, screenHeight: mode.height });
    }
    report.push({slug: entry.slug, mode: mode.name, step, articleSha256, ...details});
    appendFileSync(join(outputDir, 'renders.ndjson'), `${JSON.stringify(report.at(-1))}\n`);
    writeFileSync(join(outputDir, 'progress.json'), `${JSON.stringify({cases: report.length, latest: currentLabel, complete: false}, null, 2)}\n`);
    if (details.failures.length) throw new Error(`${currentLabel}: ${details.failures.join('; ')}`);
    }
  }
  if (proofCaptures && mode.media === 'print') {
    const printed = await cdp.call('Page.printToPDF', { printBackground: true, preferCSSPageSize: true, generateTaggedPDF: true });
    const bytes = Buffer.from(printed.data, 'base64');
    if (bytes.length < 1000 || bytes.subarray(0, 4).toString() !== '%PDF') throw new Error(`${entry.slug}: invalid printed lesson`);
    writeFileSync(join(outputDir, `${entry.slug}--lesson.pdf`), bytes);
  }
}
try {
  await waitForBrowser();
  const targetResponse = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' });
  if (!targetResponse.ok) throw new Error(`Could not create Chrome target: ${targetResponse.status}`);
  const target = await targetResponse.json();
  cdp = new Cdp(target.webSocketDebuggerUrl);
  await cdp.ready();
  await cdp.call('Page.enable');
  await cdp.call('Runtime.enable');
  const geometryControls = await cdp.call('Runtime.evaluate', {
    expression: `(() => {
      const labelIsContained = ${svgLabelIsContained.toString()};
      const namespace = 'http://www.w3.org/2000/svg';
      const svg = document.createElementNS(namespace, 'svg');
      svg.setAttribute('viewBox', '0 0 300 180');
      svg.setAttribute('width', '300');
      svg.setAttribute('height', '180');
      const cases = [
        {name: 'valid-direct', x: 32, y: 40, expected: true},
        {name: 'valid-rotated', transform: 'translate(32 150) rotate(-90)', expected: true},
        {name: 'valid-nested', x: 12, y: 20, ancestor: 'translate(60 60) rotate(20)', expected: true},
        {name: 'outside-direct', x: 330, y: 40, expected: false},
        {name: 'outside-rotated', transform: 'translate(-20 150) rotate(-90)', expected: false},
        {name: 'outside-nested', x: 12, y: 20, ancestor: 'translate(330 60) rotate(20)', expected: false},
      ];
      document.body.append(svg);
      try {
        return cases.map(control => {
          const group = document.createElementNS(namespace, 'g');
          if (control.ancestor) group.setAttribute('transform', control.ancestor);
          const text = document.createElementNS(namespace, 'text');
          text.setAttribute('x', String(control.x ?? 0));
          text.setAttribute('y', String(control.y ?? 0));
          text.setAttribute('font-size', '16');
          if (control.transform) text.setAttribute('transform', control.transform);
          text.textContent = control.name;
          group.append(text);
          svg.append(group);
          const contained = labelIsContained(svg, text, 1);
          if (contained !== control.expected) throw new Error('SVG geometry control failed: ' + control.name);
          return {name: control.name, contained, expected: control.expected};
        });
      } finally { svg.remove(); }
    })()`, returnByValue: true,
  });
  if (geometryControls.exceptionDetails) throw new Error(geometryControls.exceptionDetails.exception?.description ?? geometryControls.exceptionDetails.text);
  writeFileSync(join(outputDir, 'geometry-controls.json'), `${JSON.stringify(geometryControls.result.value, null, 2)}\n`);
  if (lessonMode) {
    const flowControls = await cdp.call('Runtime.evaluate', {
      expression: `(() => {
        const fitsWorkspace = ${lessonFitsWorkspace.toString()};
        const workspace = document.createElement('div');
        const stage = document.createElement('div');
        const figure = document.createElement('figure');
        figure.style.cssText = 'height:180px;margin:0';
        stage.append(figure);
        workspace.append(stage);
        document.body.append(workspace);
        try {
          return [{name: 'natural-stage', maximum: '', expected: true}, {name: 'capped-stage', maximum: '80px', expected: false}].map(control => {
            stage.style.maxHeight = control.maximum;
            const fits = fitsWorkspace(figure, workspace, 1);
            if (fits !== control.expected) throw new Error('Lesson flow control failed: ' + control.name);
            return {name: control.name, fits, expected: control.expected};
          });
        } finally { workspace.remove(); }
      })()`, returnByValue: true,
    });
    if (flowControls.exceptionDetails) throw new Error(flowControls.exceptionDetails.exception?.description ?? flowControls.exceptionDetails.text);
    writeFileSync(join(outputDir, 'flow-controls.json'), `${JSON.stringify(flowControls.result.value, null, 2)}\n`);
  }
  if (lessonMode) {
    await cdp.call('Network.enable');
    cdp.on('Network.responseReceived', ({ requestId, response }) => {
      if (response.mimeType === 'text/html') articleResponses.set(response.url, { requestId, status: response.status });
    });
  }

  for (const mode of modes) {
    await cdp.call('Emulation.setDeviceMetricsOverride', {
      width: mode.width,
      height: mode.height,
      deviceScaleFactor: 1,
      mobile: Boolean(mode.mobile),
      screenWidth: mode.width,
      screenHeight: mode.height,
    });
    await cdp.call('Emulation.setEmulatedMedia', {
      media: mode.media,
      features: [{ name: 'prefers-color-scheme', value: mode.theme }, ...(lessonMode ? [{name: 'prefers-reduced-motion', value: 'reduce'}] : [])],
    });

    for (const entry of entries) {
      if (lessonMode) {
        await checkLesson(entry, mode);
        if ((entries.indexOf(entry) + 1) % 25 === 0) console.log(`Visual lessons verified: ${entries.indexOf(entry) + 1}/${entries.length} in ${mode.name}.`);
        continue;
      }
      currentLabel = `${entry.slug}/${mode.name}/navigation`;
      const loaded = cdp.once('Page.loadEventFired');
      await cdp.call('Page.navigate', { url: `${baseUrl}/${entry.category}/${entry.slug}/` });
      await loaded;
      await delay(500);
      const settled = await cdp.call('Runtime.evaluate', {
        expression: `document.fonts.ready.then(() => new Promise((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(() => resolve(true)))
        ))`,
        returnByValue: true,
        awaitPromise: true,
      });
      if (settled.exceptionDetails) {
        throw new Error(settled.exceptionDetails.exception?.description ?? settled.exceptionDetails.text);
      }

      for (const visualId of entry.visualIds) {
        const label = `${entry.slug}/${visualId}/${mode.name}`;
        currentLabel = label;
        await waitForVisualLayout(cdp, visualId, mode.theme, label);
        const expression = `(() => new Promise((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(() => {
            window.scrollTo({ left: 0, top: 0, behavior: 'instant' });
            const comments = document.createTreeWalker(document, NodeFilter.SHOW_COMMENT);
            let marker;
            while (comments.nextNode()) {
              if (comments.currentNode.data.trim() === ${JSON.stringify(`visual:${visualId}`)}) {
                marker = comments.currentNode;
                break;
              }
            }
            const supplementalTitle = ${JSON.stringify(supplemental)}
              ? document.querySelector(${JSON.stringify(`[data-supplemental-visual="${visualId}"]`)})
              : null;
            let visual = supplementalTitle?.matches('figure.learning-figure')
              ? supplementalTitle : supplementalTitle?.nextElementSibling ?? marker?.nextElementSibling;
            while (visual && !visual.matches('figure.learning-figure, .mermaid')) visual = visual.nextElementSibling;
            if (!visual) throw new Error(${JSON.stringify(`${label}: visual not found`)});
            const visualHost = visual.closest('.visual-review-visual');
            const visualHostStyle = visualHost ? getComputedStyle(visualHost) : null;
            if (visualHost) {
              visualHost.scrollTop = 0;
              visualHost.dataset.visualReviewVerticalHost = ${JSON.stringify(visualId)};
            }
            const caption = visual.matches('figure')
              ? visual.querySelector('figcaption')
              : visual.nextElementSibling?.matches('.diagram-caption') ? visual.nextElementSibling : null;
            const captionText = caption?.textContent.trim() ?? '';
            const visualTitle = visual.matches('figure')
              ? visual.querySelector('.visual-title')
              : supplementalTitle ?? (marker?.previousElementSibling?.matches('.visual-title') ? marker.previousElementSibling : null);
            const titleRect = visualTitle?.getBoundingClientRect();
            const titleStyle = visualTitle ? getComputedStyle(visualTitle) : null;
            const visualKicker = visual.matches('.mermaid') && visualTitle?.previousElementSibling?.matches('.visual-kicker')
              ? visualTitle.previousElementSibling
              : null;
            const scroll = visual.matches('.mermaid') ? visual : visual.querySelector('.visual-scroll');
            if (scroll) {
              scroll.scrollLeft = 0;
              scroll.dataset.visualReviewScroll = ${JSON.stringify(visualId)};
            }
            const rect = visual.getBoundingClientRect();
            const captionRect = caption?.getBoundingClientRect();
            const captureBoxes = [
              visualKicker?.getBoundingClientRect(),
              visualTitle?.getBoundingClientRect(),
              rect,
              captionRect,
            ].filter(Boolean);
            const captureRect = {
              left: Math.min(...captureBoxes.map((box) => box.left)),
              top: Math.min(...captureBoxes.map((box) => box.top)),
              right: Math.max(...captureBoxes.map((box) => box.right)),
              bottom: Math.max(...captureBoxes.map((box) => box.bottom)),
            };
            const svgElements = [...visual.querySelectorAll('svg')];
            const scrollRect = scroll?.getBoundingClientRect();
            const scrollContentRects = scroll
              ? [...scroll.querySelectorAll('svg')].map((svg) => svg.getBoundingClientRect())
                .filter((box) => box.width > 0 || box.height > 0)
              : [];
            const clippedText = ${JSON.stringify(entry.medium === 'svg')} ? svgElements.flatMap((svg) => {
              const viewBox = svg.viewBox.baseVal;
              if (!viewBox?.width) return [];
              const labelIsContained = ${svgLabelIsContained.toString()};
              return [...svg.querySelectorAll('text')].flatMap((text) => {
                return labelIsContained(svg, text, 0.5) ? [] : [text.textContent.trim()];
              });
            }) : [];
            const renderedTextSizes = svgElements.flatMap((svg) => {
              const viewBoxWidth = svg.viewBox.baseVal.width;
              const renderedWidth = svg.getBoundingClientRect().width;
              if (!viewBoxWidth || !renderedWidth) return [];
              const labels = svg.querySelectorAll(
                'text, foreignObject .nodeLabel, foreignObject .edgeLabel, foreignObject .label, foreignObject .cluster-label'
              );
              return [...labels].filter((label) => label.textContent.trim()).map((label) =>
                Number.parseFloat(getComputedStyle(label).fontSize) * renderedWidth / viewBoxWidth
              ).filter(Number.isFinite);
            });
            const unresolvedSvgReferences = svgElements.flatMap((svg, svgIndex) => {
              const ids = new Set([...svg.querySelectorAll('[id]')].map((element) => element.id));
              const missing = new Set();
              for (const element of [svg, ...svg.querySelectorAll('*')]) {
                const style = getComputedStyle(element);
                const candidates = [
                  style.markerStart,
                  style.markerMid,
                  style.markerEnd,
                  style.clipPath,
                  style.filter,
                  style.maskImage,
                  style.fill,
                  style.stroke,
                  ...element.getAttributeNames().map((name) => element.getAttribute(name)),
                ].filter(Boolean);
                for (const candidate of candidates) {
                  for (const match of candidate.matchAll(/url\(([^)]+)\)/g)) {
                    const target = match[1].trim().replace(/^["']|["']$/g, '');
                    const hash = target.lastIndexOf('#');
                    if (hash < 0) continue;
                    const id = target.slice(hash + 1);
                    if (!ids.has(id)) missing.add(id);
                  }
                }
                for (const name of ['href', 'xlink:href']) {
                  const reference = element.getAttribute(name);
                  if (reference?.startsWith('#') && !ids.has(reference.slice(1))) missing.add(reference.slice(1));
                }
              }
              return [...missing].map((id) => 'svg ' + (svgIndex + 1) + ': #' + id);
            });
            const rangeWidth = (element) => {
              if (!element) return 0;
              const range = document.createRange();
              range.selectNodeContents(element);
              return range.getBoundingClientRect().width;
            };
            resolve({
              innerWidth,
              pageWidth: document.documentElement.scrollWidth,
              rect: { x: rect.left + scrollX, y: rect.top + scrollY, width: rect.width, height: rect.height },
              captureRect: {
                x: captureRect.left + scrollX,
                y: captureRect.top + scrollY,
                width: captureRect.right - captureRect.left,
                height: captureRect.bottom - captureRect.top,
              },
              figureFitsViewport: rect.left >= -0.5 && rect.right <= innerWidth + 0.5,
              verticalClip: visualHost ? {
                clientHeight: visualHost.clientHeight,
                scrollHeight: visualHost.scrollHeight,
                overflowY: visualHostStyle.overflowY,
                captureRect: {
                  x: visualHost.getBoundingClientRect().left + scrollX,
                  y: visualHost.getBoundingClientRect().top + scrollY,
                  width: visualHost.getBoundingClientRect().width,
                  height: visualHost.getBoundingClientRect().height,
                },
              } : null,
              scroll: scroll ? {
                clientWidth: scroll.clientWidth,
                scrollWidth: scroll.scrollWidth,
                overflowX: getComputedStyle(scroll).overflowX,
                viewportLeft: scrollRect.left,
                viewportRight: scrollRect.right,
                contentLeft: scrollContentRects.length
                  ? Math.min(...scrollContentRects.map((box) => box.left))
                  : null,
                contentRight: scrollContentRects.length
                  ? Math.max(...scrollContentRects.map((box) => box.right))
                  : null,
              } : null,
              svgCount: svgElements.length,
              mermaid: visual.matches('.mermaid'),
              titleCount: visual.querySelectorAll('svg > title').length,
              descCount: visual.querySelectorAll('svg > desc').length,
              minRenderedText: renderedTextSizes.length ? Math.min(...renderedTextSizes) : null,
              visualTitle: visualTitle?.textContent.trim() ?? '',
              visualTitleGlyphWidth: rangeWidth(visualTitle),
              visualTitleVisibility: visualTitle ? {
                width: titleRect.width,
                height: titleRect.height,
                clientWidth: visualTitle.clientWidth,
                scrollWidth: visualTitle.scrollWidth,
                visibility: titleStyle.visibility,
                opacity: Number(titleStyle.opacity),
                clip: titleStyle.clip,
                clipPath: titleStyle.clipPath,
              } : null,
              caption: captionText,
              rawCaptionMath: captionText.split('$').length > 2 || captionText.includes(String.fromCharCode(92)),
              captionGlyphWidth: rangeWidth(caption),
              clippedText,
              proof: (() => {
                const argument = visual.querySelector(':scope > .visual-argument');
                if (!argument) return null;
                const svg = argument.querySelector('svg');
                const box = svg.getBoundingClientRect();
                const labels = [...svg.querySelectorAll('text')].filter((label) => label.textContent.trim()).map((label) => ({ text: label.textContent, box: label.getBoundingClientRect() }));
                const overlaps = [];
                for (let first = 0; first < labels.length; first++) for (let second = first + 1; second < labels.length; second++) {
                  const left = labels[first];
                  const right = labels[second];
                  if (Math.min(left.box.right, right.box.right) - Math.max(left.box.left, right.box.left) > 1 && Math.min(left.box.bottom, right.box.bottom) - Math.max(left.box.top, right.box.top) > 1) overlaps.push([left.text, right.text]);
                }
                const nodeClipping = [...svg.querySelectorAll('[data-proof-token], [data-proof-cell]')].flatMap((node) => {
                  const label = node.tagName === 'g' ? node.querySelector(':scope > text') : node.parentElement.querySelector(':scope > text');
                  if (label?.tagName !== 'text') return ['missing cell label'];
                  const declared = node.getAttribute('data-proof-box')?.split(',').map(Number);
                  if (node.tagName === 'g' && (!declared || declared.length !== 4 || declared.some(value => !Number.isFinite(value)))) return ['missing explicit cell bounds'];
                  const bounds = declared ? { x: declared[0], y: declared[1], width: declared[2], height: declared[3] } : node.getBBox();
                  const textBounds = label.getBBox();
                  return textBounds.x < bounds.x - .5 || textBounds.y < bounds.y - .5 || textBounds.x + textBounds.width > bounds.x + bounds.width + .5 || textBounds.y + textBounds.height > bounds.y + bounds.height + .5 ? [label.textContent] : [];
                });
                const derivation = visual.querySelector(':scope > .visual-derivation');
                const argumentRect = argument.getBoundingClientRect();
                const boxes = [titleRect, argumentRect, captionRect];
                return {
                  identity: argument.dataset.visualArgument,
                  markup: argument.outerHTML,
                  minText: Math.min(...[...svg.querySelectorAll('text')].map((label) => parseFloat(getComputedStyle(label).fontSize) * box.width / svg.viewBox.baseVal.width)),
                  width: box.width,
                  height: box.height,
                  captureRect: {
                    x: Math.min(...boxes.map(bounds => bounds.left)) + scrollX,
                    y: Math.min(...boxes.map(bounds => bounds.top)) + scrollY,
                    width: Math.max(...boxes.map(bounds => bounds.right)) - Math.min(...boxes.map(bounds => bounds.left)),
                    height: Math.max(...boxes.map(bounds => bounds.bottom)) - Math.min(...boxes.map(bounds => bounds.top)),
                  },
                  overlaps,
                  nodeClipping,
                  captionBeforeDerivation: !derivation || captionRect.bottom <= derivation.getBoundingClientRect().top + 1,
                };
              })(),
              textOverlaps: visual.matches('.editorial-illustration') ? (() => {
                const labels = [...visual.querySelectorAll('svg text')].filter((label) => label.textContent.trim()).map((label) => ({ text: label.textContent, box: label.getBoundingClientRect() }));
                const overlaps = [];
                for (let first = 0; first < labels.length; first++) for (let second = first + 1; second < labels.length; second++) {
                  const left = labels[first];
                  const right = labels[second];
                  if (Math.min(left.box.right, right.box.right) - Math.max(left.box.left, right.box.left) > 1 && Math.min(left.box.bottom, right.box.bottom) - Math.max(left.box.top, right.box.top) > 1) overlaps.push([left.text, right.text]);
                }
                return overlaps;
              })() : [],
              nodeClipping: svgElements.flatMap((svg) => [...svg.querySelectorAll('[data-editorial-node]')].flatMap((node) => {
                const bounds = node.querySelector('rect')?.getBBox();
                const label = node.querySelector('text');
                const textBounds = label?.getBBox();
                return bounds && textBounds && (textBounds.x < bounds.x - .5 || textBounds.y < bounds.y - .5 || textBounds.x + textBounds.width > bounds.x + bounds.width + .5 || textBounds.y + textBounds.height > bounds.y + bounds.height + .5) ? [label.textContent] : [];
              })),
              unresolvedSvgReferences,
              printMinWidths: svgElements.map((svg) => getComputedStyle(svg).minWidth),
            });
          }));
        }))()`;
        const evaluation = await cdp.call('Runtime.evaluate', {
          expression,
          returnByValue: true,
          awaitPromise: true,
        });
        if (evaluation.exceptionDetails) {
          throw new Error(evaluation.exceptionDetails.exception?.description ?? evaluation.exceptionDetails.text);
        }
        const details = evaluation.result.value;
        if (!details.proof || details.proof.identity !== visualId) throw new Error(`${label} lost its proof composition`);
        if (serialize(parseFragment(details.proof.markup)) !== serialize(parseFragment(renderVisualArgument(visualId)))) throw new Error(`${label} serves a stale proof composition`);
        if (details.proof.width < 1 || details.proof.height < 1) throw new Error(`${label} hides its proof composition`);
        if (mode.media !== 'print' && details.proof.minText < 12) throw new Error(`${label} proof text is smaller than 12px: ${details.proof.minText}`);
        if (details.proof.overlaps.length) throw new Error(`${label} proof text overlaps: ${JSON.stringify(details.proof.overlaps)}`);
        if (details.proof.nodeClipping.length) throw new Error(`${label} proof text escapes a cell: ${details.proof.nodeClipping.join(', ')}`);
        if (!details.proof.captionBeforeDerivation) throw new Error(`${label} detaches its proof caption from the drawing`);
        delete details.proof.markup;
        if (mode.mobile && details.innerWidth !== 390) throw new Error(`${label} reported innerWidth ${details.innerWidth}`);
        if (details.pageWidth > details.innerWidth + 1) throw new Error(`${label} creates page overflow (${details.pageWidth} > ${details.innerWidth})`);
        if (!details.figureFitsViewport) throw new Error(`${label} exceeds the viewport`);
        if (mode.mobile && details.verticalClip &&
          details.verticalClip.scrollHeight > details.verticalClip.clientHeight + 1 &&
          ['auto', 'scroll', 'hidden', 'clip'].includes(details.verticalClip.overflowY)) {
          throw new Error(
            `${label} clips the promoted visual vertically ` +
            `(${details.verticalClip.scrollHeight} > ${details.verticalClip.clientHeight}, overflow-y ${details.verticalClip.overflowY})`
          );
        }
        if (!details.caption.startsWith('Read it this way:') || details.captionGlyphWidth < 1) {
          throw new Error(`${label} lost its rendered direct caption`);
        }
        if (!details.visualTitle || details.visualTitleGlyphWidth < 1) {
          throw new Error(`${label} lost its rendered learning title`);
        }
        const titleVisibility = details.visualTitleVisibility;
        if (!titleVisibility || titleVisibility.width <= 1 || titleVisibility.height < 8 ||
          titleVisibility.visibility !== 'visible' || titleVisibility.opacity <= 0 ||
          titleVisibility.clip !== 'auto' || titleVisibility.clipPath !== 'none' ||
          titleVisibility.scrollWidth > titleVisibility.clientWidth + 1) {
          throw new Error(`${label} hides or clips its learning title: ${JSON.stringify(titleVisibility)}`);
        }
        if (details.rawCaptionMath) throw new Error(`${label} retains raw math delimiters in its caption`);
        if (details.mermaid && (details.svgCount === 0 || details.minRenderedText === null ||
          details.svgCount !== details.titleCount || details.svgCount !== details.descCount)) {
          throw new Error(`${label} lost its rendered Mermaid graph or accessibility markup`);
        }
        if (mode.media !== 'print' && details.minRenderedText !== null && details.minRenderedText < 8) {
          throw new Error(`${label} renders text at ${details.minRenderedText.toFixed(2)}px`);
        }
        if (entry.medium === 'svg') {
          if (details.svgCount === 0 || details.svgCount !== details.titleCount || details.svgCount !== details.descCount) {
            throw new Error(`${label} lost SVG accessibility markup`);
          }
          if (details.clippedText.length) throw new Error(`${label} has text outside its viewBox: ${details.clippedText.join(', ')}`);
          if (details.textOverlaps.length) throw new Error(`${label} overlaps text: ${JSON.stringify(details.textOverlaps)}`);
          if (details.nodeClipping.length) throw new Error(`${label} has text outside its node: ${details.nodeClipping.join(', ')}`);
          if (details.unresolvedSvgReferences.length) {
            throw new Error(`${label} has unresolved SVG references: ${details.unresolvedSvgReferences.join(', ')}`);
          }
          if (mode.media === 'print' && details.printMinWidths.some((width) => width !== '0px')) {
            throw new Error(`${label} retains an SVG minimum width in print`);
          }
        }

        const hasVerticalScroll = details.verticalClip &&
          details.verticalClip.scrollHeight > details.verticalClip.clientHeight + 1 &&
          ['auto', 'scroll'].includes(details.verticalClip.overflowY) &&
          mode.media !== 'print';
        const screenshot = await cdp.call('Page.captureScreenshot', {
          format: 'png',
          captureBeyondViewport: true,
          clip: { ...(hasVerticalScroll ? details.verticalClip.captureRect : details.captureRect), scale: 1 },
        });
        const prefix = `${mode.name}--${entry.slug}--${visualId}`;
        writeFileSync(join(outputDir, `${prefix}.png`), Buffer.from(screenshot.data, 'base64'));
        if (proofCaptures) {
          const proofScreenshot = await cdp.call('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { ...details.proof.captureRect, scale: 1 } });
          writeFileSync(join(outputDir, `${prefix}--proof.png`), Buffer.from(proofScreenshot.data, 'base64'));
        }

        if (allFrames && mode.media === 'screen') {
          const frameCountEvaluation = await cdp.call('Runtime.evaluate', {
            expression: `document.querySelector(${JSON.stringify(`[data-coding-slug="${entry.slug}"]`)})?.querySelectorAll('[data-coding-frame-button]').length ?? 0`,
            returnByValue: true,
          });
          const frameCount = frameCountEvaluation.result.value;
          details.frameCaptures = [];
          for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
            const transitionEvaluation = await cdp.call('Runtime.evaluate', {
              expression: `(() => new Promise((resolve) => {
                const visual = document.querySelector(${JSON.stringify(`[data-coding-slug="${entry.slug}"]`)});
                visual.querySelectorAll('[data-coding-frame-button]')[${frameIndex}].click();
                requestAnimationFrame(() => requestAnimationFrame(() => resolve({
                  activeFrame: Number(visual.dataset.activeFrame),
                  animationCount: visual.getAnimations({ subtree: true }).filter((animation) => animation.playState === 'running').length,
                })));
              }))()`,
              returnByValue: true,
              awaitPromise: true,
            });
            if (transitionEvaluation.exceptionDetails) {
              throw new Error(transitionEvaluation.exceptionDetails.exception?.description ?? transitionEvaluation.exceptionDetails.text);
            }
            const transition = transitionEvaluation.result.value;
            if (transition.activeFrame !== frameIndex) throw new Error(`${label}/frame-${frameIndex}: direct selection activated frame ${transition.activeFrame}`);

            if (frameIndex > 0 && transition.animationCount > 0) {
              const motionScreenshot = await cdp.call('Page.captureScreenshot', {
                format: 'png',
                captureBeyondViewport: true,
                clip: { ...details.captureRect, scale: 1 },
              });
              writeFileSync(join(outputDir, `${prefix}--frame-${String(frameIndex + 1).padStart(3, '0')}--motion.png`), Buffer.from(motionScreenshot.data, 'base64'));
            }
            await cdp.call('Runtime.evaluate', {
              expression: `document.querySelector(${JSON.stringify(`[data-coding-slug="${entry.slug}"]`)})?.getAnimations({ subtree: true }).forEach((animation) => animation.finish())`,
              returnByValue: true,
            });
            await delay(20);

            const frameEvaluation = await cdp.call('Runtime.evaluate', {
              expression: `(() => {
                const visual = document.querySelector(${JSON.stringify(`[data-coding-slug="${entry.slug}"]`)});
                const figure = visual.closest('figure.learning-figure');
                const frames = [...visual.querySelectorAll('[data-coding-frame]')];
                const visibleFrames = frames.filter((frame) => !frame.hidden && getComputedStyle(frame).display !== 'none');
                const rect = figure.getBoundingClientRect();
                const renderedTextSizes = [...visibleFrames[0].querySelectorAll('svg')].flatMap((svg) => {
                  const viewBoxWidth = svg.viewBox.baseVal.width;
                  const renderedWidth = svg.getBoundingClientRect().width;
                  if (!viewBoxWidth || !renderedWidth) return [];
                  return [...svg.querySelectorAll('text')].filter((text) => text.textContent.trim()).map((text) =>
                    Number.parseFloat(getComputedStyle(text).fontSize) * renderedWidth / viewBoxWidth
                  ).filter(Number.isFinite);
                });
                const motionKeys = [...visibleFrames[0].querySelectorAll('[data-motion-key]')].map((element) => element.dataset.motionKey);
                const mechanismClipping = [...visibleFrames[0].querySelectorAll('.coding-mechanism svg')].flatMap((svg) => {
                  const viewBox = svg.viewBox.baseVal;
                  return [...svg.querySelectorAll('text')].flatMap((text) => {
                    const box = text.getBBox();
                    return box.x < viewBox.x - .5 || box.y < viewBox.y - .5 ||
                      box.x + box.width > viewBox.x + viewBox.width + .5 ||
                      box.y + box.height > viewBox.y + viewBox.height + .5 ? [text.textContent] : [];
                  });
                });
                return {
                  activeFrame: Number(visual.dataset.activeFrame),
                  visibleFrames: visibleFrames.length,
                  pageWidth: document.documentElement.scrollWidth,
                  innerWidth,
                  figureFitsViewport: rect.left >= -0.5 && rect.right <= innerWidth + 0.5,
                  minRenderedText: renderedTextSizes.length ? Math.min(...renderedTextSizes) : null,
                  duplicateMotionKeys: motionKeys.filter((key, index) => motionKeys.indexOf(key) !== index),
                  mechanismClipping,
                  captureRect: { x: rect.left + scrollX, y: rect.top + scrollY, width: rect.width, height: rect.height },
                };
              })()`,
              returnByValue: true,
            });
            if (frameEvaluation.exceptionDetails) {
              throw new Error(frameEvaluation.exceptionDetails.exception?.description ?? frameEvaluation.exceptionDetails.text);
            }
            const frameDetails = frameEvaluation.result.value;
            const frameLabel = `${label}/frame-${frameIndex}`;
            if (frameDetails.activeFrame !== frameIndex || frameDetails.visibleFrames !== 1) throw new Error(`${frameLabel} is not the sole active frame`);
            if (frameDetails.pageWidth > frameDetails.innerWidth + 1 || !frameDetails.figureFitsViewport) throw new Error(`${frameLabel} creates horizontal overflow`);
            if (mode.mobile && frameDetails.minRenderedText !== null && frameDetails.minRenderedText < 8) throw new Error(`${frameLabel} renders text at ${frameDetails.minRenderedText.toFixed(2)}px`);
            if (frameDetails.duplicateMotionKeys.length > 0) throw new Error(`${frameLabel} duplicates motion keys: ${[...new Set(frameDetails.duplicateMotionKeys)].join(', ')}`);
            if (frameDetails.mechanismClipping.length > 0) throw new Error(`${frameLabel} clips mechanism text: ${frameDetails.mechanismClipping.join(', ')}`);
            const frameScreenshot = await cdp.call('Page.captureScreenshot', {
              format: 'png',
              captureBeyondViewport: true,
              clip: { ...frameDetails.captureRect, scale: 1 },
            });
            const frameFile = `${prefix}--frame-${String(frameIndex + 1).padStart(3, '0')}.png`;
            writeFileSync(join(outputDir, frameFile), Buffer.from(frameScreenshot.data, 'base64'));
            details.frameCaptures.push({ frameIndex, file: frameFile, animationCount: transition.animationCount, ...frameDetails });
          }
        }

        if (hasVerticalScroll) {
          const verticalEndEvaluation = await cdp.call('Runtime.evaluate', {
            expression: `(() => {
              const host = document.querySelector(${JSON.stringify(`[data-visual-review-vertical-host="${visualId}"]`)});
              host.scrollTop = host.scrollHeight;
              const visual = host.querySelector('figure.learning-figure, .mermaid');
              const viewport = host.getBoundingClientRect();
              const content = visual.getBoundingClientRect();
              return {
                scrollTop: host.scrollTop,
                maxScrollTop: host.scrollHeight - host.clientHeight,
                viewportBottom: viewport.bottom,
                contentBottom: content.bottom,
              };
            })()`,
            returnByValue: true,
          });
          if (verticalEndEvaluation.exceptionDetails) {
            throw new Error(
              verticalEndEvaluation.exceptionDetails.exception?.description ??
              verticalEndEvaluation.exceptionDetails.text
            );
          }
          const verticalEnd = verticalEndEvaluation.result.value;
          if (Math.abs(verticalEnd.scrollTop - verticalEnd.maxScrollTop) > 1) {
            throw new Error(`${label} cannot reach its vertical scroll end`);
          }
          if (verticalEnd.contentBottom > verticalEnd.viewportBottom + 1) {
            throw new Error(`${label} has content beyond its reachable vertical scroll end`);
          }
          details.verticalClip.endScrollTop = verticalEnd.scrollTop;
          details.verticalClip.maxScrollTop = verticalEnd.maxScrollTop;
          const verticalEndScreenshot = await cdp.call('Page.captureScreenshot', {
            format: 'png',
            captureBeyondViewport: true,
            clip: { ...details.verticalClip.captureRect, scale: 1 },
          });
          writeFileSync(
            join(outputDir, `${prefix}--vertical-scroll-end.png`),
            Buffer.from(verticalEndScreenshot.data, 'base64')
          );
        }

        if (details.scroll && details.scroll.scrollWidth <= details.scroll.clientWidth + 1) {
          if (details.scroll.contentLeft !== null && details.scroll.contentLeft < details.scroll.viewportLeft - 1) {
            throw new Error(`${label} paints content before its viewport without a reachable scroll range`);
          }
          if (details.scroll.contentRight !== null && details.scroll.contentRight > details.scroll.viewportRight + 1) {
            throw new Error(`${label} paints content after its viewport without a reachable scroll range`);
          }
        }
        if (details.scroll && details.scroll.scrollWidth > details.scroll.clientWidth + 1 && mode.media !== 'print') {
          if (details.scroll.contentLeft !== null && details.scroll.contentLeft < details.scroll.viewportLeft - 1) {
            throw new Error(`${label} has content before its reachable scroll start`);
          }
          const endEvaluation = await cdp.call('Runtime.evaluate', {
            expression: `(() => {
              const scroll = document.querySelector(${JSON.stringify(`[data-visual-review-scroll="${visualId}"]`)});
              scroll.scrollLeft = scroll.scrollWidth;
              const viewport = scroll.getBoundingClientRect();
              const content = [...scroll.querySelectorAll('svg')].map((svg) => svg.getBoundingClientRect())
                .filter((box) => box.width > 0 || box.height > 0);
              return {
                scrollLeft: scroll.scrollLeft,
                maxScrollLeft: scroll.scrollWidth - scroll.clientWidth,
                viewportRight: viewport.right,
                contentRight: content.length ? Math.max(...content.map((box) => box.right)) : null,
              };
            })()`,
            returnByValue: true,
          });
          if (endEvaluation.exceptionDetails) {
            throw new Error(endEvaluation.exceptionDetails.exception?.description ?? endEvaluation.exceptionDetails.text);
          }
          const endDetails = endEvaluation.result.value;
          if (endDetails.contentRight !== null && endDetails.contentRight > endDetails.viewportRight + 1) {
            throw new Error(`${label} has content beyond its reachable scroll end`);
          }
          details.scroll.endScrollLeft = endDetails.scrollLeft;
          details.scroll.maxScrollLeft = endDetails.maxScrollLeft;
          details.scroll.endContentRight = endDetails.contentRight;
          const endScreenshot = await cdp.call('Page.captureScreenshot', {
            format: 'png',
            captureBeyondViewport: true,
            clip: { ...details.captureRect, scale: 1 },
          });
          writeFileSync(join(outputDir, `${prefix}--scroll-end.png`), Buffer.from(endScreenshot.data, 'base64'));
        }
        report.push({ mode: mode.name, ...entry, visualId, ...details });
        if (report.length % 20 === 0) {
          writeFileSync(join(outputDir, 'progress.json'), `${JSON.stringify({ renders: report, complete: false }, null, 2)}\n`);
          console.log(`Native visual progress: ${report.length} cases; latest ${label}`);
        }
      }
    }
  }

  if (lessonMode && JSON.stringify(sourceSnapshot()) !== JSON.stringify(frozenSource)) throw new Error('Source changed during native lesson verification');
  writeFileSync(join(outputDir, 'report.json'), `${JSON.stringify({ complete: true, lessonMode, bindDist, baseUrl, sourceFiles: frozenSource, renders: report, noVisualSlugs }, null, 2)}\n`);
  console.log(`Visual rendering passed: ${report.length} renders, ${noVisualSlugs.length} no-visual outcome(s); screenshots saved to ${outputDir}`);
} catch (error) {
  writeFileSync(join(outputDir, 'failure.json'), `${JSON.stringify({ label: currentLabel, error: error.stack, browserExitCode: browser.exitCode, browserStderr: browserErrors.slice(-4000) }, null, 2)}\n`);
  throw error;
} finally {
  cdp?.close();
  await stopBrowser();
}