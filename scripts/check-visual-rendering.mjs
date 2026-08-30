import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';
import process from 'node:process';

const root = resolve(import.meta.dirname, '..');

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

const executable = browserExecutable();
if (!executable) throw new Error('Chrome or Edge was not found; set CHROME_PATH or --browser');
if (process.platform !== 'win32' && /^[A-Za-z]:|^\/mnt\/[a-z]\//.test(executable)) {
  throw new Error('Run this script with Windows Node when using a Windows browser so its CDP port is reachable');
}

const baseUrl = option('base-url', 'http://127.0.0.1:8123').replace(/\/$/, '');
const outputDir = resolve(root, option('output', 'scratch/visual-render-review'));
const auditsDir = join(root, 'data', 'visual-audits');
const entries = slugs.map((slug) => {
  const auditPath = join(auditsDir, `${slug}.json`);
  if (!existsSync(auditPath)) throw new Error(`Missing audit sidecar for ${slug}`);
  const audit = JSON.parse(readFileSync(auditPath, 'utf8'));
  if (audit.status !== 'implemented') throw new Error(`${slug} is ${audit.status}, not implemented`);
  const articlePath = join(root, audit.article);
  const article = readFileSync(articlePath, 'utf8');
  const category = article.match(/^---\n[\s\S]*?^category:\s*["']?([^"'\n]+)["']?\s*$[\s\S]*?^---$/m)?.[1]?.trim();
  if (!category) throw new Error(`Could not read category from ${audit.article}`);
  return {
    slug,
    category,
    medium: audit.medium,
    visualIds: audit.implementation.visualIds,
    article: basename(audit.article),
  };
});

const modes = [
  { name: 'desktop-light', width: 1280, height: 900, theme: 'light', media: 'screen' },
  { name: 'desktop-dark', width: 1280, height: 900, theme: 'dark', media: 'screen' },
  { name: 'mobile-light', width: 390, height: 900, theme: 'light', media: 'screen', mobile: true },
  { name: 'mobile-dark', width: 390, height: 900, theme: 'dark', media: 'screen', mobile: true },
  { name: 'print', width: 1280, height: 900, theme: 'light', media: 'print' },
];

class Cdp {
  constructor(url) {
    this.socket = new WebSocket(url);
    this.nextId = 1;
    this.pending = new Map();
    this.events = new Map();
    this.socket.onmessage = ({ data }) => {
      const message = JSON.parse(data);
      if (message.id) {
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        if (message.error) pending.reject(new Error(message.error.message));
        else pending.resolve(message.result);
        return;
      }
      const listeners = this.events.get(message.method) ?? [];
      this.events.delete(message.method);
      for (const resolveEvent of listeners) resolveEvent(message.params);
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
      this.pending.set(id, { resolve: resolveCall, reject: rejectCall });
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

  close() {
    this.socket.close();
  }
}

const delay = (milliseconds) => new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
const port = 9300 + (process.pid % 500);
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
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profileDir}`,
  'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
let browserErrors = '';
browser.stderr.on('data', (chunk) => { browserErrors += chunk.toString(); });

async function waitForBrowser() {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (browser.exitCode !== null) throw new Error(`Browser exited early: ${browserErrors.slice(-1000)}`);
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (response.ok) return;
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
let cdp;
try {
  await waitForBrowser();
  const targetResponse = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' });
  if (!targetResponse.ok) throw new Error(`Could not create Chrome target: ${targetResponse.status}`);
  const target = await targetResponse.json();
  cdp = new Cdp(target.webSocketDebuggerUrl);
  await cdp.ready();
  await cdp.call('Page.enable');
  await cdp.call('Runtime.enable');

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
      features: [{ name: 'prefers-color-scheme', value: mode.theme }],
    });

    for (const entry of entries) {
      const loaded = cdp.once('Page.loadEventFired');
      await cdp.call('Page.navigate', { url: `${baseUrl}/${entry.category}/${entry.slug}/` });
      await loaded;
      await delay(500);

      for (const visualId of entry.visualIds) {
        const expression = `(() => new Promise((resolve) => {
          document.documentElement.dataset.theme = ${JSON.stringify(mode.theme)};
          requestAnimationFrame(() => requestAnimationFrame(() => {
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
            if (!visual) throw new Error('Visual not found');
            const caption = visual.matches('figure')
              ? visual.querySelector('figcaption')
              : visual.nextElementSibling?.matches('.diagram-caption') ? visual.nextElementSibling : null;
            const scroll = visual.matches('.mermaid') ? visual : visual.querySelector('.visual-scroll');
            if (scroll) scroll.dataset.visualReviewScroll = ${JSON.stringify(visualId)};
            const rect = visual.getBoundingClientRect();
            const captionRect = caption?.getBoundingClientRect();
            const captureRect = captionRect ? {
              left: Math.min(rect.left, captionRect.left),
              top: Math.min(rect.top, captionRect.top),
              right: Math.max(rect.right, captionRect.right),
              bottom: Math.max(rect.bottom, captionRect.bottom),
            } : rect;
            const svgElements = [...visual.querySelectorAll('svg')];
            const clippedText = ${JSON.stringify(entry.medium === 'svg')} ? svgElements.flatMap((svg) => {
              const viewBox = svg.viewBox.baseVal;
              if (!viewBox?.width) return [];
              return [...svg.querySelectorAll('text')].flatMap((text) => {
                const box = text.getBBox();
                const epsilon = 0.5;
                return box.x < viewBox.x - epsilon || box.y < viewBox.y - epsilon ||
                  box.x + box.width > viewBox.x + viewBox.width + epsilon ||
                  box.y + box.height > viewBox.y + viewBox.height + epsilon
                  ? [text.textContent.trim()]
                  : [];
              });
            }) : [];
            const renderedTextSizes = svgElements.flatMap((svg) => {
              const viewBoxWidth = svg.viewBox.baseVal.width;
              const renderedWidth = svg.getBoundingClientRect().width;
              if (!viewBoxWidth || !renderedWidth) return [];
              return [...svg.querySelectorAll('text')].map((text) =>
                Number.parseFloat(getComputedStyle(text).fontSize) * renderedWidth / viewBoxWidth
              ).filter(Number.isFinite);
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
              scroll: scroll ? {
                clientWidth: scroll.clientWidth,
                scrollWidth: scroll.scrollWidth,
                overflowX: getComputedStyle(scroll).overflowX,
              } : null,
              svgCount: svgElements.length,
              titleCount: visual.querySelectorAll('svg > title').length,
              descCount: visual.querySelectorAll('svg > desc').length,
              minRenderedText: renderedTextSizes.length ? Math.min(...renderedTextSizes) : null,
              caption: caption?.textContent.trim() ?? '',
              captionGlyphWidth: rangeWidth(caption),
              clippedText,
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
        const label = `${entry.slug}/${visualId}/${mode.name}`;
        if (mode.mobile && details.innerWidth !== 390) throw new Error(`${label} reported innerWidth ${details.innerWidth}`);
        if (details.pageWidth > details.innerWidth + 1) throw new Error(`${label} creates page overflow (${details.pageWidth} > ${details.innerWidth})`);
        if (!details.figureFitsViewport) throw new Error(`${label} exceeds the viewport`);
        if (!details.caption.startsWith('Read it this way:') || details.captionGlyphWidth < 1) {
          throw new Error(`${label} lost its rendered direct caption`);
        }
        if (entry.medium === 'svg') {
          if (details.svgCount === 0 || details.svgCount !== details.titleCount || details.svgCount !== details.descCount) {
            throw new Error(`${label} lost SVG accessibility markup`);
          }
          if (details.clippedText.length) throw new Error(`${label} has text outside its viewBox: ${details.clippedText.join(', ')}`);
          if (mode.media !== 'print' && details.minRenderedText !== null && details.minRenderedText < 8) {
            throw new Error(`${label} renders text at ${details.minRenderedText.toFixed(2)}px`);
          }
          if (mode.media === 'print' && details.printMinWidths.some((width) => width !== '0px')) {
            throw new Error(`${label} retains an SVG minimum width in print`);
          }
        }

        const screenshot = await cdp.call('Page.captureScreenshot', {
          format: 'png',
          captureBeyondViewport: true,
          clip: { ...details.captureRect, scale: 1 },
        });
        const prefix = `${mode.name}--${entry.slug}--${visualId}`;
        writeFileSync(join(outputDir, `${prefix}.png`), Buffer.from(screenshot.data, 'base64'));

        if (details.scroll && details.scroll.scrollWidth > details.scroll.clientWidth + 1 && mode.media !== 'print') {
          await cdp.call('Runtime.evaluate', {
            expression: `(() => {
              const scroll = document.querySelector(${JSON.stringify(`[data-visual-review-scroll="${visualId}"]`)});
              scroll.scrollLeft = scroll.scrollWidth;
            })()`,
          });
          const endScreenshot = await cdp.call('Page.captureScreenshot', {
            format: 'png',
            captureBeyondViewport: true,
            clip: { ...details.captureRect, scale: 1 },
          });
          writeFileSync(join(outputDir, `${prefix}--scroll-end.png`), Buffer.from(endScreenshot.data, 'base64'));
        }
        report.push({ mode: mode.name, ...entry, visualId, ...details });
      }
    }
  }

  writeFileSync(join(outputDir, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
  console.log(`Visual rendering passed: ${report.length} renders; screenshots saved to ${outputDir}`);
} finally {
  cdp?.close();
  await stopBrowser();
}