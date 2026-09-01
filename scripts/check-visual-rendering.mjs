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

const baseUrl = option('base-url', 'http://127.0.0.1:8123').replace(/\/$/, '');
const outputDir = resolve(root, option('output', 'scratch/visual-render-review'));
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
  return {
    slug,
    status: audit.status,
    category,
    medium: audit.medium,
    visualIds: audit.implementation.visualIds,
    article: basename(audit.article),
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

async function waitForVisualLayout(cdp, visualId, theme, label) {
  const evaluation = await cdp.call('Runtime.evaluate', {
    expression: `(async () => {
      document.documentElement.dataset.theme = ${JSON.stringify(theme)};
      await document.fonts.ready;
      const locateVisual = () => {
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
      let stableSince = performance.now();
      for (let attempt = 0; attempt < 40; attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, 50));
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        window.scrollTo({ left: 0, top: 0, behavior: 'instant' });
        const visual = locateVisual();
        if (!visual) throw new Error(${JSON.stringify(`${label}: visual not found`)});
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
        if (signature !== previous) {
          previous = signature;
          stableSince = now;
        } else if (now - stableSince >= 250) {
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
        await waitForVisualLayout(cdp, visualId, mode.theme, label);
        const expression = `(() => new Promise((resolve) => {
          document.documentElement.dataset.theme = ${JSON.stringify(mode.theme)};
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
            let visual = marker?.nextElementSibling;
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
              : marker?.previousElementSibling?.matches('.visual-title') ? marker.previousElementSibling : null;
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
              return [...svg.querySelectorAll('text')].flatMap((text) => {
                const box = text.getBBox();
                const rootMatrix = svg.getCTM();
                const textMatrix = text.getCTM();
                const matrix = rootMatrix && textMatrix
                  ? rootMatrix.inverse().multiply(textMatrix)
                  : { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };
                const corners = [
                  [box.x, box.y],
                  [box.x + box.width, box.y],
                  [box.x, box.y + box.height],
                  [box.x + box.width, box.y + box.height],
                ].map(([x, y]) => ({
                  x: matrix.a * x + matrix.c * y + matrix.e,
                  y: matrix.b * x + matrix.d * y + matrix.f,
                }));
                const bounds = {
                  left: Math.min(...corners.map((point) => point.x)),
                  top: Math.min(...corners.map((point) => point.y)),
                  right: Math.max(...corners.map((point) => point.x)),
                  bottom: Math.max(...corners.map((point) => point.y)),
                };
                const epsilon = 0.5;
                return bounds.left < viewBox.x - epsilon || bounds.top < viewBox.y - epsilon ||
                  bounds.right > viewBox.x + viewBox.width + epsilon ||
                  bounds.bottom > viewBox.y + viewBox.height + epsilon
                  ? [text.textContent.trim()]
                  : [];
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
              titleCount: visual.querySelectorAll('svg > title').length,
              descCount: visual.querySelectorAll('svg > desc').length,
              minRenderedText: renderedTextSizes.length ? Math.min(...renderedTextSizes) : null,
              visualTitle: visualTitle?.textContent.trim() ?? '',
              visualTitleGlyphWidth: rangeWidth(visualTitle),
              caption: captionText,
              rawCaptionMath: captionText.split('$').length > 2 || captionText.includes(String.fromCharCode(92)),
              captionGlyphWidth: rangeWidth(caption),
              clippedText,
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
        if (details.rawCaptionMath) throw new Error(`${label} retains raw math delimiters in its caption`);
        if (mode.media !== 'print' && details.minRenderedText !== null && details.minRenderedText < 8) {
          throw new Error(`${label} renders text at ${details.minRenderedText.toFixed(2)}px`);
        }
        if (entry.medium === 'svg') {
          if (details.svgCount === 0 || details.svgCount !== details.titleCount || details.svgCount !== details.descCount) {
            throw new Error(`${label} lost SVG accessibility markup`);
          }
          if (details.clippedText.length) throw new Error(`${label} has text outside its viewBox: ${details.clippedText.join(', ')}`);
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
      }
    }
  }

  writeFileSync(join(outputDir, 'report.json'), `${JSON.stringify({ renders: report, noVisualSlugs }, null, 2)}\n`);
  console.log(`Visual rendering passed: ${report.length} renders, ${noVisualSlugs.length} no-visual outcome(s); screenshots saved to ${outputDir}`);
} finally {
  cdp?.close();
  await stopBrowser();
}