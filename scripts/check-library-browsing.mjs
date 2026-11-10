import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { once } from 'node:events';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { parse } from 'parse5';

const root = resolve(import.meta.dirname, '..');
const option = (name, fallback = '') => process.argv.find(argument => argument.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const baseUrl = option('base-url', 'http://127.0.0.1:4354').replace(/\/$/, '');
const executable = option('browser', process.env.CHROME_PATH);
assert(executable, '--browser or CHROME_PATH is required');
const output = resolve(option('output', join(tmpdir(), `mlmentorship-browsing-${process.pid}`)));
mkdirSync(output, { recursive: true });
const findNode = (node, predicate) => predicate(node) ? node : (node.childNodes ?? []).map(child => findNode(child, predicate)).find(Boolean);
const builtReview = parse(readFileSync(join(root, 'dist/review/index.html'), 'utf8'));
const dataNode = findNode(builtReview, node => node.tagName === 'script' && node.attrs.some(attribute => attribute.name === 'id' && attribute.value === 'visual-review-data'));
const entries = JSON.parse(dataNode.childNodes.map(node => node.value ?? '').join(''));
assert.equal(entries.length, 391);
const chapters = [...new Map(entries.map(entry => [`${entry.volumeId}/${entry.chapterId}`, entry])).values()];
const books = [...new Set(entries.map(entry => entry.volumeId))];
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const sourcePaths = ['src/components/Header.astro', 'src/components/Search.astro', 'src/components/ThemeToggle.astro', 'src/components/BookReaderNav.astro', 'src/components/LibrarySidebar.astro', 'src/components/VisualReviewCard.astro', 'src/components/VisualPostList.astro', 'src/layouts/BaseLayout.astro', 'src/layouts/DocsLayout.astro', 'src/pages/index.astro', 'src/pages/review.astro', 'src/pages/library/[volume].astro', 'src/pages/library/[volume]/[chapter].astro', 'data/visual-lessons.json'];
const sourceFiles = Object.fromEntries(sourcePaths.map(path => [path, digest(readFileSync(join(root, path)))]));
const cases = [];
const screenshots = [];

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
        if (message.error) pending.reject(new Error(`${pending.method}: ${message.error.message}`));
        else pending.resolve(message.result);
      } else {
        const listeners = this.events.get(message.method) ?? [];
        this.events.delete(message.method);
        listeners.forEach(listener => listener(message.params));
      }
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
    return new Promise(resolveEvent => {
      const listeners = this.events.get(method) ?? [];
      listeners.push(resolveEvent);
      this.events.set(method, listeners);
    });
  }
}

const profile = mkdtempSync(join(tmpdir(), 'mlmentorship-browsing-chrome-'));
const browser = spawn(executable, ['--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
let cdp;
let report;
try {
  const endpoint = await new Promise((resolveEndpoint, rejectEndpoint) => {
    let stderr = '';
    browser.stderr.on('data', chunk => {
      stderr += chunk.toString();
      const match = stderr.match(/DevTools listening on (ws:\/\/\S+)/);
      if (match) resolveEndpoint(match[1]);
    });
    browser.once('error', rejectEndpoint);
    browser.once('exit', code => rejectEndpoint(new Error(`Browser exited ${code}: ${stderr}`)));
  });
  const debugOrigin = new URL(endpoint).origin.replace('ws:', 'http:');
  const targets = await (await fetch(`${debugOrigin}/json/list`)).json();
  cdp = new Cdp(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
  await cdp.ready();
  await cdp.call('Page.enable');
  await cdp.call('Runtime.enable');
  await cdp.call('Network.enable');
  await cdp.call('Network.setCacheDisabled', { cacheDisabled: true });
  await cdp.call('Emulation.setFocusEmulationEnabled', { enabled: true });
  await cdp.call('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] });

  async function evaluate(expression) {
    const result = await cdp.call('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text);
    return result.result.value;
  }
  async function wait(expression) {
    return evaluate(`(async () => { const deadline = performance.now() + 15000; while (performance.now() < deadline) { if (${expression}) return true; await new Promise(resolveFrame => requestAnimationFrame(resolveFrame)); } throw new Error(${JSON.stringify(`Timed out: ${expression}`)}); })()`);
  }
  async function navigate(route) {
    let loaded = cdp.once('Page.loadEventFired');
    await cdp.call('Page.navigate', { url: 'about:blank' });
    await loaded;
    loaded = cdp.once('Page.loadEventFired');
    await cdp.call('Page.navigate', { url: `${baseUrl}${route}` });
    await loaded;
    await evaluate('document.fonts.ready.then(() => true)');
    if (route.includes('/review/#')) await wait('document.querySelector("[data-review-visual-host]")?.getAttribute("aria-busy") === "false"');
    else if (route.startsWith('/review/')) await wait('!document.querySelector("[data-review-index-tools]")?.hidden');
  }
  async function click(selector) {
    const point = await evaluate(`(() => { const element = document.querySelector(${JSON.stringify(selector)}); if (!element) throw new Error('Click target missing'); let box = element.getBoundingClientRect(); if (box.top < 0 || box.bottom > innerHeight) { element.scrollIntoView({block:'center',behavior:'instant'}); box = element.getBoundingClientRect(); } return {x:box.x + box.width / 2,y:box.y + box.height / 2}; })()`);
    await cdp.call('Input.dispatchMouseEvent', { type: 'mousePressed', button: 'left', clickCount: 1, ...point });
    await cdp.call('Input.dispatchMouseEvent', { type: 'mouseReleased', button: 'left', clickCount: 1, ...point });
  }
  async function key(value) {
    const keyCode = { ArrowLeft: 37, ArrowRight: 39, Escape: 27 }[value];
    await cdp.call('Input.dispatchKeyEvent', { type: 'keyDown', key: value, code: value, windowsVirtualKeyCode: keyCode });
    await cdp.call('Input.dispatchKeyEvent', { type: 'keyUp', key: value, code: value, windowsVirtualKeyCode: keyCode });
  }
  async function screenshot(name) {
    const result = await cdp.call('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
    const bytes = Buffer.from(result.data, 'base64');
    writeFileSync(join(output, `${name}.png`), bytes);
    screenshots.push({ path: `${name}.png`, sha256: digest(bytes) });
  }
  async function layout(label) {
    const result = await evaluate(`(() => { const visible = element => element.getClientRects().length && getComputedStyle(element).visibility !== 'hidden'; const header = [...document.querySelectorAll('.bar > .brand, .bar > [data-reading-mode-nav], .bar > .bar-end')].filter(visible).map(element => element.getBoundingClientRect().toJSON()); const overlap = header.some((box,index) => header.slice(index + 1).some(other => Math.min(box.right,other.right) - Math.max(box.left,other.left) > 1 && Math.min(box.bottom,other.bottom) - Math.max(box.top,other.top) > 1)); const unnamed = [...document.querySelectorAll('button')].filter(visible).filter(element => !element.textContent.trim() && !element.getAttribute('aria-label') && !element.getAttribute('title')).length; return {width:innerWidth,overflow:document.documentElement.scrollWidth > innerWidth + 1,overlap,unnamed}; })()`);
    assert.equal(result.overflow, false, `${label}: horizontal overflow`);
    assert.equal(result.overlap, false, `${label}: overlapping header controls`);
    assert.equal(result.unnamed, 0, `${label}: unnamed controls`);
    cases.push({ label, ...result });
  }

  for (const width of [320, 390, 768, 1280, 1920]) {
    await cdp.call('Emulation.setDeviceMetricsOverride', { width, height: 900, deviceScaleFactor: 1, mobile: false });
    await navigate('/');
    for (let index = 0; index < books.length; index += 1) {
      await evaluate(`document.querySelectorAll('[data-book-card] [data-landing-visual]')[${index}].scrollIntoView({block:'center',behavior:'instant'})`);
      await wait(`document.querySelectorAll('[data-book-card] [data-landing-visual]')[${index}].getAttribute('aria-busy') === 'false'`);
    }
    await evaluate('scrollTo({top:0,behavior:"instant"})');
    await layout(`home-${width}`);
    const placement = await evaluate(`(() => { const library = document.querySelector('#curriculum'); const featured = document.querySelector('.featured-posts'); const signup = document.querySelector('[data-email-form]'); return {books:document.querySelectorAll('[data-book-card]').length, beforeFeatured:Boolean(library.compareDocumentPosition(featured) & Node.DOCUMENT_POSITION_FOLLOWING), beforeSignup:Boolean(library.compareDocumentPosition(signup) & Node.DOCUMENT_POSITION_FOLLOWING)}; })()`);
    assert.equal(placement.books, 10);
    assert(placement.beforeFeatured && placement.beforeSignup);
    assert.equal(await evaluate('document.querySelectorAll("[data-book-card] [data-landing-visual][data-loaded=true]").length'), books.length, 'Every book preview must load');
    assert.equal(await evaluate('document.querySelector(\'[data-book-card][href="/library/training-research/"] [data-visual-lesson-step]\').dataset.visualLessonStep'), 'cross-entropy-confident-wrong-gradient:detail', 'Small book covers must use the compact authored diagram rather than a miniature long sequence');
    assert.equal(await evaluate('document.querySelector(\'[data-book-card][href="/library/evaluation-product/"] [data-visual-lesson-step]\').dataset.visualLessonStep'), 'confusion-matrix-denominators:mechanism', 'The evaluation book must have a compact authored drawing rather than a narrow calibration strip');
    const clipped = await evaluate(`(() => { const samples=[...document.querySelectorAll('[data-book-card] [data-landing-visual][data-loaded=true]')]; return samples.filter(preview => { const svg=preview.querySelector('svg'); if (!svg) return true; const frame=preview.getBoundingClientRect(); const drawing=svg.getBoundingClientRect(); return drawing.width < 1 || drawing.height < 1 || drawing.left < frame.left - 1 || drawing.right > frame.right + 1 || drawing.top < frame.top - 1 || drawing.bottom > frame.bottom + 1; }).length; })()`);
    assert.equal(clipped, 0, `home-${width}: clipped book artwork`);
    await screenshot(`home-${width}`);
    await navigate('/review/');
    await layout(`visual-library-${width}`);
    const startUrl = await evaluate('location.href');
    await key('ArrowRight');
    assert.equal(await evaluate('location.href'), startUrl);
    await click('[data-review-index-search]');
    await cdp.call('Input.insertText', { text: 'flashattention' });
    await wait('document.querySelector("[data-review-index-count]").textContent === "1 lesson"');
    assert.equal(await evaluate('document.querySelector("li:not([hidden]) [data-review-index-entry]").dataset.reviewIndexEntry'), 'flashattention');
    await screenshot(`visual-search-${width}`);
    await evaluate('document.querySelector("[data-review-index-search]").value = ""');
    await cdp.call('Input.insertText', { text: 'no-such-lesson-981' });
    await wait('!document.querySelector("[data-review-index-empty]").hidden');
    await click('[data-review-index-clear]');
    assert.equal(await evaluate('document.querySelector("[data-review-index-count]").textContent'), '391 lessons');
    assert.deepEqual(await evaluate('[...document.querySelectorAll("[data-review-index-volume][open]")].map(element => element.dataset.reviewIndexVolume)'), ['foundations']);
    cases.push({ label: `search-and-reset-${width}`, passed: true });
  }

  for (const width of [320, 1280]) {
    await cdp.call('Emulation.setDeviceMetricsOverride', { width, height: 900, deviceScaleFactor: 1, mobile: false });
    for (const book of books) {
      await navigate(`/library/${book}/`);
      await layout(`book-${book}-${width}`);
    }
    for (const chapter of chapters) {
      await navigate(`/library/${chapter.volumeId}/${chapter.chapterId}/`);
      await layout(`chapter-${chapter.chapterId}-${width}`);
      const href = await evaluate('document.querySelector("[data-reading-mode=visual]").getAttribute("href")');
      const scope = new URL(href, baseUrl).searchParams;
      assert.equal(scope.get('book'), chapter.volumeId);
      assert.equal(scope.get('chapter'), chapter.chapterId);
    }
  }

  for (const width of [390, 1280]) {
    await cdp.call('Emulation.setDeviceMetricsOverride', { width, height: 900, deviceScaleFactor: 1, mobile: false });
    for (const theme of ['light', 'dark']) {
      await navigate('/library/llms-agents/transformer-attention/');
      await evaluate(`localStorage.setItem('theme', ${JSON.stringify(theme)})`);
      await navigate('/review/?book=llms-agents&chapter=transformer-attention');
      await wait('document.querySelector("[data-review-index-chapter=transformer-attention]").open');
      assert.equal(await evaluate('document.querySelector("[data-reading-mode=full]").getAttribute("href")'), '/library/llms-agents/transformer-attention/');
      await screenshot(`scoped-library-${width}-${theme}`);
      await navigate('/review/#flashattention');
      await layout(`visual-lesson-${width}-${theme}`);
      assert.equal(await evaluate('document.querySelector("[data-review-subscribe]").hidden'), true);
      await screenshot(`visual-lesson-${width}-${theme}`);
      await evaluate('scrollTo({top:250,behavior:"instant"})');
      const returnScroll = await evaluate('scrollY');
      await click('[data-review-explanation-toggle]');
      await wait('document.querySelector("[data-review-explanation-toggle]").getAttribute("aria-expanded") === "true" && getComputedStyle(document.querySelector("[data-review-explanation]")).visibility === "visible"');
      if (width < 721) await wait('document.querySelector("[data-review-explanation]").getBoundingClientRect().top >= 0 && document.querySelector("[data-review-explanation]").getBoundingClientRect().top < innerHeight / 2');
      await layout(`explanation-${width}-${theme}`);
      await screenshot(`explanation-${width}-${theme}`);
      await click('[data-review-explanation-close]');
      await wait(`document.querySelector('[data-review-explanation]').hidden && Math.abs(scrollY - ${returnScroll}) < 2`);
      await click('[data-review-contents-details] > summary');
      await wait('document.querySelector("[data-review-contents-details]").open');
      assert.equal(await evaluate('document.querySelector("[data-review-content-entry][aria-current]").dataset.reviewContentEntry'), 'flashattention');
      assert.equal(await evaluate('(() => { const canvas = document.createElement("canvas"); canvas.width = canvas.height = 1; const context = canvas.getContext("2d"); context.fillStyle = getComputedStyle(document.querySelector(".review-contents-panel")).backgroundColor; context.fillRect(0, 0, 1, 1); return context.getImageData(0, 0, 1, 1).data[3]; })()'), 255, 'Contents must hide the drawing beneath the menu');
      await screenshot(`contents-${width}-${theme}`);
      await key('Escape');
      assert.equal(await evaluate('document.querySelector("[data-review-contents-details]").open'), false);
      await click('#search-trigger');
      await wait('document.querySelector("#search-dialog").open && document.querySelector(".pagefind-ui__search-input")');
      assert.equal(await evaluate('document.querySelector("#pagefind-host .pagefind-ui").classList.contains("pagefind-ui--reset")'), true, 'Search controls must use Pagefind reset styles rather than browser defaults');
      await click('.pagefind-ui__search-input');
      await cdp.call('Input.insertText', { text: 'FlashAttention' });
      await wait('document.querySelector(".pagefind-ui__result-link")?.getAttribute("href")?.startsWith("/review/#")');
      await screenshot(`global-search-${width}-${theme}`);
      await key('Escape');
      assert.equal(await evaluate('document.querySelector("#search-dialog").open'), false);
      if (width === 390 && theme === 'light') {
        await cdp.call('Emulation.setTouchEmulationEnabled', { enabled: true });
        await navigate('/review/#weight-decay-vs-l2');
        const originalHash = await evaluate('location.hash');
        async function swipe(selector, horizontal, vertical) {
          const point = await evaluate(`(() => { const element=document.querySelector(${JSON.stringify(selector)}); element.scrollIntoView({block:'center',behavior:'instant'}); const box=element.getBoundingClientRect(); return {x:box.x+Math.min(box.width / 2,160),y:Math.max(220,Math.min(innerHeight-150,box.top+60))}; })()`);
          await cdp.call('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ ...point, radiusX: 2, radiusY: 2, force: 1, id: 1 }] });
          await cdp.call('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: point.x + horizontal, y: point.y + vertical, radiusX: 2, radiusY: 2, force: 1, id: 1 }] });
          await cdp.call('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
          await evaluate('new Promise(resolveFrame => requestAnimationFrame(() => requestAnimationFrame(resolveFrame)))');
        }
        assert(await evaluate(`(() => { const canvas=[...document.querySelectorAll('[data-review-visual-host] .visual-lesson-canvas')].find(element => element.getBoundingClientRect().height > 0 && element.scrollWidth > element.clientWidth); if (!canvas) return false; canvas.dataset.browsingTouch = 'true'; return true; })()`), 'Touch test needs a genuinely scrollable diagram');
        await swipe('[data-browsing-touch]', -90, 0);
        await wait('document.querySelector("[data-browsing-touch]").scrollLeft > 0');
        assert.equal(await evaluate('location.hash'), originalHash, 'Diagram panning must not change lessons');
        await swipe('[data-review-title]', 80, 120);
        assert.equal(await evaluate('location.hash'), originalHash, 'Vertical scrolling must not change lessons');
        await swipe('[data-review-title]', -90, 0);
        await wait(`location.hash !== ${JSON.stringify(originalHash)} && document.querySelector('[data-review-visual-host]').getAttribute('aria-busy') === 'false'`);
        await cdp.call('Emulation.setTouchEmulationEnabled', { enabled: false });
        await click('[data-review-previous]');
        await wait(`location.hash === ${JSON.stringify(originalHash)} && document.querySelector('[data-review-visual-host]').getAttribute('aria-busy') === 'false'`);
        cases.push({ label: 'native-touch-panning', passed: true }, { label: 'native-vertical-scroll', passed: true }, { label: 'native-horizontal-lesson-swipe', passed: true });
        await navigate('/review/#flashattention');
      }
      const hash = await evaluate('location.hash');
      await click('[data-review-next]');
      await wait(`location.hash !== ${JSON.stringify(hash)} && document.querySelector('[data-review-visual-host]').getAttribute('aria-busy') === 'false'`);
      await key('ArrowLeft');
      await wait(`location.hash === ${JSON.stringify(hash)} && document.querySelector('[data-review-visual-host]').getAttribute('aria-busy') === 'false'`);
      const articleLoaded = cdp.once('Page.loadEventFired');
      await click('[data-reading-mode=full]');
      await articleLoaded;
      await wait('location.pathname === "/concepts/flashattention/" && document.readyState === "complete"');
      await evaluate('document.fonts.ready.then(() => document.activeViewTransition?.finished).then(() => true)');
      await layout(`full-lesson-${width}-${theme}`);
      await screenshot(`full-lesson-${width}-${theme}`);
      await click('.reader-tool > summary');
      await wait('document.querySelector(".reader-tool").open');
      assert(await evaluate('document.querySelectorAll(".reader-menu a").length > 2'));
      await screenshot(`article-sections-${width}-${theme}`);
      if (width < 1100) {
        await click('[data-site-menu] > summary');
        await wait('document.querySelector("[data-site-menu]").open');
        await key('Escape');
        assert.equal(await evaluate('document.querySelector("[data-site-menu]").open'), false);
      }
      cases.push({ label: `reading-round-trip-${width}-${theme}`, passed: true });
    }
  }
  for (const [path, expected] of Object.entries(sourceFiles)) assert.equal(digest(readFileSync(join(root, path))), expected, `Source changed during browsing validation: ${path}`);
  const reviewResponse = Buffer.from(await (await fetch(`${baseUrl}/review/`)).arrayBuffer());
  assert(reviewResponse.equals(readFileSync(join(root, 'dist/review/index.html'))), 'Live review HTML must equal the build');
  report = { complete: true, baseUrl, sourceFiles, books: books.length, chapters: chapters.length, cases, screenshots, reviewHtmlSha256: digest(reviewResponse) };
} finally {
  if (browser.exitCode === null) {
    const exited = once(browser, 'exit');
    if (cdp) await cdp.call('Browser.close');
    else browser.kill('SIGTERM');
    await exited;
  }
  cdp?.socket.close();
  rmSync(profile, { recursive: true, force: true });
}

writeFileSync(join(output, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
console.log(`Browsing validation passed: ${cases.length} cases, ${books.length} books, ${chapters.length} chapters. Evidence: ${output}`);
