import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import process from 'node:process';

const root = resolve(import.meta.dirname, '..');
const generator = readFileSync(join(root, 'scripts/generate-coding-question-book.mjs'), 'utf8');
const client = readFileSync(join(root, 'src/utils/codingVisuals.ts'), 'utf8');
const styles = readFileSync(join(root, 'src/styles/global.css'), 'utf8');
const sample = readFileSync(join(root, 'src/content/posts/2026-09-01-two-sum.md'), 'utf8');

assert.match(generator, /data-coding-previous/, 'Previous control is required');
assert.match(generator, /data-coding-next/, 'Next control is required');
assert.match(generator, /data-coding-play/, 'Play control is required');
assert.match(client, /ArrowLeft|ArrowRight/, 'keyboard arrow semantics are required');
assert.match(client, /prefers-reduced-motion: reduce/, 'reduced-motion stepping is required');
assert.match(client, /data-motion-key/, 'stable motion-key interpolation is required');
assert.equal(client.match(/visual\.addEventListener\('keydown'/g)?.length, 1, 'one keydown handler per visual is required');
assert.match(styles, /@media print[\s\S]*coding-trace-frame/, 'print must reveal authored frames');
assert.match(styles, /@media \(max-width: 640px\)/, 'exact-mobile behavior is required');
assert.match(styles, /\.coding-trace-controls\s*\{[^}]*order:\s*-1;/, 'trace controls must stay above variable-height frames');
assert.match(sample, /data-coding-frame="0"(?![^>]*hidden)/, 'no-JS first frame must be visible');
assert.match(sample, /data-coding-controls hidden/, 'no-JS controls must stay hidden');

function option(name, fallback = '') {
	const prefix = `--${name}=`;
	return process.argv.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

const baseUrlOption = option('base-url');
if (!baseUrlOption) {
	console.log('Coding visual no-JS, keyboard, reduced-motion, mobile, print, and motion source checks passed.');
	process.exit(0);
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

const executable = browserExecutable();
if (!executable) throw new Error('Chrome or Edge was not found; set CHROME_PATH or --browser');
const baseUrl = baseUrlOption.replace(/\/$/, '');
const { codingQuestionVisuals } = await import('./coding-visuals/index.mjs');
const requestedSlugs = option('slugs').split(',').map((slug) => slug.trim()).filter(Boolean);
const slugs = requestedSlugs.length > 0 ? requestedSlugs : Object.keys(codingQuestionVisuals);
for (const slug of slugs) if (!codingQuestionVisuals[slug]) throw new Error(`Unknown coding visual slug: ${slug}`);
const routes = slugs.map((slug) => {
	const article = readFileSync(join(root, `src/content/posts/2026-09-01-${slug}.md`), 'utf8');
	const category = article.match(/^---\n[\s\S]*?^category:\s*["']?([^"'\n]+)["']?\s*$[\s\S]*?^---$/m)?.[1]?.trim();
	if (!category) throw new Error(`Could not read category for ${slug}`);
	return { slug, url: `${baseUrl}/${category}/${slug}/` };
});

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
				return;
			}
			const listeners = this.events.get(message.method) ?? [];
			this.events.delete(message.method);
			for (const listener of listeners) listener(message.params);
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

	close() {
		this.socket.close();
	}
}

const delay = (milliseconds) => new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
const port = 9800 + (process.pid % 500);
const profileDir = join(tmpdir(), `mlmentorship-coding-interactions-${process.pid}`);
rmSync(profileDir, { recursive: true, force: true });
mkdirSync(profileDir, { recursive: true });
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
		} catch {
			await delay(100);
		}
	}
}

async function evaluate(cdp, expression, awaitPromise = false) {
	let response;
	try {
		response = await cdp.call('Runtime.evaluate', { expression, returnByValue: true, awaitPromise });
	} catch (error) {
		throw new Error(`${error.message}; expression: ${expression.replace(/\s+/g, ' ').slice(0, 180)}`);
	}
	if (response.exceptionDetails) throw new Error(response.exceptionDetails.exception?.description ?? response.exceptionDetails.text);
	return response.result.value;
}

async function navigate(cdp, url) {
	const loaded = cdp.once('Page.loadEventFired');
	await cdp.call('Page.navigate', { url });
	await loaded;
	for (let attempt = 0; attempt < 30; attempt += 1) {
		await delay(50);
		const ready = await evaluate(cdp, `document.readyState === 'complete' && document.fonts.status === 'loaded'`);
		if (ready) break;
	}
	await delay(120);
}

const snapshotBody = `
	const activeFrame = visual.querySelector('[data-coding-frame]:not([hidden])');
	const keyed = [...activeFrame.querySelectorAll('[data-motion-key]')].map((element) => {
		const rect = element.getBoundingClientRect();
		const style = getComputedStyle(element);
		const descendantPaint = [...element.querySelectorAll('circle, rect, path, line, text')].map((child) => {
			const childStyle = getComputedStyle(child);
			return [childStyle.color, childStyle.backgroundColor, childStyle.borderColor, childStyle.fill, childStyle.stroke].join('|');
		}).join('||');
		return [element.dataset.motionKey, {
			x: rect.x, y: rect.y, width: rect.width, height: rect.height,
			text: element.textContent.trim(),
			style: [style.color, style.backgroundColor, style.borderColor, style.fill, style.stroke, descendantPaint].join('|'),
		}];
	});
	return {
		activeFrame: Number(visual.dataset.activeFrame),
		activeKey: activeFrame.dataset.frameKey,
		frameCount: visual.querySelectorAll('[data-coding-frame]').length,
		controlsTop: visual.querySelector('[data-coding-controls]').getBoundingClientRect().top,
		scrollY: window.scrollY,
		enhanced: visual.dataset.codingEnhanced,
		animationCount: typeof animationCount === 'undefined' ? 0 : animationCount,
		keyed,
	};
`;

function trackedDifference(before, after) {
	const previous = new Map(before.keyed);
	let appearance = false;
	let position = false;
	for (const [key, value] of after.keyed) {
		const prior = previous.get(key);
		if (!prior) continue;
		if (Math.abs(prior.x - value.x) > 0.5 || Math.abs(prior.y - value.y) > 0.5) position = true;
		if (prior.text !== value.text || prior.style !== value.style || prior.width !== value.width || prior.height !== value.height) appearance = true;
	}
	return { appearance, position };
}

const report = [];
let cdp;
try {
	await waitForBrowser();
	const targetResponse = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' });
	const target = await targetResponse.json();
	cdp = new Cdp(target.webSocketDebuggerUrl);
	await cdp.ready();
	await cdp.call('Page.enable');
	await cdp.call('Runtime.enable');
	await cdp.call('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });

	for (const route of routes) {
		await cdp.call('Emulation.setScriptExecutionDisabled', { value: false });
		await cdp.call('Emulation.setEmulatedMedia', { media: 'screen', features: [{ name: 'prefers-reduced-motion', value: 'no-preference' }] });
		await navigate(cdp, route.url);
		const initial = await evaluate(cdp, `(() => { const visual = document.querySelector('[data-coding-visual]'); ${snapshotBody} })()`);
		assert.equal(initial.enhanced, 'true', `${route.slug}: enhancement did not initialize`);
		assert.equal(initial.activeFrame, 0, `${route.slug}: initial frame is not zero`);
		let trackedAppearance = false;
		let trackedPosition = false;
		let animationObserved = false;
		let previous = initial;
		for (let frameIndex = 1; frameIndex < initial.frameCount; frameIndex += 1) {
			const transition = await evaluate(cdp, `(async () => {
				const visual = document.querySelector('[data-coding-visual]');
				visual.querySelector('[data-coding-next]').click();
				await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
				const animations = visual.getAnimations({ subtree: true }).filter((animation) => animation.playState === 'running');
				const animationCount = animations.length;
				for (const animation of animations) animation.finish();
				${snapshotBody}
			})()`, true);
			assert.equal(transition.activeFrame, frameIndex, `${route.slug}: Next skipped to frame ${transition.activeFrame}`);
			assert.ok(Math.abs(transition.controlsTop - initial.controlsTop) < 0.5, `${route.slug}: controls moved while advancing frames`);
			assert.equal(transition.scrollY, initial.scrollY, `${route.slug}: page scrolled while advancing frames`);
			const difference = trackedDifference(previous, transition);
			trackedAppearance ||= difference.appearance;
			trackedPosition ||= difference.position;
			animationObserved ||= transition.animationCount > 0;
			previous = transition;
		}
		assert.ok(trackedAppearance || trackedPosition, `${route.slug}: no keyed state changed across its frames`);
		if (trackedPosition) assert.ok(animationObserved, `${route.slug}: keyed positions changed without a live transition`);

		const previousIndex = await evaluate(cdp, `(() => { const visual = document.querySelector('[data-coding-visual]'); visual.querySelector('[data-coding-previous]').click(); return Number(visual.dataset.activeFrame); })()`);
		assert.equal(previousIndex, initial.frameCount - 2, `${route.slug}: Previous did not move exactly one frame`);
		const directIndex = Math.min(2, initial.frameCount - 1);
		const selectedIndex = await evaluate(cdp, `(() => { const visual = document.querySelector('[data-coding-visual]'); visual.querySelectorAll('[data-coding-frame-button]')[${directIndex}].click(); return Number(visual.dataset.activeFrame); })()`);
		assert.equal(selectedIndex, directIndex, `${route.slug}: direct selection failed`);
		const keyResults = await evaluate(cdp, `(() => {
			const visual = document.querySelector('[data-coding-visual]');
			const press = (key) => { visual.focus(); visual.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true })); return Number(visual.dataset.activeFrame); };
			return { home: press('Home'), right: press('ArrowRight'), end: press('End'), left: press('ArrowLeft') };
		})()`);
		assert.deepEqual(keyResults, { home: 0, right: 1, end: initial.frameCount - 1, left: initial.frameCount - 2 }, `${route.slug}: keyboard navigation skipped frames`);
		const playResult = await evaluate(cdp, `(async () => {
			const visual = document.querySelector('[data-coding-visual]');
			visual.dispatchEvent(new KeyboardEvent('keydown', { key: 'Home', bubbles: true }));
			visual.querySelector('[data-coding-play]').click();
			await new Promise((resolve) => setTimeout(resolve, 980));
			const result = { activeFrame: Number(visual.dataset.activeFrame), playing: visual.dataset.codingPlaying };
			if (visual.dataset.codingPlaying === 'true') visual.querySelector('[data-coding-play]').click();
			return result;
		})()`, true);
		assert.ok(playResult.activeFrame >= 1, `${route.slug}: Play did not visibly advance`);

		await cdp.call('Emulation.setEmulatedMedia', { media: 'screen', features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] });
		await navigate(cdp, route.url);
		const reduced = await evaluate(cdp, `(() => {
			const visual = document.querySelector('[data-coding-visual]');
			visual.querySelector('[data-coding-play]').click();
			return {
				activeFrame: Number(visual.dataset.activeFrame),
				playing: visual.dataset.codingPlaying,
				playLabel: visual.querySelector('[data-coding-play-label]').textContent.trim(),
				animations: visual.getAnimations({ subtree: true }).length,
			};
		})()`);
		assert.deepEqual(reduced, { activeFrame: 1, playing: 'false', playLabel: 'Next step', animations: 0 }, `${route.slug}: reduced motion must step once without autoplay`);

		await cdp.call('Emulation.setScriptExecutionDisabled', { value: true });
		await navigate(cdp, route.url);
		const noJs = await evaluate(cdp, `(() => {
			const visual = document.querySelector('[data-coding-visual]');
			const frames = [...visual.querySelectorAll('[data-coding-frame]')];
			return {
				enhanced: visual.dataset.codingEnhanced ?? null,
				firstVisible: !frames[0].hidden,
				hiddenRemainder: frames.slice(1).every((frame) => frame.hidden),
				controlsHidden: visual.querySelector('[data-coding-controls]').hidden,
				timelineHidden: visual.querySelector('[data-coding-timeline]').hidden,
				lessonLengths: [...visual.querySelectorAll('[data-coding-review]')].map((item) => item.textContent.trim().length),
				firstFrameText: frames[0].textContent.trim().length,
			};
		})()`);
		assert.equal(noJs.enhanced, null, `${route.slug}: no-JS page was unexpectedly enhanced`);
		assert.ok(noJs.firstVisible && noJs.hiddenRemainder && noJs.controlsHidden && noJs.timelineHidden, `${route.slug}: no-JS first-frame fallback is incomplete`);
		assert.equal(noJs.lessonLengths.length, 3, `${route.slug}: no-JS learning cues are missing`);
		assert.ok(noJs.lessonLengths.every((length) => length > 20) && noJs.firstFrameText > 20, `${route.slug}: no-JS content is not understandable`);
		report.push({ slug: route.slug, frames: initial.frameCount, trackedAppearance, trackedPosition, animationObserved });
	}

	const output = resolve(root, option('output', 'scratch/coding-interaction-report.json'));
	mkdirSync(resolve(output, '..'), { recursive: true });
	writeFileSync(output, `${JSON.stringify({ routes: report }, null, 2)}\n`);
	console.log(`Coding visual browser interactions passed: ${report.length} routes and ${report.reduce((sum, item) => sum + item.frames, 0)} authored frames.`);
} finally {
	cdp?.close();
	await stopBrowser();
}
