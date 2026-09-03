import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const read = (path) => readFileSync(join(root, path), 'utf8');

const license = read('LICENSE');
const notices = read('THIRD_PARTY_NOTICES.md');
const packageJson = JSON.parse(read('package.json'));
const packageLock = JSON.parse(read('package-lock.json'));
const baseHead = read('src/components/BaseHead.astro');
const baseLayout = read('src/layouts/BaseLayout.astro');
const header = read('src/components/Header.astro');
const sidebar = read('src/components/LibrarySidebar.astro');
const review = read('src/pages/review.astro');
const home = read('src/pages/index.astro');

assert.equal(existsSync(join(root, 'LICENSE.txt')), false, 'legacy LICENSE.txt must stay removed');
assert.match(license, /PROPRIETARY LICENSE/);
assert.match(license, /Copyright \(c\) 2026 Hamidreza Saghir\. All rights reserved\./);
assert.doesNotMatch(license, /Permission is hereby granted|MIT License/);
assert.doesNotMatch(notices, /Michael Rose|original template lineage/);
assert.equal(packageJson.private, true);
assert.equal(packageJson.license, 'UNLICENSED');
assert.equal(packageLock.packages[''].license, 'UNLICENSED');

assert.match(baseLayout, /<html lang="en" data-theme="dark" data-library-sidebar="collapsed">/);
assert.match(baseHead, /saved === 'light' \? 'light' : 'dark'/);
assert.match(baseHead, /dataset\.librarySidebar = localStorage\.getItem\('library-sidebar'\) === 'expanded' \? 'expanded' : 'collapsed'/);
assert.match(header, /data-reading-mode=\{mode\.dataKey\}/);
assert.match(header, /pathname === '\/' \|\| isReviewRoute/);
assert.match(sidebar, /aria-label="Open library contents"/);
assert.match(sidebar, /aside\[data-library-sidebar\]/);
assert.match(review, /const desktopExplanation = window\.matchMedia\('\(min-width: 721px\)'\)/);
assert.match(review, /desktopExplanation\.addEventListener\('change', \(\) => setExplanation\(explanationOpen, false\)/);
assert.match(review, /class="review-contents-row"/);
assert.match(review, /contentsDetails\.open = false/);
assert.match(review, /max-height: min\(70dvh, 42rem/);
assert.match(review, /window\.addEventListener\('scroll', \(\) =>/);
assert.match(review, /document\.addEventListener\('pointerdown', \(event\) =>/);
assert.match(review, /querySelectorAll<HTMLAnchorElement>\('\[data-reading-mode="full"\]'\)/);
assert.match(home, /<h1>The ML Interview Field Guide<\/h1>/);
assert.match(home, /class="cover-primary" href="\/review\/"/);

console.log('Site contracts passed: dark visual-first defaults, collapsible contents, and proprietary ownership are enforced.');
