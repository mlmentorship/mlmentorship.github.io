# User journey UX review

**Date:** August 28, 2026

**Site tested:** production build served locally from `dist`

**Scope:** first visit, subject browsing, exact search, chapter reading, role preparation, returning use, mobile, and no-JavaScript use

## Resolution status

The friction documented below was used as the implementation checklist. The August 28 resolution pass added:

- explicit pedagogical order for all 283 entries;
- smaller chapters with dedicated static chapter routes;
- correct placement for technical guides, pruning, distillation, evaluation, RAG, inference, and training material;
- inherited Core/Role-specific/Specialist, difficulty, role, round, and prerequisite metadata;
- dedicated static routes for each role path plus one-click Practice Mode starts;
- a saved browser-local readiness plan, next-three-task queue, week context, recalibration delta, and import/export;
- local study history for concepts, guides, role steps, labs, confident question attempts, and simulations;
- a timed multi-round simulation runner with private notes;
- Pagefind title weighting, Book/Shelf/Type filters, common aliases, and explicit missing-topic guidance;
- honest Published, Updated, and Reviewed date labels.
- one explicit identity: an ML interview field guide with a private local Workbook;
- four global destinations: Contents, Questions, Workbook, and About;
- one Workbook surface for plan, next task, retries, method, backup, and simulation handoff;
- permanent redirects from the former Practice-method and Progress pages into the Workbook;
- optional reading paths recast as book front matter rather than another interview-prep entry point.
- a persistent reader bar modeled on the Scaling Book pattern, with book context, cross-chapter Previous/Next actions, and a hierarchical Sections menu;
- separate Sections and On this page controls so book navigation no longer competes with article headings.
- one sans-serif visual system, 15px reading text, smaller titles, tighter spacing, softer color, and substantially less button and card chrome;
- compact Book and Chapter pages that show each navigation list once rather than repeating summaries and expanded descriptions.
- one sticky desktop library rail on every page; the current book is expanded, other books are collapsed, and tablet and mobile keep the compact article Sections bar.

Two suggestions were intentionally not implemented:

- Answers remain fully readable rather than collapsed behind JavaScript.
- Mixed-session passes remain individual confirmations because bulk completion would weaken the evidence rule.

### Final verification

- Astro reported 0 errors, 0 warnings, and 0 hints across 73 files.
- The production build indexed 352 pages and verified 358 generated HTML and RSS links.
- Browser tests restored a complete plan and study record from JSON, then serialized the same state through Export.
- Home, Book, Chapter, Article, Role path, Readiness, and Progress returned complete HTML without JavaScript and had no mobile horizontal overflow.
- Lighthouse scored Home and Article at 100 for Accessibility, Best Practices, SEO, and Agentic Browsing.

The detailed journeys below preserve the pre-fix observations that motivated the work. “Remaining friction” describes the baseline, not the final state.

## Bottom line

The site now behaves as an ordered online book rather than an alphabetical archive. A new user gets a default starting recommendation, can reach a subject in one click, and can reach an article in two clicks. Chapters state scope, sequence, priority, difficulty, prerequisites, role relevance, and interview rounds.

Candidates can choose one focused role route, create a private plan from current evidence, continue from one next action, and record several forms of preparation evidence. The complete reading and role structure remains usable as static HTML. Browser storage adds convenience without becoming a requirement.

## Test journeys: pre-fix baseline

### 1. First visit

**Task:** Understand the site and choose a starting point.

**Result:** Good.

The first screen states that the site is for senior ML and AI interview preparation. It names the four main roles and offers three direct starts:

- choose a role path;
- browse all interview questions;
- open the practice workflow.

The curriculum now groups 283 entries into four shelves and nine books. This is easier to understand than choosing Questions, Concepts, or Guides before choosing a subject.

**Remaining friction:** Nine books are still a large first decision. The shelf descriptions help, but there is no short recommendation such as “Most candidates start with Core ML; add one specialist shelf only if the role requires it.”

### 2. Find a subject

**Task:** Find distributed training and inference material.

**Path:** Home → ML systems and infrastructure.

**Result:** Good. One click reaches a page with two named chapters and 24 entries.

**Remaining friction:** The systems chapter mixes material with different prerequisites and purposes. For example, Transformer compute and memory accounting appears last because the list is alphabetical. Knowledge distillation and pruning also appear in Systems even though they fit model training better.

### 3. Study a subject in order

**Task:** Start linear algebra from the beginning.

**Observed order:** Determinant, Eigenvalues, Matrices as linear maps, Matrix calculus, Positive definite matrices, SVD and PCA.

**Result:** Poor for an online book. Matrices as linear maps should appear before determinant and eigenvalue material. Previous and next links repeat this alphabetical order.

The same problem appears in systems. Transformer accounting, hardware limits, collectives, sharded matrix multiplication, and parallelism selection need a designed sequence. Alphabetical order puts dependencies after advanced material.

### 4. Find one exact topic

**Task:** Search for FSDP.

**Result:** Good. The exact article appears first, followed by useful section matches and related pages.

**Task:** Search for causal inference.

**Result:** Poor. There is no dedicated causal-inference page. Search returns causal-attention material because it matches the word “causal.” The interface does not say that an exact subject is missing.

Search works well for known vocabulary. It is weaker for synonyms, missing subjects, and ambiguous terms. The result list can also become busy because one article contributes several section links.

### 5. Read an article

**Task:** Read Transformer compute and memory accounting.

**Result:** Good.

Strengths:

- one centered reading column;
- clear Book and Chapter breadcrumb;
- readable serif body text;
- short description and reading time;
- one collapsed Contents control;
- no required JavaScript;
- previous and next links at the end.

**Remaining friction:** Opening Contents shows 14 page sections and 16 chapter entries at once. On a phone, this becomes a long menu. It also exposes the alphabetical chapter order.

The metadata says “16 of 16 in this chapter.” This looks like learning progress, but it only reports alphabetical position.

### 6. Choose a role path

**Task:** Prepare for a Research Engineer interview.

**Result:** Useful but dense.

The role page contains four role paths, frontier format overlays, domain supplements, and 91 links. Anchors help, but users still land on one very long page. A candidate who chose Research Engineer continues to carry the other three roles below and above the selected path.

The global header uses “Paths” for broad reading routes. Prep uses “Path” for role paths. These two meanings are easy to confuse.

### 7. Run readiness

**Task:** Create a realistic starting plan.

**Result:** Honest but demanding.

The conservative defaults are good. Evidence starts at not attempted, and external coding, SQL, practical software, and general systems rounds can remain visible readiness blockers.

The form still contains 18 selects and 13 checkboxes in total. Four selects are under the external-round disclosure. A user must understand rounds, domains, evidence areas, level, runway, and workload before receiving a result. On mobile, this is a long form.

The result links to a generic weekly plan and a separate role path. It does not create one saved task list from the selected role, rounds, domain, gaps, and available time.

### 8. Return later

**Task:** Continue preparation after earlier attempts.

**Result:** Partial.

The Prep page prioritizes due question retries. The Progress page records question scores, weak dimensions, attempts, and due dates.

It does not track:

- role-path steps;
- books or chapters studied;
- concept checks;
- labs;
- simulations;
- the selected week of a plan;
- external round baselines.

Export exists, but import does not. Progress therefore remains tied to one browser unless the user manually reads the export.

### 9. Mobile and no-JavaScript use

**Result:** Strong overall.

The Home, Book, and Article pages work as static HTML without JavaScript. Native disclosure controls remain usable. Tested pages had no horizontal overflow except the role-path page on a very narrow viewport. That overflow came from long slash-separated wording and was fixed during this review.

The Home and Article layouts fit small screens well. Large Book pages remain long:

- ML foundations: 58 entries;
- Model training and research: 51 entries;
- LLMs, agents, and post-training: 41 entries.

A mobile user must scroll through every expanded entry description even when only one later chapter is relevant.

The baseline Lighthouse mobile checks returned 100 for accessibility and 100 for SEO. The final audit also reached 100 for Best Practices after the analytics script was changed to explicit HTTPS.

## Highest-impact friction and resolution

### P0. Chapters are not ordered for learning: resolved

The strongest mismatch is between book language and alphabetical behavior.

**Resolution:** Explicit chapter order now controls Chapter pages, article position, Contents, and previous and next links. Type-specific reference indexes remain alphabetical.

### P0. Several books and chapters are too broad: resolved

The largest flat chapters are:

- LLM Internals: 28 entries;
- Training Fundamentals: 24 entries;
- Systems and Infrastructure: 16 entries;
- Reinforcement Learning: 14 entries.

**Resolution:** The broad chapters were split into smaller routes, including:

- LLM architecture and attention;
- LLM inference and serving;
- post-training, evaluation, and alignment;
- optimization and numerical stability;
- training data and efficiency;
- hardware and roofline;
- sharding and parallelism;
- distributed reliability and profiling.

### P0. Some content is in the wrong book: resolved

All Guides currently appear under Interview and career practice. Several belong elsewhere:

- Marin 8B belongs in Model training and research;
- Personalized search ranking belongs in Retrieval and ranking;
- LLM inference cost belongs in LLMs or Systems;
- LLM evals belongs in Evaluation and product ML;
- Designing RAG belongs in LLMs and agents.

Knowledge distillation and pruning also fit Model training better than Systems.

**Resolution:** Every entry now has one explicit placement in the curriculum. The misplaced guides, distillation, and pruning material moved to the correct books.

### P1. Users still cannot see required versus optional material: resolved

A book page shows content type, but not:

- core or optional;
- role relevance;
- difficulty;
- prerequisite;
- expected interview round;
- recommended practice after reading.

**Resolution:** Chapter metadata provides priority, roles, difficulty, prerequisites, rounds, and ordered practice without a server or account.

### P1. Book pages are too long: resolved

The three largest books show 41 to 58 expanded entries. Descriptions help discovery, but make mobile pages very long.

**Resolution:** Book pages are short overviews. Dedicated static chapter routes contain the ordered entries and remain complete without JavaScript.

### P1. Paths has two meanings: resolved

Global Paths opens general reading paths. Prep Path opens role paths.

**Resolution:** Global navigation now uses Contents, Questions, Workbook, and About. Optional reading paths remain as book front matter; role preparation is contextual inside the Workbook.

### P1. Role preparation is one large page: resolved

Four role paths, format overlays, and domain supplements share one page with 91 links.

**Resolution:** Each role has a dedicated static route. The shared page is now a short four-role chooser with optional overlays collapsed.

### P1. Readiness does not produce a usable daily plan: resolved

Readiness ranks gaps but sends the user to two separate generic pages.

**Resolution:** Readiness saves role, rounds, gaps, workload, current week, and next tasks locally. Progress supports JSON export and import.

### P2. Search needs better missing-topic behavior: resolved

Exact technical terms work well. Ambiguous or missing topics can return misleading matches.

**Resolution:** Search adds aliases, exact-title weighting, Book/Shelf/Type filters, fewer section subresults, and explicit missing-topic guidance.

### P2. Date wording overstates review status: resolved

Article metadata uses “Reviewed” even when only the original `date` exists.

**Resolution:** Articles show Published for the original date and use Updated or Reviewed only when explicit metadata exists.

## Things that should not change

- Keep the clear first-screen product description.
- Keep subject-first shelves and books.
- Keep exact search available globally.
- Keep Questions as a direct interview-practice view.
- Keep one centered article column.
- Keep Contents collapsed by default.
- Keep static HTML as the complete fallback.
- Keep local-only preparation data and explicit privacy language.
- Do not add accounts, cloud state, server search, or client-only routing.

## Completed sequence

1. Explicit pedagogical order and previous and next navigation.
2. Smaller chapters and dedicated chapter routes.
3. Correct placement for guides and cross-cutting concepts.
4. Role, difficulty, prerequisite, round, and priority metadata.
5. Short Book overview pages.
6. A clear field-guide identity with one companion Workbook.
7. One browser-local plan generated from readiness evidence.
8. One dominant next action, with later tasks and appendices disclosed only on demand.
