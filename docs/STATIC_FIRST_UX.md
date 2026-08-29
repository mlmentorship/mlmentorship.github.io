# Static-first UX direction

**Updated:** August 28, 2026

**Hosting:** GitHub Pages

**Product:** mlmentorship

## Decision

mlmentorship is an ML interview field guide with a private local Workbook. It is not a web application with a marketing site in front of it.

The root page is the table of contents. It should help a visitor choose one of three tasks:

1. read a book or chapter;
2. find an interview question;
3. continue a private Workbook task.

The site should not make a visitor pass through a large sales-style landing page before reaching the material.

## GitHub Pages is a hard constraint

Every core task must work from generated HTML, CSS, and static assets.

### Core features that fit

- normal links and directory-style routes;
- build-time content indexes;
- Pagefind search generated during the build;
- previous and next chapter links;
- HTML `details` navigation on small screens;
- downloadable files;
- RSS;
- optional browser-local progress;
- a custom static 404 page.

### Progressive enhancement only

JavaScript may improve a task, but it must not be the only way to read or navigate the library.

- Search may use JavaScript because Questions, Concepts, and Guides remain browsable without it.
- Theme choice may use local storage because the system theme remains a fallback.
- Practice progress may use local storage because questions and rubrics remain usable without saved state.
- Newsletter forms may use a third party only when RSS remains visible as a fallback.

### Do not add

- accounts or login;
- server sessions;
- cloud-synced progress;
- API-backed recommendations;
- a database;
- server-side search;
- required client-side routing;
- features that hide content until JavaScript runs;
- large page-wide script bundles;
- broad link prefetching across the library.

If one of these becomes necessary, it needs a separate hosting decision. It should not be simulated with fragile browser code.

## Information architecture

### Header

Keep four choices:

- Contents
- Questions
- Workbook
- About

The wordmark and Contents link to the table of contents. Search acts as the index. Reading paths are front matter, not another primary destination.

### Root page

Lead with the product promise: senior ML and AI interview preparation. Follow it with four shelves and nine subject books. Each book contains ordered chapters and combines concepts, questions, and guides. Keep type-specific indexes for direct and legacy links, but do not make users choose a publishing format before choosing a subject.

### Workbook

Use one `/prep/` surface for preparation state:

1. one saved role and round plan;
2. one dominant next action;
3. due and recent question attempts;
4. a three-step practice-method disclosure;
5. links to a role guide or simulation only when useful.

Do not place a second navigation bar inside every Workbook appendix. Weekly templates, current company processes, executable labs, stories, and final-week material are contextual appendices, not peer entry points. Legacy Practice and Progress routes redirect to the relevant Workbook section.

Readiness fields must start at "not attempted recently." Never assume a workable baseline. General coding, practical software, SQL, and generic systems can remain external curricula, but confirmed rounds must remain visible readiness dependencies. Save the resulting plan only in browser storage, expose one next task before later tasks, and support explicit JSON backup and restore.

### Reading paths

Keep the existing `/start-here/` URL for inbound links. Treat it as optional book front matter with a few cross-book routes. Do not put it in the primary header or repeat the interview workflow there.

### Books and category indexes

Book pages are the primary discovery surface. They show ordered chapter summaries, scope, difficulty, and local progress. Dedicated static chapter routes show the full ordered entry list, role relevance, interview rounds, and prerequisites. Questions, Concepts, and Guides indexes remain complete alternate views. No archive uses a persistent side rail.

### Article pages

Use this order:

1. a persistent desktop book rail, with a compact Book, Previous, Next, and Sections bar on smaller screens;
2. breadcrumb;
3. title and short description;
4. reading metadata;
5. one collapsed On this page row for article headings;
6. article;
7. previous and next links.

The desktop rail lists every chapter in the current book, expands the current chapter's ordered entries, and marks the current article with a quiet rule. On smaller screens the Sections menu provides the same hierarchy. Previous and Next traverse the whole book across chapter boundaries. The global Contents link continues to mean all books.

Use one centered reading column. Persistent left and right rails should not compete with the article. Repeated newsletter and author cards should not interrupt every chapter. The global footer can provide identity, RSS, and About links.

## Reading design

- Use one quiet system sans-serif stack for reading text, headings, and navigation.
- Keep code in a system monospace stack.
- Keep article text near 15px with a line length near 65 to 70 characters.
- Keep article titles below 2rem on normal desktop viewports; reserve larger type for the root table of contents.
- Use rules and whitespace more often than cards and shadows.
- Use muted text for metadata and context. Reserve the warm accent for links, focus, and rare primary actions.
- Prefer plain text controls over bordered buttons in persistent navigation.
- Do not repeat the same chapter list in summary and expanded forms on one page.
- Avoid hover movement on primary reading links.
- Preserve dark mode and reduced-motion preferences.

## Small-screen behavior

A hidden desktop sidebar must not remove all chapter context.

Article pages should expose two static controls:

- On this page, for the current article headings;
- Sections, for the current book, chapter, and sibling entries.

On this page and mobile Sections should use native `details` elements. They work without JavaScript and are keyboard accessible. On small screens the book bar keeps only the short book title, arrow controls, and Sections label. The desktop rail is plain static HTML and remains visible while the article scrolls.

## Performance rules

- Do not prefetch every visible link on index pages.
- Load Pagefind only after the user opens search.
- Load the newsletter provider only on a page that displays its form.
- Prefer system fonts to render-blocking third-party font stylesheets.
- Keep analytics asynchronous and non-blocking.
- Test the generated site, not only the development server.

## Acceptance checks

- A new visitor can reach Book I, Questions, or the Workbook in one click from the root page.
- A returning visitor can use search from every page.
- A phone user can move to a sibling article without returning to an index.
- All primary navigation works with JavaScript disabled.
- Browser-storage failure does not block reading or practice.
- The generated site passes internal-link validation.
- The deployment remains a single GitHub Pages artifact with no runtime server.
