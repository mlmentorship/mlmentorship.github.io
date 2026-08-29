# Content Roadmap

Tracker for what's been written for the relaunch and what's still on the inventory list from the blueprint.
This file is for internal planning; it isn't published as a site page.

**Updated 2026-08-29 after the AI-lab, scaling-systems, UX, editorial consistency, and upper-IC depth passes.**

For the next curriculum and product phase, see [docs/ML_CAREER_CURRICULUM_AND_UX_ROADMAP.md](docs/ML_CAREER_CURRICULUM_AND_UX_ROADMAP.md). It prioritizes ML-specific concept gaps and a role-to-round-to-practice workflow while preserving the decision not to duplicate generic LeetCode, SQL, or backend interview resources.

For the chapter-level scaling review, see [docs/JAX_SCALING_BOOK_CONTENT_AUDIT.md](docs/JAX_SCALING_BOOK_CONTENT_AUDIT.md). It keeps framework-neutral systems reasoning in the core library and treats JAX APIs as an optional specialist dependency.

For hosting and interface decisions, see [docs/STATIC_FIRST_UX.md](docs/STATIC_FIRST_UX.md). GitHub Pages is a hard constraint: core reading and navigation must work as generated HTML, CSS, and static assets.

For tested user journeys and their resolution status, see [docs/USER_JOURNEY_UX_REVIEW_2026-08-28.md](docs/USER_JOURNEY_UX_REVIEW_2026-08-28.md).

For writing rules and the full-library review, see [docs/EDITORIAL_STYLE.md](docs/EDITORIAL_STYLE.md) and [docs/CONTENT_STYLE_AUDIT_2026-08-28.md](docs/CONTENT_STYLE_AUDIT_2026-08-28.md).

## Targets

| Section | Current | Previous 12-month target | Status |
| --- | ---: | ---: | --- |
| Guides | 13 | 15 | Depth over volume |
| Interview Q&A | 85 | 50 | Exceeded |
| Concepts | 185 | 100 | Exceeded |
| Deep system-design case studies | 9 | 8 | Exceeded |
| **Total posts** | **283** | **174** | Breadth target exceeded |

**The library no longer needs broad reference expansion as its default.** Future work should prioritize realistic work samples, deep system-design cases, source maintenance, and revision of existing pages.

## Done: static-first reading UX, 2026-08-28

- [x] Lead the root page with one concise value proposition and place the complete curriculum immediately after it
- [x] Give every book a one-click Start reading action while keeping chapter overviews available
- [x] Keep the Workbook optional; offer a timed sample question and a compact returning-user link without making planning the pitch
- [x] Add book pages with ordered chapters that combine concepts, questions, and guides
- [x] Verify at build time that all 283 entries appear exactly once in the library taxonomy
- [x] Keep `/start-here/` as a focused reading-path page for existing links
- [x] Reduce the header to Contents, Questions, Workbook, and About
- [x] Use one system sans-serif stack without a third-party font request
- [x] Replace the three-column article shell with one reading column, one location rail, and a mobile Sections disclosure
- [x] Stop broad viewport prefetching across large indexes
- [x] Load the newsletter provider only on pages that render the form
- [x] Remove repeated newsletter and author cards from every article
- [x] Remove competing archive and right-side rails; keep one contextual book rail
- [x] Exclude archive listing text from Pagefind so exact article results rank first
- [x] Warn when local practice storage is unavailable
- [x] Keep a static custom 404 page with recovery links
- [x] Reduce prep navigation to Start, Path, Practice, Review, and Simulate
- [x] Replace the prep tool directory with one ordered workflow and a returning-user Continue card
- [x] Default readiness evidence to not attempted instead of a presumed workable answer
- [x] Surface external coding, SQL, practical software, and general systems dependencies without duplicating those curricula
- [x] Add a Research Scientist path, readiness overlay, and simulation
- [x] Make Practice Mode full-screen and shorter on mobile
- [x] Order every chapter pedagogically and use that order for article position and previous/next links
- [x] Split oversized chapters and add dedicated static chapter routes
- [x] Move misplaced technical guides and compression concepts to their subject books
- [x] Show chapter priority, difficulty, role relevance, rounds, and prerequisites
- [x] Split role preparation into four dedicated static paths with direct Practice Mode starts
- [x] Save readiness as a local plan with next tasks, current week, recalibration delta, and import/export
- [x] Track local study history across concepts, guides, role steps, labs, confident attempts, and simulations
- [x] Add a timed multi-round simulation runner with private scratch notes
- [x] Add search title weighting, filters, aliases, and missing-topic guidance
- [x] Display Published, Updated, and Reviewed dates honestly
- [x] Give undecided candidates a default Core-chapter starting sequence
- [x] Load analytics over explicit HTTPS and pass final Lighthouse Best Practices audits
- [x] Resolve the product identity as an ML interview field guide with one private Workbook
- [x] Consolidate the prep hub, practice method, and progress queue into one Workbook
- [x] Reduce global navigation to Contents, Questions, Workbook, and About
- [x] Demote generic schedules, role guides, and specialist tools to contextual appendices
- [x] Preserve old Practice and Progress URLs with permanent redirects
- [x] Add a persistent Book / Previous / Next / Sections reader bar
- [x] Make Previous and Next cross chapter boundaries within a book
- [x] Separate book Sections from article-level On this page navigation
- [x] Unify reading and interface typography around one minimal system font
- [x] Reduce persistent header and reader-bar height and remove button chrome
- [x] Remove repeated chapter indexes and entry descriptions from table-of-contents pages
- [x] Reduce article metadata, role badges, title scale, spacing, borders, and shadows
- [x] Add one sticky desktop library rail to every page; expand the current book and collapse the other eight by default
- [x] Pin the current book, chapter, and active entry inside long scrolling rails
- [x] Keep the compact Sections bar as the tablet and mobile fallback

## Done: editorial consistency pass, 2026-08-28

- [x] Audit all 283 concepts, questions, and guides
- [x] Give all 185 concepts an answer-first Summary section
- [x] Merge 143 separate stakes sections into their opening summaries
- [x] Remove every Why-it-matters heading
- [x] Replace high-confidence generated-prose templates with direct technical claims
- [x] Split dense sentences and shorten every description to 32 words or fewer
- [x] Preserve valid technical terms such as loss landscape and robust estimator
- [x] Enforce pyramid openings, sentence length, description length, banned phrases, and em-dash exclusion in the build

## Done: evidence-backed foundations batch, 2026-08-28

- [x] Expectation, variance, covariance, and correlation
- [x] Practical probability-distribution choice for ML
- [x] Bootstrap, paired resampling, and dependent-data variants
- [x] Data leakage and point-in-time correctness
- [x] Decision thresholds, asymmetric costs, and abstention
- [x] Causal inference for ML decisions
- [x] Integrate all six into ordered books, reference indexes, role guidance, search, and related reading

## Done: ranking, feedback, and lineage batch, 2026-08-28

- [x] Learning-to-rank pointwise, pairwise, listwise, and lambda objectives
- [x] Position bias, logging support, IPS, SNIPS, clipping, and doubly robust ranking
- [x] Delayed labels, selective labels, censoring, and policy feedback loops
- [x] Multi-task loss balancing, negative transfer, MMoE, and PLE
- [x] ML data lineage, versioning, replay, deletion, and rollback
- [x] Integrate all five into ordered books, role supplements, reference indexes, and existing practice pages

## Done: depth and upper-IC foundation batch, 2026-08-28

- [x] Publish a 6,000-word multi-team ML platform case with technical invariants, migration, ownership, adoption, cost, and portfolio decisions
- [x] Expand senior-level calibration through staff and principal scope without treating level as a separate job family
- [x] Add a reusable upper-IC level path that composes with AS, MLE, RS, and RE paths
- [x] Add principal readiness, evidence bars, Workbook routing, and a five-round level simulation
- [x] Extend the private story bank with technical-strategy, portfolio, durable-leverage, and succession evidence
- [x] Integrate the case into the Systems book, question taxonomy, domain supplements, search, and related reading

## Done: principal and senior-principal depth batch, 2026-08-29

- [x] Publish an 8,000-word enterprise agent-platform case covering authority, tool effects, durable state, memory, security, evaluation, migration, and multi-organization strategy
- [x] Publish a 4,000-word synthetic annotated mock with ten challenged turns, score movement, weak alternatives, and spaced retry drills
- [x] Extend level calibration through company-dependent senior-principal and distinguished scope
- [x] Add technical strategy as a selectable interview round
- [x] Add senior-principal readiness, a dedicated simulation, Workbook routing, and story evidence
- [x] Expand the upper-IC path to include both architecture cases and the annotated mock

## Done: high-value depth completion, 2026-08-29

- [x] Close the P0 concept set with entropy, conditional entropy, mutual information, information gain, and estimation limits
- [x] Publish a fixed-budget reasoning-model case and annotated upper-IC strategy mock
- [x] Publish a real-time multimodal assistant case spanning audio, video, screen, timing, privacy, and failure
- [x] Publish a short-form video ecosystem case and annotated senior-principal strategy mock
- [x] Publish a foundation-model data-platform case spanning provenance, mixtures, contamination, deletion, and infrastructure
- [x] Publish an AI coding-product case spanning repository context, safe execution, evaluation, developer control, and rollout
- [x] Publish an independent safety control-plane case for high-impact agents
- [x] Integrate all nine entries into books, indexes, domain paths, technical-strategy practice, simulations, search, and related reading

## Done: scaling-systems batch, 2026-08-28

- [x] Audit all twelve chapters of the JAX Scaling Book against the current library
- [x] Transformer parameter, FLOP, training-state, activation, and KV-cache accounting
- [x] Sharded matrix multiplication from tensor axes to collectives
- [x] GPU and TPU network topology without generation-specific tables
- [x] Strong scaling, MFU, and parallelism selection
- [x] Context parallelism and exact ring attention
- [x] Framework-neutral distributed workload profiling
- [x] Worked 70B model configuration to memory, time, cost, and layout question
- [x] Add disaggregated prefill and decode to the inference-service design
- [x] Correct collective, TP, FSDP, pipeline, checkpointing, and inference cost assumptions
- [x] Keep broad JAX API teaching outside the core path

## Done: AI-lab concept batch, 2026-08-28

- [x] Hypothesis testing and confidence intervals
- [x] Reproducibility and fair model comparison
- [x] Contrastive and self-supervised learning
- [x] Neural scaling laws and compute-optimal training
- [x] Markov decision processes and Bellman equations
- [x] LLM-as-judge evaluation
- [x] Evaluation validity and benchmark contamination
- [x] Synthetic data generation and verification
- [x] RL with verifiable rewards and GRPO
- [x] Test-time compute, search, and verifiers
- [x] Link the new concepts from research, post-training, alignment, multimodal, agent, and infrastructure domain paths

## Done: frontier-lab work-sample batch, 2026-07-11

- [x] Source-labeled OpenAI, Anthropic, Google DeepMind, Meta, and xAI process guide under Prep
- [x] Agentic multi-file ML evaluation lab
- [x] Broken frontier LLM training lab
- [x] Black-box model-behavior research lab
- [x] LLM inference scheduler lab and full system-design question
- [x] Anthropic public accelerator-challenge practice protocol
- [x] Timed ML math oral
- [x] Post-training environment and grader lab
- [x] Technical project presentation packet
- [x] Values and mission packet
- [x] Active implementation set: decoder, KV cache, beam search, LoRA, autodiff
- [x] Fault-tolerant collectives and distributed-training recovery
- [x] Mechanistic interpretability, monitorability, scalable oversight, AI control, and model organisms
- [x] LLM security threat modeling and red-team design
- [x] Preference data, reward models, RL environments, graders, and foundation-model data curation
- [x] Loss-spike diagnosis at scale
- [x] Multimodal foundation models and robotics policy learning
- [x] Proof-of-work, exceptional-work statement, artifacts, and references guide

---

## Done (launch batch, 2026-05-07)

### Essays (6 / 15)

- [x] LLM Evals: The hardest part of shipping LLMs
- [x] Senior through senior-principal ML scope
- [x] The 5 things every applied scientist interview is testing for
- [x] Designing a RAG system that actually works
- [x] How to think about LLM inference cost
- [x] Applied Scientist vs MLE vs Research Engineer

### Interview Q&A (50 / 50, TARGET HIT)

#### ML Fundamentals (8)

- [x] How would you evaluate an LLM application you've built?
- [x] Walk me through the bias-variance tradeoff
- [x] Why does dropout work?
- [x] Explain backprop in your own words
- [x] When would you not use cross-validation?
- [x] L1 vs L2 regularization, beyond the formula
- [x] How do you choose a loss function?
- [x] Bayesian vs frequentist: a practitioner's framing

#### Deep Learning Production (7)

- [x] How would you debug a model that's not learning?
- [x] How do you choose a learning rate?
- [x] Walk me through how you'd train a 100B parameter model
- [x] Mixed precision: what's actually happening?
- [x] How do you deal with class imbalance in 2026?
- [x] Explain backprop through time
- [x] Why does Adam sometimes generalize worse than SGD?

#### LLM Systems (10)

- [x] When would you fine-tune vs prompt vs RAG?
- [x] Fine-tuning vs prompting: the deep version
- [x] How do you handle hallucinations in production?
- [x] How would you reduce LLM inference cost by 10x?
- [x] Implement attention from scratch
- [x] Walk me through speculative decoding
- [x] Design a RAG system for legal documents
- [x] How do you evaluate an agent?
- [x] How would you build evals for a coding assistant?
- [x] How do you A/B test a chatbot?
- [x] Design a system for safe LLM deployment in healthcare
- [x] Build an LLM coding assistant from scratch

#### Recsys / Search / Ranking (7)

- [x] Design YouTube's recommender
- [x] Two-tower vs cross-encoder: when to use which?
- [x] Design Spotify's homepage
- [x] How would you do cold-start for a new user?
- [x] How would you evaluate a search ranker?
- [x] Design Amazon's people also bought
- [x] Negative sampling strategies: what actually matters
- [x] Recsys in the LLM era: what changes?

#### ML System Design (5)

- [x] Design fraud detection for a payment company
- [x] Design a content moderation system
- [x] Design real-time personalization
- [x] Design a feature store from scratch
- [x] Design ML monitoring

#### Behavioral / Applied Scientist (5)

- [x] How do you decide what to work on?
- [x] Tell me about a time you disagreed with someone senior
- [x] What's the most over-rated technique in ML right now?
- [x] How do you scope an ambiguous problem?
- [x] Tell me about your most ambitious project

#### Math (3)

- [x] Derive logistic regression from MLE
- [x] Why is softmax + cross-entropy the right pairing?
- [x] Explain the reparameterization trick

#### Coding (2)

- [x] Implement KNN efficiently
- [x] Debug this training loop

### Reference (15 / 100)

- [x] FlashAttention
- [x] BatchNorm vs LayerNorm
- [x] Speculative decoding
- [x] Cross-entropy and softmax
- [x] Adam, AdamW, and the modern optimizer landscape
- [x] KV cache
- [x] Regularization (L1/L2/dropout/early stopping)
- [x] Transformer architecture
- [x] RAG overview
- [x] Mixed precision training (FP16/BF16/FP8)
- [x] Tokenization (BPE/WordPiece/SentencePiece)
- [x] Quantization (INT8/INT4/FP8)
- [x] Calibration
- [x] RoPE / ALiBi positional encoding
- [x] RLHF and DPO

### System Design (1 / 8)

- [x] Personalized search ranking (end-to-end)

### Reference: gap-analysis batch (12 / 100, added 2026-05-31)

Written to close gaps found by diffing the CS/ML interview deck against published posts.

#### Speech / sequence transduction

- [x] Connectionist temporal classification (CTC)
- [x] RNN-Transducer (RNN-T)
- [x] Automatic speech recognition (ASR)

#### Probabilistic graphical models

- [x] Conditional random fields
- [x] Belief propagation
- [x] Factor analysis and probabilistic PCA

#### Interpretability & training

- [x] Model interpretability (LIME, SHAP, Grad-CAM)
- [x] Discrete gradient estimators (REINFORCE, Gumbel-Softmax, straight-through)
- [x] Neural network training recipe

#### Retrieval & recsys

- [x] TF-IDF and BM25
- [x] Knowledge-graph embeddings
- [x] Content-based filtering

---

## Demand-gated backlog

The declared P0 concept set and deep-case target are complete. New public content now requires observed demand, a missing interview transfer, or current specialist review.

### Maintenance

- [ ] Reverify frontier interview sources quarterly and date every material change
- [ ] Run all public lab fixtures against their stated Python and PyTorch versions
- [ ] Add downloadable archives only if individual-file access creates real user friction
- [ ] Review older pages for stale 2026 defaults and unsupported numeric claims
- [ ] Add source or confidence labels where a page currently presents a moving frontier as settled
- [ ] Track which labs candidates actually use before adding more formats

### Quality work requiring external evidence

- [ ] Publish observed mock packets only with informed consent and useful annotation; three synthetic upper-IC transfer mocks are complete
- [ ] Add hidden-test guidance for people facilitating the public implementation labs
- [ ] Commission domain review for robotics, multimodal, alignment, and accelerator pages
- [ ] Add visual architecture diagrams only where they improve a decision, not as decoration

---

## Cadence going forward

Recommended cadence:

- **One deep case study every 6 to 8 weeks**
- **One source-maintenance pass each quarter**
- **One technical review pass each month across a coherent cluster**
- **New questions or concepts only when a real loop, user request, or review reveals a gap**

Do not return to a weekly volume target. The library is already broad enough.

---

## Voice / style notes

- Opinionated, specific, factual.
- No fabricated stories, no personal project anecdotes.
- Each piece should have a strong claim, not a hedge.
- Cross-link aggressively to related essays / interviews / reference.
- Each essay ends with: "Related: [link], [link]". Each interview Q ends with the same.
- Reference pages are ~600 to 1000 words; interview Qs ~1500 to 2500; essays ~2500 to 5000; system design ~5000 to 10000.
- No em dashes.

## Product work over time

- Measure lab starts and completions without collecting candidate code or written answers
- Improve downloadable lab packaging if needed
- Add a visible source freshness marker to the frontier process page
- Keep Questions, Guides, and Concepts visually primary over Prep tooling
