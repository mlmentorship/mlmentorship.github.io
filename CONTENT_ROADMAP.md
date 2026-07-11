# Content Roadmap

Tracker for what's been written for the relaunch and what's still on the inventory list from the blueprint.
This file is for internal planning; it isn't published as a site page.

**Updated 2026-07-11 after the frontier-lab work-sample batch.**

## Targets

| Section | Current | Previous 12-month target | Status |
| --- | ---: | ---: | --- |
| Guides | 10 | 15 | Depth over volume |
| Interview Q&A | 76 | 50 | Exceeded |
| Concepts | 157 | 100 | Exceeded |
| Deep system-design case studies | 1 | 8 | Still underweight |
| **Total posts** | **243** | **174** | Breadth target exceeded |

**The library no longer needs broad reference expansion as its default.** Future work should prioritize realistic work samples, deep system-design cases, source maintenance, and revision of existing pages.

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
- [x] What L5 vs L6 actually means at FAANG ML
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

## Current backlog

### Deep case studies

- [ ] Training and serving a frontier reasoning model under a fixed cluster budget
- [ ] End-to-end agent platform: environments, graders, training, deployment, and incident response
- [ ] Real-time multimodal assistant: audio, vision, tool use, latency, and privacy
- [ ] Short-form video recommendation with long-term ecosystem objectives
- [ ] Foundation-model data platform: provenance, filtering, mixture, deletion, and audit
- [ ] AI coding product: repository context, agent loop, sandbox, evals, and rollout
- [ ] Safety control plane for high-impact tool-using agents

### Maintenance

- [ ] Reverify frontier interview sources quarterly and date every material change
- [ ] Run all public lab fixtures against their stated Python and PyTorch versions
- [ ] Add downloadable archives only if individual-file access creates real user friction
- [ ] Review older pages for stale 2026 defaults and unsupported numeric claims
- [ ] Add source or confidence labels where a page currently presents a moving frontier as settled
- [ ] Track which labs candidates actually use before adding more formats

### Quality gaps

- [ ] Add one observed mock packet for every deep case study
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
