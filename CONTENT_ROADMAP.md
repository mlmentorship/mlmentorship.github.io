# Content Roadmap

Tracker for what's been written for the relaunch and what's still on the inventory list from the blueprint.
This file is for internal planning; it isn't published as a site page.

**As of 2026-05-07 launch batch.**

## Targets

| Section | Launched | 6-month target | 12-month target |
|---|---|---|---|
| Essays | 6 | 8 | 15 |
| Interview Q&A | 50 | 50 | 50 |
| Reference | 27 | 60 | 100 |
| System Design | 1 | 4 | 8 |
| **Total** | **84** | **122** | **174** |

**Interview Q&A target hit at launch.** Remaining roadmap focuses on essays, reference pages, and system-design case studies.

---

## Done (launch batch, 2026-05-07)

### Essays (6 / 15)
- [x] LLM Evals: The hardest part of shipping LLMs
- [x] What L5 vs L6 actually means at FAANG ML
- [x] The 5 things every applied scientist interview is testing for
- [x] Designing a RAG system that actually works
- [x] How to think about LLM inference cost
- [x] Applied Scientist vs MLE vs Research Engineer

### Interview Q&A (50 / 50 — TARGET HIT)

**ML Fundamentals (8)**
- [x] How would you evaluate an LLM application you've built?
- [x] Walk me through the bias-variance tradeoff
- [x] Why does dropout work?
- [x] Explain backprop in your own words
- [x] When would you not use cross-validation?
- [x] L1 vs L2 regularization, beyond the formula
- [x] How do you choose a loss function?
- [x] Bayesian vs frequentist: a practitioner's framing

**Deep Learning Production (7)**
- [x] How would you debug a model that's not learning?
- [x] How do you choose a learning rate?
- [x] Walk me through how you'd train a 100B parameter model
- [x] Mixed precision: what's actually happening?
- [x] How do you deal with class imbalance in 2026?
- [x] Explain backprop through time
- [x] Why does Adam sometimes generalize worse than SGD?

**LLM Systems (10)**
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

**Recsys / Search / Ranking (7)**
- [x] Design YouTube's recommender
- [x] Two-tower vs cross-encoder: when to use which?
- [x] Design Spotify's homepage
- [x] How would you do cold-start for a new user?
- [x] How would you evaluate a search ranker?
- [x] Design Amazon's people also bought
- [x] Negative sampling strategies: what actually matters
- [x] Recsys in the LLM era: what changes?

**ML System Design (5)**
- [x] Design fraud detection for a payment company
- [x] Design a content moderation system
- [x] Design real-time personalization
- [x] Design a feature store from scratch
- [x] Design ML monitoring

**Behavioral / Applied Scientist (5)**
- [x] How do you decide what to work on?
- [x] Tell me about a time you disagreed with someone senior
- [x] What's the most over-rated technique in ML right now?
- [x] How do you scope an ambiguous problem?
- [x] Tell me about your most ambitious project

**Math (3)**
- [x] Derive logistic regression from MLE
- [x] Why is softmax + cross-entropy the right pairing?
- [x] Explain the reparameterization trick

**Coding (2)**
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

### Reference — gap-analysis batch (12 / 100, added 2026-05-31)

Written to close gaps found by diffing the CS/ML interview deck against published posts.

**Speech / sequence transduction**
- [x] Connectionist temporal classification (CTC)
- [x] RNN-Transducer (RNN-T)
- [x] Automatic speech recognition (ASR)

**Probabilistic graphical models**
- [x] Conditional random fields
- [x] Belief propagation
- [x] Factor analysis and probabilistic PCA

**Interpretability & training**
- [x] Model interpretability (LIME, SHAP, Grad-CAM)
- [x] Discrete gradient estimators (REINFORCE, Gumbel-Softmax, straight-through)
- [x] Neural network training recipe

**Retrieval & recsys**
- [x] TF-IDF and BM25
- [x] Knowledge-graph embeddings
- [x] Content-based filtering

---

## Remaining inventory (queued)

### Essays (7 remaining)
- [ ] Modern recsys: what changed since 2020
- [ ] The complete guide to LLM fine-tuning decisions in 2026
- [ ] Every model is a kernel method (and why this matters)
- [ ] The transformer-as-graph-neural-net lens
- [ ] Why batch norm and layer norm are the same idea
- [ ] What 'good ML taste' actually means
- [ ] What I'd tell my younger self about ML careers

### Reference (85 remaining, selected priority list)

**LLM internals (priority)**
- [ ] LoRA / QLoRA / parameter-efficient fine-tuning
- [ ] Continuous batching for LLM serving
- [ ] Paged attention / vLLM
- [ ] FP8 training
- [ ] Mixture of experts
- [ ] State space models / Mamba
- [ ] ReAct, function calling, tool use
- [ ] Self-consistency, chain-of-thought, tree-of-thought

**Modeling (priority)**
- [ ] Initialization (Xavier, He, orthogonal)
- [ ] Activation functions (ReLU, GELU, SwiGLU)
- [ ] Loss functions reference (MSE, cross-entropy, focal, contrastive)
- [ ] Embedding methods (Word2Vec, BERT-style, sentence-transformers)
- [ ] Sequence models (RNN, LSTM, GRU)
- [ ] Convolutional architectures (ResNet, EfficientNet)
- [ ] GNN basics (GCN, GAT, GraphSAGE)
- [ ] Diffusion models
- [ ] VAE
- [ ] GAN

**Statistics / theory (priority)**
- [ ] Maximum likelihood estimation
- [ ] Bayesian vs frequentist
- [ ] Hypothesis testing
- [ ] Multiple comparison correction
- [ ] Bootstrapping
- [ ] Cross-validation variants
- [ ] Information theory basics (entropy, KL, mutual info)

**ML systems (priority)**
- [ ] Feature stores
- [ ] Data versioning
- [ ] Model monitoring
- [ ] ML CI/CD
- [ ] Distributed training (DDP, FSDP, ZeRO, pipeline parallel, tensor parallel)
- [ ] Model serving (TensorRT, vLLM, Triton)

(The remaining ~50 reference pages are filled in opportunistically as gaps appear.)

### System Design (7 remaining)
- [ ] RAG for legal contracts (deep)
- [ ] Fraud detection at a payment company
- [ ] Content moderation with LLMs
- [ ] Personalized email subject lines
- [ ] Real-time recsys for short-form video
- [ ] Building an LLM coding assistant
- [ ] ML feature store from scratch

---

## Cadence going forward

Recommended cadence to hit 12-month targets:
- **Essays**: 1 every 3 weeks (7 more in next 6 months)
- **Interview Q&A**: 1 per week (~25 more in next 6 months)
- **Reference**: ~2 per week (~45 more in next 6 months)
- **System Design**: 1 per quarter (~3 more in next 6 months)

That hits roughly the 6-month milestones. Stretch to weekly essay + 2/week Q&A for the 12-month targets.

---

## Voice / style notes

- Opinionated, specific, factual.
- No fabricated stories, no personal project anecdotes.
- Each piece should have a strong claim, not a hedge.
- Cross-link aggressively to related essays / interviews / reference.
- Each essay ends with: "Related: [link], [link]". Each interview Q ends with the same.
- Reference pages are ~600 to 1000 words; interview Qs ~1500 to 2500; essays ~2500 to 5000; system design ~5000 to 10000.
- No em dashes.

## Things to add over time

- A logo refresh (current `logo.png` is 8+ years old)
- Hero images / social-share previews per essay (currently using default teaser)
- Newsletter signup (Buttondown or ConvertKit)
- Analytics (Plausible recommended)
- A search box for the site
- An RSS feed (atom.xml is configured but verify it's working post-launch)
