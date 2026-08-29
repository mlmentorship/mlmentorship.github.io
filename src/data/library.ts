export type LibraryContentCategory = 'concepts' | 'questions' | 'guides';

export interface LibraryChapter {
  id: string;
  title: string;
  description: string;
  difficulty: 'Foundation' | 'Intermediate' | 'Advanced' | 'Mixed';
  priority: 'Core' | 'Role-specific' | 'Specialist';
  roles: string[];
  rounds: string[];
  prerequisites?: Array<{ label: string; href: string }>;
  slugs: string[];
}

export interface LibraryVolume {
  id: string;
  number: string;
  title: string;
  shortTitle: string;
  description: string;
  chapters: LibraryChapter[];
}

export interface LibraryShelf {
  id: string;
  title: string;
  description: string;
  volumeIds: string[];
}

export const LIBRARY_VOLUMES: LibraryVolume[] = [
  {
    id: 'foundations',
    number: 'I',
    title: 'ML foundations',
    shortTitle: 'Foundations',
    description: 'Math, probability, classical machine learning, deep learning, and the core questions that test them.',
    chapters: [
      {
        id: 'linear-algebra', title: 'Linear algebra and geometry', description: 'Start with linear maps, then study volume, spectra, decompositions, and derivatives.',
        difficulty: 'Foundation', priority: 'Core', roles: ['All ML roles'], rounds: ['Math', 'ML breadth'],
        slugs: ['matrices-as-linear-maps', 'determinant-and-volume', 'positive-definite-matrices', 'eigenvalues-and-spectral-theorem', 'svd-and-pca', 'matrix-calculus'],
      },
      {
        id: 'probability-statistics', title: 'Probability and statistics', description: 'Build from moments and distributions to estimation, uncertainty, testing, and resampling.',
        difficulty: 'Foundation', priority: 'Core', roles: ['AS', 'RS', 'RE', 'MLE'], rounds: ['Math', 'Statistics', 'Research'],
        slugs: ['expectation-variance-covariance-correlation', 'probability-distributions-in-ml', 'entropy-mutual-information', 'epistemic-vs-aleatoric-uncertainty', 'bayes-rule-and-posterior', 'maximum-likelihood-estimation', 'bias-variance-of-estimators', 'central-limit-theorem', 'hypothesis-testing-confidence-intervals', 'bootstrap-and-resampling', 'exponential-family', 'kl-divergence', 'monte-carlo-and-importance-sampling', 'markov-chains'],
      },
      {
        id: 'supervised-learning', title: 'Supervised learning', description: 'Move from linear models to kernels, trees, and ensembles.',
        difficulty: 'Foundation', priority: 'Core', roles: ['AS', 'MLE', 'RE'], rounds: ['ML breadth', 'Modeling'],
        prerequisites: [{ label: 'Probability and statistics', href: '/library/foundations/probability-statistics/' }],
        slugs: ['linear-regression', 'logistic-regression', 'naive-bayes', 'svm-and-kernels', 'kernel-methods-and-the-kernel-trick', 'decision-trees', 'random-forests', 'gradient-boosting'],
      },
      {
        id: 'unsupervised-learning', title: 'Unsupervised learning', description: 'Clustering and low-dimensional views of unlabeled data.',
        difficulty: 'Foundation', priority: 'Core', roles: ['AS', 'MLE', 'RE'], rounds: ['ML breadth'],
        prerequisites: [{ label: 'Linear algebra and geometry', href: '/library/foundations/linear-algebra/' }],
        slugs: ['k-means-clustering', 'dbscan', 'tsne-and-umap'],
      },
      {
        id: 'neural-networks', title: 'Neural network foundations', description: 'Understand expressiveness, optimization by backpropagation, and stable deep networks.',
        difficulty: 'Foundation', priority: 'Core', roles: ['All ML roles'], rounds: ['ML breadth', 'ML implementation'],
        prerequisites: [{ label: 'Linear algebra and geometry', href: '/library/foundations/linear-algebra/' }],
        slugs: ['universal-approximation-theorem', 'activation-functions', 'backpropagation', 'exploding-vanishing-gradients', 'residual-connections', 'attention-mechanism'],
      },
      {
        id: 'representations-architectures', title: 'Representations and architectures', description: 'Connect attention, encoders, self-supervision, graphs, and generative structure.',
        difficulty: 'Intermediate', priority: 'Core', roles: ['AS', 'RS', 'RE'], rounds: ['ML breadth', 'Research'],
        prerequisites: [{ label: 'Neural network foundations', href: '/library/foundations/neural-networks/' }],
        slugs: ['encoder-decoder-architectures', 'contrastive-self-supervised-learning', 'graph-neural-networks', 'autoregressive-vs-diffusion'],
      },
      {
        id: 'probabilistic-models', title: 'Probabilistic and latent-variable models', description: 'Build latent-variable and structured models before modern deep generative models.',
        difficulty: 'Intermediate', priority: 'Role-specific', roles: ['AS', 'RS', 'RE'], rounds: ['Math', 'Research', 'ML breadth'],
        prerequisites: [{ label: 'Probability and statistics', href: '/library/foundations/probability-statistics/' }],
        slugs: ['gaussian-processes', 'factor-analysis-and-ppca', 'expectation-maximization', 'gaussian-mixture-models', 'graphical-models', 'belief-propagation', 'hidden-markov-models', 'forward-backward-and-viterbi'],
      },
      {
        id: 'generative-models', title: 'Deep generative models', description: 'Study autoencoders, flows, adversarial learning, and diffusion in a useful sequence.',
        difficulty: 'Intermediate', priority: 'Role-specific', roles: ['AS', 'RS', 'RE'], rounds: ['ML breadth', 'Research'],
        prerequisites: [{ label: 'Probabilistic and latent-variable models', href: '/library/foundations/probabilistic-models/' }],
        slugs: ['variational-autoencoders', 'normalizing-flows', 'gans-overview', 'diffusion-models', 'discrete-gradient-estimators'],
      },
      {
        id: 'foundation-questions', title: 'Foundation interview questions', description: 'Test mechanism, assumptions, trade-offs, and changed conditions.',
        difficulty: 'Mixed', priority: 'Core', roles: ['All ML roles'], rounds: ['ML breadth', 'Math'],
        slugs: ['bias-variance-tradeoff', 'how-to-choose-loss-function', 'explain-backprop', 'why-does-dropout-work', 'l1-vs-l2-beyond-formula', 'how-to-choose-learning-rate', 'when-not-cross-validation', 'bayesian-vs-frequentist'],
      },
    ],
  },
  {
    id: 'training-research',
    number: 'II',
    title: 'Model training and research',
    shortTitle: 'Training & research',
    description: 'Optimization, reliable experiments, implementation, debugging, and research judgment.',
    chapters: [
      {
        id: 'objectives-regularization', title: 'Objectives and regularization', description: 'Start with training objectives, then control fit and generalization.',
        difficulty: 'Foundation', priority: 'Core', roles: ['All ML roles'], rounds: ['ML breadth', 'Training'],
        prerequisites: [{ label: 'Neural network foundations', href: '/library/foundations/neural-networks/' }],
        slugs: ['cross-entropy-softmax', 'regularization', 'dropout', 'label-smoothing', 'weight-decay-vs-l2', 'mixup-and-cutmix', 'z-loss'],
      },
      {
        id: 'optimization-schedules', title: 'Optimization and schedules', description: 'Move from SGD and Adam to schedules, clipping, and initialization.',
        difficulty: 'Intermediate', priority: 'Core', roles: ['All ML roles'], rounds: ['ML breadth', 'Training', 'Research'],
        prerequisites: [{ label: 'Objectives and regularization', href: '/library/training-research/objectives-regularization/' }],
        slugs: ['sgd-with-momentum', 'adam-and-adamw', 'learning-rate-schedules', 'wsd-and-wsd-s', 'microannealing', 'gradient-clipping', 'weight-initialization'],
      },
      {
        id: 'numerics-stability', title: 'Numerics and training stability', description: 'Understand normalization, precision, floating-point behavior, and loss failures.',
        difficulty: 'Intermediate', priority: 'Core', roles: ['MLE', 'RE', 'RS'], rounds: ['Training', 'Debugging'],
        prerequisites: [{ label: 'Optimization and schedules', href: '/library/training-research/optimization-schedules/' }],
        slugs: ['batchnorm-vs-layernorm', 'floating-point-formats', 'mixed-precision-training', 'loss-spikes-at-scale'],
      },
      {
        id: 'efficiency-data-scaling', title: 'Efficiency, data, and scaling', description: 'Reduce memory and work, build training data, and reason about scale.',
        difficulty: 'Advanced', priority: 'Role-specific', roles: ['RE', 'RS', 'MLE'], rounds: ['Training systems', 'Research'],
        prerequisites: [{ label: 'Numerics and training stability', href: '/library/training-research/numerics-stability/' }],
        slugs: ['activation-checkpointing', 'gradient-accumulation', 'sequence-packing', 'foundation-model-data-curation', 'synthetic-data-generation-verification', 'design-foundation-model-data-platform', 'neural-scaling-laws', 'knowledge-distillation', 'pruning'],
      },
      {
        id: 'training-recipe', title: 'Training recipe and case study', description: 'Connect the parts into one training plan, then inspect a real open training log.',
        difficulty: 'Intermediate', priority: 'Core', roles: ['AS', 'RS', 'RE', 'MLE'], rounds: ['Training', 'Project deep-dive'],
        prerequisites: [{ label: 'Optimization and schedules', href: '/library/training-research/optimization-schedules/' }],
        slugs: ['neural-network-training-recipe', 'lessons-from-marin-8b'],
      },
      {
        id: 'training-questions', title: 'Training and debugging questions', description: 'Practice diagnosis, numerical decisions, imbalance, sequence training, and optimizer trade-offs.',
        difficulty: 'Mixed', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Training', 'Debugging', 'ML breadth'],
        slugs: ['debug-model-not-learning', 'mixed-precision-deep', 'class-imbalance', 'bptt-backprop-through-time', 'adam-vs-sgd-generalization'],
      },
      {
        id: 'implementation', title: 'ML implementation', description: 'Implement core algorithms and transformer mechanisms with tests and explicit contracts.',
        difficulty: 'Mixed', priority: 'Core', roles: ['MLE', 'RE', 'RS'], rounds: ['ML implementation'],
        slugs: ['implement-knn', 'implement-reverse-mode-autograd', 'implement-attention-from-scratch', 'implement-transformer-decoder', 'implement-kv-cache-decode', 'implement-beam-search', 'implement-lora-adapter', 'implement-batched-top-k', 'implement-streaming-classification-metrics'],
      },
      {
        id: 'codebase-debugging', title: 'Training code and codebase debugging', description: 'Debug a training loop and extend an unfamiliar ML repository safely.',
        difficulty: 'Advanced', priority: 'Role-specific', roles: ['MLE', 'RE'], rounds: ['ML implementation', 'AI-assisted codebase'],
        slugs: ['debug-training-loop', 'agentic-ml-codebase-interview'],
      },
      {
        id: 'research-methods', title: 'Research methods and derivations', description: 'Derive, critique, design ablations, and investigate behavior from evidence.',
        difficulty: 'Advanced', priority: 'Core', roles: ['RS', 'RE', 'AS'], rounds: ['Math', 'Research', 'Work sample'],
        prerequisites: [{ label: 'Probability and statistics', href: '/library/foundations/probability-statistics/' }],
        slugs: ['derive-logistic-regression', 'softmax-cross-entropy-pairing', 'reparameterization-trick', 'derive-ml-math-under-pressure', 'design-ablation-study', 'critique-ml-paper', 'investigate-black-box-model-behavior'],
      },
    ],
  },
  {
    id: 'evaluation-product',
    number: 'III',
    title: 'Evaluation and product ML',
    shortTitle: 'Evaluation & product',
    description: 'Metrics, experimental validity, calibration, product decisions, and production evaluation.',
    chapters: [
      {
        id: 'metrics-calibration', title: 'Metrics, calibration, and decisions', description: 'Start with confusion counts, test probability quality, then choose an operating threshold or abstention rule.',
        difficulty: 'Foundation', priority: 'Core', roles: ['AS', 'MLE', 'RE', 'RS'], rounds: ['Evaluation', 'ML breadth'],
        slugs: ['confusion-matrix-and-classification-metrics', 'precision-recall-f1', 'roc-pr-auc', 'calibration', 'expected-calibration-error', 'decision-thresholds-asymmetric-costs-abstention'],
      },
      {
        id: 'evaluation-validity', title: 'Evaluation design and validity', description: 'Design splits and experiments that survive leakage, variance, and contamination.',
        difficulty: 'Intermediate', priority: 'Core', roles: ['AS', 'RS', 'RE', 'MLE'], rounds: ['Evaluation', 'Experimentation', 'Research'],
        prerequisites: [{ label: 'Probability and statistics', href: '/library/foundations/probability-statistics/' }],
        slugs: ['cross-validation-strategies', 'data-leakage-point-in-time-correctness', 'ab-testing-for-ml', 'reproducibility-fair-model-comparison', 'evaluation-validity-benchmark-contamination'],
      },
      {
        id: 'model-output-evaluation', title: 'Model output evaluation', description: 'Evaluate language, ranking, interpretability, and judge-based systems.',
        difficulty: 'Intermediate', priority: 'Role-specific', roles: ['AS', 'RS', 'MLE', 'Safety/evals'], rounds: ['Evaluation', 'Research', 'LLM systems'],
        slugs: ['perplexity-and-bits-per-token', 'ranking-metrics-ndcg-map-mrr', 'model-interpretability', 'llm-as-judge', 'llm-evals-the-hardest-part', 'how-would-you-evaluate-an-llm-application', 'evaluate-an-agent', 'evals-for-coding-assistant'],
      },
      {
        id: 'product-experimentation', title: 'Product and experimentation practice', description: 'Choose metrics, run valid tests, diagnose disagreement, and make a ship decision.',
        difficulty: 'Intermediate', priority: 'Core', roles: ['AS', 'Product MLE'], rounds: ['Product', 'Experimentation'],
        prerequisites: [{ label: 'Evaluation design and validity', href: '/library/evaluation-product/evaluation-validity/' }],
        slugs: ['causal-inference-for-ml-decisions', 'delayed-labels-selective-labels-feedback-loops', 'choose-ml-product-metrics', 'design-ml-ab-test', 'ab-test-chatbot', 'debug-offline-online-metric-gap'],
      },
    ],
  },
  {
    id: 'llms-agents',
    number: 'IV',
    title: 'LLMs, agents, and post-training',
    shortTitle: 'LLMs & agents',
    description: 'Transformer internals, inference, retrieval, evaluation, agents, alignment, and post-training.',
    chapters: [
      {
        id: 'transformer-attention', title: 'Transformer architecture and attention', description: 'Build from the transformer block to efficient and sparse attention variants.',
        difficulty: 'Intermediate', priority: 'Core', roles: ['RS', 'RE', 'MLE', 'AS'], rounds: ['ML breadth', 'LLM systems'],
        prerequisites: [{ label: 'Neural network foundations', href: '/library/foundations/neural-networks/' }],
        slugs: ['transformer-architecture', 'multi-head-attention', 'self-attention-vs-cross-attention', 'grouped-query-attention', 'flashattention', 'sparse-attention', 'linear-attention', 'mixture-of-experts'],
      },
      {
        id: 'tokens-position-context', title: 'Tokens, position, and long context', description: 'Understand input representation, positional information, and context extension.',
        difficulty: 'Intermediate', priority: 'Core', roles: ['RS', 'RE', 'MLE', 'AS'], rounds: ['ML breadth', 'LLM systems'],
        prerequisites: [{ label: 'Transformer architecture and attention', href: '/library/llms-agents/transformer-attention/' }],
        slugs: ['tokenization', 'positional-encoding', 'rotary-position-embeddings', 'long-context-llms'],
      },
      {
        id: 'inference-decoding', title: 'Inference and decoding', description: 'Move from decoding policy to cache layout, batching, speculation, and test-time compute.',
        difficulty: 'Advanced', priority: 'Role-specific', roles: ['RE', 'MLE', 'LLM engineer'], rounds: ['LLM systems', 'Inference'],
        prerequisites: [{ label: 'Transformer architecture and attention', href: '/library/llms-agents/transformer-attention/' }],
        slugs: ['decoding-strategies', 'kv-cache', 'prefill-vs-decode', 'paged-attention', 'continuous-batching', 'speculative-decoding', 'quantization', 'test-time-compute-search-verifiers', 'walk-through-speculative-decoding'],
      },
      {
        id: 'post-training-safety', title: 'Post-training, alignment, and safety', description: 'Preference learning, verifiable rewards, oversight, threats, and red-team design.',
        difficulty: 'Advanced', priority: 'Role-specific', roles: ['RS', 'RE', 'Safety/evals', 'Post-training'], rounds: ['Research', 'Post-training', 'Values'],
        prerequisites: [{ label: 'Reinforcement learning foundations', href: '/library/reinforcement-learning/foundations/' }],
        slugs: ['rlhf-and-dpo', 'preference-data-and-reward-models', 'rl-verifiable-rewards-grpo', 'scalable-oversight-and-ai-control', 'model-organisms-of-misalignment', 'llm-security-threat-models', 'design-post-training-data-and-rl-environment', 'design-llm-red-team-program', 'design-agent-safety-control-plane'],
      },
      {
        id: 'interpretability-monitoring', title: 'Interpretability and monitoring', description: 'Inspect internal mechanisms and reason about monitored chain-of-thought signals.',
        difficulty: 'Advanced', priority: 'Specialist', roles: ['RS', 'Safety/evals'], rounds: ['Research', 'Model behavior'],
        slugs: ['mechanistic-interpretability', 'chain-of-thought-monitorability'],
      },
      {
        id: 'fine-tuning-rag', title: 'Fine-tuning and retrieval-augmented generation', description: 'Choose adaptation or retrieval, then design and evaluate the resulting system.',
        difficulty: 'Intermediate', priority: 'Role-specific', roles: ['AS', 'MLE', 'LLM engineer'], rounds: ['LLM systems', 'ML design'],
        prerequisites: [{ label: 'Transformer architecture and attention', href: '/library/llms-agents/transformer-attention/' }],
        slugs: ['rag-overview', 'fine-tune-vs-prompt-vs-rag', 'fine-tuning-deep', 'rag-for-legal-docs', 'designing-rag-that-works'],
      },
      {
        id: 'llm-applications', title: 'LLM application and agent design', description: 'Handle failure, safety, tools, authority, and product integration in real applications.',
        difficulty: 'Advanced', priority: 'Role-specific', roles: ['AS', 'MLE', 'RE', 'LLM engineer'], rounds: ['LLM systems', 'ML design', 'Technical strategy'],
        slugs: ['handle-hallucinations-in-production', 'llm-deployment-healthcare', 'build-llm-coding-assistant', 'design-ai-coding-product', 'design-enterprise-agent-platform'],
      },
    ],
  },
  {
    id: 'systems',
    number: 'V',
    title: 'ML systems and infrastructure',
    shortTitle: 'Systems',
    description: 'Accelerators, distributed training, inference systems, reliability, cost, and full ML architecture.',
    chapters: [
      {
        id: 'hardware-performance', title: 'Hardware and performance', description: 'Count work and memory, understand accelerator limits, then read a trace.',
        difficulty: 'Advanced', priority: 'Role-specific', roles: ['RE', 'Systems MLE', 'Performance'], rounds: ['Systems', 'Performance'],
        prerequisites: [{ label: 'Neural network foundations', href: '/library/foundations/neural-networks/' }],
        slugs: ['transformer-compute-memory-accounting', 'gpu-memory-hierarchy', 'accelerator-network-topology', 'profiling-distributed-ml-workloads', 'optimize-accelerator-workload'],
      },
      {
        id: 'distributed-parallelism', title: 'Distributed training and parallelism', description: 'Build from collectives and sharded matrix operations to multi-axis parallel plans.',
        difficulty: 'Advanced', priority: 'Role-specific', roles: ['RE', 'Systems MLE', 'RS'], rounds: ['Systems', 'Training'],
        prerequisites: [{ label: 'Hardware and performance', href: '/library/systems/hardware-performance/' }],
        slugs: ['all-reduce-and-collectives', 'sharded-matrix-multiplication', 'fsdp-and-zero', 'tensor-parallelism', 'pipeline-parallelism', 'context-parallelism-and-ring-attention', 'strong-scaling-and-parallelism-selection'],
      },
      {
        id: 'training-reliability', title: 'Training plans and reliability', description: 'Size a run, choose a layout, and preserve consistent state through failures.',
        difficulty: 'Advanced', priority: 'Role-specific', roles: ['RE', 'Systems MLE', 'RS'], rounds: ['Systems design', 'Training incident'],
        prerequisites: [{ label: 'Distributed training and parallelism', href: '/library/systems/distributed-parallelism/' }],
        slugs: ['train-100b-model', 'plan-70b-training-run', 'fault-tolerant-collectives', 'design-fault-tolerant-distributed-training', 'debug-frontier-llm-training-run'],
      },
      {
        id: 'inference-systems', title: 'Inference systems and cost', description: 'Design a serving system, estimate cost, and choose measured optimizations.',
        difficulty: 'Advanced', priority: 'Role-specific', roles: ['MLE', 'RE', 'LLM engineer'], rounds: ['Inference', 'Systems design'],
        prerequisites: [{ label: 'Inference and decoding', href: '/library/llms-agents/inference-decoding/' }],
        slugs: ['design-production-llm-inference-service', 'design-reasoning-model-fixed-budget', 'annotated-reasoning-strategy-mock', 'reduce-llm-inference-cost-10x', 'llm-inference-cost'],
      },
      {
        id: 'ml-platforms', title: 'ML platforms and operations', description: 'Version data and models, then design serving, monitoring, budget, and freshness as one operating system.',
        difficulty: 'Advanced', priority: 'Core', roles: ['MLE', 'AS', 'RE'], rounds: ['ML system design'],
        slugs: ['ml-data-lineage-versioning', 'design-feature-store', 'design-ml-monitoring', 'design-multi-team-ml-platform', 'design-ml-system-fixed-budget', 'real-time-personalization'],
      },
      {
        id: 'applied-system-design', title: 'Applied ML system design', description: 'Apply the lifecycle to high-stakes classification and human-review systems.',
        difficulty: 'Advanced', priority: 'Core', roles: ['MLE', 'AS'], rounds: ['ML system design'],
        prerequisites: [{ label: 'ML platforms and operations', href: '/library/systems/ml-platforms/' }],
        slugs: ['design-fraud-detection', 'content-moderation'],
      },
    ],
  },
  {
    id: 'retrieval-ranking',
    number: 'VI',
    title: 'Retrieval, ranking, and recommendations',
    shortTitle: 'Retrieval & ranking',
    description: 'Embeddings, candidate generation, ranking, search metrics, cold start, and feedback loops.',
    chapters: [
      {
        id: 'retrieval-foundations', title: 'Retrieval foundations', description: 'Move from sparse retrieval and embeddings to approximate search and two-tower systems.',
        difficulty: 'Intermediate', priority: 'Role-specific', roles: ['Ranking MLE', 'AS', 'MLE'], rounds: ['Retrieval', 'ML design'],
        slugs: ['tf-idf-and-bm25', 'embedding-spaces-and-similarity', 'approximate-nearest-neighbors', 'two-tower-retrieval', 'knowledge-graph-embeddings', 'content-based-filtering'],
      },
      {
        id: 'recommendation-models', title: 'Recommendation models', description: 'Learn collaborative and feature-aware factorization methods.',
        difficulty: 'Intermediate', priority: 'Role-specific', roles: ['Ranking MLE', 'AS'], rounds: ['ML breadth', 'Recommendation design'],
        prerequisites: [{ label: 'Linear algebra and geometry', href: '/library/foundations/linear-algebra/' }],
        slugs: ['matrix-factorization-recsys', 'alternating-least-squares', 'factorization-machines'],
      },
      {
        id: 'ranking-practice', title: 'Retrieval and ranking practice', description: 'Choose ranking objectives, correct biased feedback, and defend retrieval, reranking, metrics, and sampling.',
        difficulty: 'Advanced', priority: 'Role-specific', roles: ['Ranking MLE', 'AS', 'MLE'], rounds: ['ML design', 'Evaluation'],
        prerequisites: [{ label: 'Retrieval foundations', href: '/library/retrieval-ranking/retrieval-foundations/' }],
        slugs: ['learning-to-rank-losses', 'position-bias-counterfactual-learning-to-rank', 'negative-sampling-strategies', 'two-tower-vs-cross-encoder', 'evaluate-search-ranker', 'personalized-search-ranking'],
      },
      {
        id: 'recommendation-design', title: 'Recommendation product design', description: 'Handle multi-task ranking, cold start, feedback, and ecosystem decisions.',
        difficulty: 'Advanced', priority: 'Role-specific', roles: ['Ranking MLE', 'AS'], rounds: ['ML system design', 'Product'],
        prerequisites: [{ label: 'Recommendation models', href: '/library/retrieval-ranking/recommendation-models/' }],
        slugs: ['multi-task-learning-objective-interference', 'design-youtube-recommender', 'design-short-form-video-ecosystem', 'annotated-ecosystem-strategy-mock', 'design-spotify-homepage', 'cold-start-new-user', 'people-also-bought', 'recsys-llm-era'],
      },
    ],
  },
  {
    id: 'reinforcement-learning',
    number: 'VII',
    title: 'Reinforcement learning and robotics',
    shortTitle: 'RL & robotics',
    description: 'Sequential decisions, value and policy methods, environments, rewards, and robotics policy learning.',
    chapters: [
      {
        id: 'foundations', title: 'RL foundations and value methods', description: 'Start with MDPs, then learn value-based control, exploration, and reward design.',
        difficulty: 'Intermediate', priority: 'Specialist', roles: ['RS', 'RE', 'Post-training', 'Robotics'], rounds: ['ML breadth', 'Research'],
        prerequisites: [{ label: 'Probability and statistics', href: '/library/foundations/probability-statistics/' }],
        slugs: ['mdps-and-bellman-equations', 'value-vs-policy-rl', 'q-learning', 'exploration-vs-exploitation', 'contextual-bandits', 'reward-shaping'],
      },
      {
        id: 'policy-optimization', title: 'Policy optimization', description: 'Build policy gradients, actor-critic methods, advantage estimation, and PPO.',
        difficulty: 'Advanced', priority: 'Specialist', roles: ['RS', 'RE', 'Post-training', 'Robotics'], rounds: ['Research', 'Post-training'],
        prerequisites: [{ label: 'RL foundations and value methods', href: '/library/reinforcement-learning/foundations/' }],
        slugs: ['policy-gradient', 'actor-critic-methods', 'advantage-estimation-and-gae', 'ppo'],
      },
      {
        id: 'environments-robotics', title: 'Environments, multiple agents, and robotics', description: 'Design environments and reason about interacting agents and learned policies in the physical world.',
        difficulty: 'Advanced', priority: 'Specialist', roles: ['RS', 'RE', 'Post-training', 'Robotics'], rounds: ['Research work sample', 'System design'],
        prerequisites: [{ label: 'Policy optimization', href: '/library/reinforcement-learning/policy-optimization/' }],
        slugs: ['rl-environments-and-graders', 'multi-agent-reinforcement-learning', 'robotics-policy-learning'],
      },
    ],
  },
  {
    id: 'vision-language-speech',
    number: 'VIII',
    title: 'Vision, language, and speech',
    shortTitle: 'Vision, language & speech',
    description: 'Visual models, multimodal systems, sequence modeling, natural language, and speech.',
    chapters: [
      {
        id: 'vision-foundations', title: 'Vision and multimodal foundations', description: 'Build from convolution to residual and transformer models, then study transfer and robustness.',
        difficulty: 'Intermediate', priority: 'Specialist', roles: ['Vision', 'Multimodal', 'AS', 'RS'], rounds: ['ML breadth', 'Research'],
        prerequisites: [{ label: 'Neural network foundations', href: '/library/foundations/neural-networks/' }],
        slugs: ['convolution-as-matmul', 'cnn-architecture', 'resnet', 'vision-transformers', 'domain-adaptation', 'adversarial-robustness', 'multimodal-foundation-models'],
      },
      {
        id: 'vision-tasks', title: 'Vision tasks', description: 'Move from object detection and suppression to dense semantic prediction.',
        difficulty: 'Intermediate', priority: 'Specialist', roles: ['Vision', 'Multimodal'], rounds: ['ML breadth', 'Model design'],
        prerequisites: [{ label: 'Vision and multimodal foundations', href: '/library/vision-language-speech/vision-foundations/' }],
        slugs: ['object-detection-overview', 'anchor-boxes-and-nms', 'semantic-segmentation'],
      },
      {
        id: 'language-representations', title: 'Language representations and sequence models', description: 'Follow text representation from embeddings and recurrence to bidirectional encoders and structured prediction.',
        difficulty: 'Intermediate', priority: 'Specialist', roles: ['NLP', 'Speech', 'AS', 'RS'], rounds: ['ML breadth', 'Research'],
        prerequisites: [{ label: 'Representations and architectures', href: '/library/foundations/representations-architectures/' }],
        slugs: ['word-embeddings', 'lstm-and-gru', 'bert-and-masked-language-modeling', 'conditional-random-fields'],
      },
      {
        id: 'speech-systems', title: 'Speech and real-time multimodal systems', description: 'Build from ASR objectives and streaming speech to a complete live multimodal assistant.',
        difficulty: 'Advanced', priority: 'Specialist', roles: ['Speech', 'Multimodal', 'RS', 'RE', 'MLE'], rounds: ['ML breadth', 'System design', 'Research'],
        prerequisites: [{ label: 'Language representations and sequence models', href: '/library/vision-language-speech/language-representations/' }],
        slugs: ['automatic-speech-recognition', 'connectionist-temporal-classification', 'rnn-transducer', 'streaming-asr', 'hybrid-vs-end-to-end-asr', 'speaker-recognition', 'design-real-time-multimodal-assistant'],
      },
    ],
  },
  {
    id: 'interview-practice',
    number: 'IX',
    title: 'Interview and career practice',
    shortTitle: 'Interview practice',
    description: 'Role choice, level calibration, project stories, behavioral judgment, and long-form field guides.',
    chapters: [
      {
        id: 'roles-levels', title: 'Roles and level calibration', description: 'Choose the right role, calibrate senior through senior-principal scope, and prepare for what interviews actually test.',
        difficulty: 'Foundation', priority: 'Core', roles: ['All candidates'], rounds: ['Recruiter', 'Hiring manager', 'Project'],
        slugs: ['as-vs-mle-vs-re', 'five-things-as-interview-tests', 'l5-vs-l6-faang-ml', 'annotated-upper-ic-agent-platform-mock'],
      },
      {
        id: 'frontier-process', title: 'Frontier application and process', description: 'Prepare evidence, artifacts, references, and format-specific expectations without relying on question rumors.',
        difficulty: 'Intermediate', priority: 'Role-specific', roles: ['Frontier lab candidates'], rounds: ['Application', 'Recruiter', 'References'],
        slugs: ['frontier-lab-proof-of-work-and-references', 'frontier-lab-interview-processes-2026'],
      },
      {
        id: 'behavioral-leadership', title: 'Behavioral and leadership practice', description: 'Move from scoping and ownership to conflict, failure, values, and judgment.',
        difficulty: 'Mixed', priority: 'Core', roles: ['All candidates'], rounds: ['Project deep-dive', 'Behavioral', 'Values'],
        slugs: ['scope-ambiguous-problem', 'decide-what-to-work-on', 'most-ambitious-project', 'present-technical-ml-project', 'disagreed-with-senior', 'advocated-quality-over-speed', 'killed-ml-project', 'defend-values-under-ethical-pressure', 'most-overrated-technique'],
      },
    ],
  },
];

export const LIBRARY_SHELVES: LibraryShelf[] = [
  {
    id: 'core-ml',
    title: 'Core ML',
    description: 'Build the technical base used across applied, research, and engineering interviews.',
    volumeIds: ['foundations', 'training-research', 'evaluation-product'],
  },
  {
    id: 'frontier-ai',
    title: 'Frontier AI systems',
    description: 'Study language models, post-training, agents, accelerators, and distributed systems.',
    volumeIds: ['llms-agents', 'systems'],
  },
  {
    id: 'specialist-tracks',
    title: 'Specialist tracks',
    description: 'Add only the specialist subject required by the role and team.',
    volumeIds: ['retrieval-ranking', 'reinforcement-learning', 'vision-language-speech'],
  },
  {
    id: 'interview-execution',
    title: 'Interview execution',
    description: 'Prepare role choice, project evidence, behavioral judgment, and senior-level communication.',
    volumeIds: ['interview-practice'],
  },
];

export function getLibraryVolume(id: string): LibraryVolume | undefined {
  return LIBRARY_VOLUMES.find((volume) => volume.id === id);
}

export function getLibraryShelf(volumeId: string): LibraryShelf | undefined {
  return LIBRARY_SHELVES.find((shelf) => shelf.volumeIds.includes(volumeId));
}

export function findLibraryLocation(slug: string): {
  volume: LibraryVolume;
  chapter: LibraryChapter;
} | undefined {
  for (const volume of LIBRARY_VOLUMES) {
    for (const chapter of volume.chapters) {
      if (chapter.slugs.includes(slug)) {
        return { volume, chapter };
      }
    }
  }
  return undefined;
}
