// Shared subcategory taxonomy used by both the category index pages and the
// left sidebar (SectionNav). Keeping this in one place ensures the grouping
// stays consistent across surfaces.
//
// Slugs not listed in a category's map fall into 'Other' and render at the
// bottom of the list.

export const INTERVIEW_SUBCATEGORY: Record<string, string> = {
  'bias-variance-tradeoff': 'ML Fundamentals',
  'why-does-dropout-work': 'ML Fundamentals',
  'explain-backprop': 'ML Fundamentals',
  'when-not-cross-validation': 'ML Fundamentals',
  'l1-vs-l2-beyond-formula': 'ML Fundamentals',
  'how-to-choose-loss-function': 'ML Fundamentals',
  'bayesian-vs-frequentist': 'ML Fundamentals',
  'how-to-choose-learning-rate': 'ML Fundamentals',

  'debug-model-not-learning': 'Deep Learning Production',
  'train-100b-model': 'Deep Learning Production',
  'mixed-precision-deep': 'Deep Learning Production',
  'class-imbalance': 'Deep Learning Production',
  'bptt-backprop-through-time': 'Deep Learning Production',
  'adam-vs-sgd-generalization': 'Deep Learning Production',

  'how-would-you-evaluate-an-llm-application': 'LLM Systems',
  'fine-tune-vs-prompt-vs-rag': 'LLM Systems',
  'fine-tuning-deep': 'LLM Systems',
  'handle-hallucinations-in-production': 'LLM Systems',
  'reduce-llm-inference-cost-10x': 'LLM Systems',
  'walk-through-speculative-decoding': 'LLM Systems',
  'rag-for-legal-docs': 'LLM Systems',
  'evaluate-an-agent': 'LLM Systems',
  'evals-for-coding-assistant': 'LLM Systems',
  'llm-deployment-healthcare': 'LLM Systems',
  'build-llm-coding-assistant': 'LLM Systems',

  'design-youtube-recommender': 'Recsys & Search',
  'two-tower-vs-cross-encoder': 'Recsys & Search',
  'design-spotify-homepage': 'Recsys & Search',
  'cold-start-new-user': 'Recsys & Search',
  'evaluate-search-ranker': 'Recsys & Search',
  'people-also-bought': 'Recsys & Search',
  'negative-sampling-strategies': 'Recsys & Search',
  'recsys-llm-era': 'Recsys & Search',

  'design-fraud-detection': 'ML System Design',
  'content-moderation': 'ML System Design',
  'real-time-personalization': 'ML System Design',
  'design-feature-store': 'ML System Design',
  'design-ml-monitoring': 'ML System Design',
  'design-ml-system-fixed-budget': 'ML System Design',

  'ab-test-chatbot': 'Product & Experimentation',
  'design-ml-ab-test': 'Product & Experimentation',
  'debug-offline-online-metric-gap': 'Product & Experimentation',
  'choose-ml-product-metrics': 'Product & Experimentation',

  'decide-what-to-work-on': 'Behavioral',
  'disagreed-with-senior': 'Behavioral',
  'most-overrated-technique': 'Behavioral',
  'scope-ambiguous-problem': 'Behavioral',
  'most-ambitious-project': 'Behavioral',
  'advocated-quality-over-speed': 'Behavioral',
  'killed-ml-project': 'Behavioral',
  'present-technical-ml-project': 'Behavioral',
  'defend-values-under-ethical-pressure': 'Behavioral',

  'derive-logistic-regression': 'Math & Research',
  'softmax-cross-entropy-pairing': 'Math & Research',
  'reparameterization-trick': 'Math & Research',
  'design-ablation-study': 'Math & Research',
  'critique-ml-paper': 'Math & Research',
  'investigate-black-box-model-behavior': 'Math & Research',
  'derive-ml-math-under-pressure': 'Math & Research',

  'implement-knn': 'ML Implementation',
  'debug-training-loop': 'ML Implementation',
  'implement-attention-from-scratch': 'ML Implementation',
  'implement-batched-top-k': 'ML Implementation',
  'implement-streaming-classification-metrics': 'ML Implementation',
  'agentic-ml-codebase-interview': 'ML Implementation',
  'debug-frontier-llm-training-run': 'ML Implementation',
  'optimize-accelerator-workload': 'ML Implementation',
  'implement-transformer-decoder': 'ML Implementation',
  'implement-kv-cache-decode': 'ML Implementation',
  'implement-beam-search': 'ML Implementation',
  'implement-lora-adapter': 'ML Implementation',
  'implement-reverse-mode-autograd': 'ML Implementation',

  'design-production-llm-inference-service': 'ML System Design',
  'design-fault-tolerant-distributed-training': 'ML System Design',

  'design-post-training-data-and-rl-environment': 'LLM Systems',
  'design-llm-red-team-program': 'LLM Systems',
};

export const INTERVIEW_ORDER = [
  'ML Fundamentals',
  'Math & Research',
  'ML Implementation',
  'ML System Design',
  'Product & Experimentation',
  'Deep Learning Production',
  'LLM Systems',
  'Recsys & Search',
  'Behavioral',
];

export const REFERENCE_SUBCATEGORY: Record<string, string> = {
  // Linear Algebra & Math
  'matrices-as-linear-maps': 'Linear Algebra & Math',
  'svd-and-pca': 'Linear Algebra & Math',
  'eigenvalues-and-spectral-theorem': 'Linear Algebra & Math',
  'determinant-and-volume': 'Linear Algebra & Math',
  'positive-definite-matrices': 'Linear Algebra & Math',
  'matrix-calculus': 'Linear Algebra & Math',

  // Probability & Statistics
  'maximum-likelihood-estimation': 'Probability & Statistics',
  'bias-variance-of-estimators': 'Probability & Statistics',
  'bayes-rule-and-posterior': 'Probability & Statistics',
  'kl-divergence': 'Probability & Statistics',
  'central-limit-theorem': 'Probability & Statistics',
  'exponential-family': 'Probability & Statistics',
  'markov-chains': 'Probability & Statistics',
  'monte-carlo-and-importance-sampling': 'Probability & Statistics',
  'epistemic-vs-aleatoric-uncertainty': 'Probability & Statistics',

  // Classical ML
  'logistic-regression': 'Classical ML',
  'linear-regression': 'Classical ML',
  'decision-trees': 'Classical ML',
  'random-forests': 'Classical ML',
  'gradient-boosting': 'Classical ML',
  'svm-and-kernels': 'Classical ML',
  'naive-bayes': 'Classical ML',
  'k-means-clustering': 'Classical ML',
  'dbscan': 'Classical ML',
  'kernel-methods-and-the-kernel-trick': 'Classical ML',
  'tsne-and-umap': 'Classical ML',

  // Deep Learning Foundations
  'activation-functions': 'Deep Learning Foundations',
  'universal-approximation-theorem': 'Deep Learning Foundations',
  'backpropagation': 'Deep Learning Foundations',
  'exploding-vanishing-gradients': 'Deep Learning Foundations',
  'residual-connections': 'Deep Learning Foundations',
  'attention-mechanism': 'Deep Learning Foundations',
  'encoder-decoder-architectures': 'Deep Learning Foundations',
  'autoregressive-vs-diffusion': 'Deep Learning Foundations',
  'graph-neural-networks': 'Deep Learning Foundations',

  // Generative Models
  'variational-autoencoders': 'Generative Models',
  'normalizing-flows': 'Generative Models',
  'gans-overview': 'Generative Models',
  'diffusion-models': 'Generative Models',
  'discrete-gradient-estimators': 'Generative Models',

  // Probabilistic Models
  'expectation-maximization': 'Probabilistic Models',
  'hidden-markov-models': 'Probabilistic Models',
  'gaussian-mixture-models': 'Probabilistic Models',
  'graphical-models': 'Probabilistic Models',
  'belief-propagation': 'Probabilistic Models',
  'factor-analysis-and-ppca': 'Probabilistic Models',
  'forward-backward-and-viterbi': 'Probabilistic Models',
  'gaussian-processes': 'Probabilistic Models',

  // Reinforcement Learning
  'q-learning': 'Reinforcement Learning',
  'policy-gradient': 'Reinforcement Learning',
  'ppo': 'Reinforcement Learning',
  'value-vs-policy-rl': 'Reinforcement Learning',
  'actor-critic-methods': 'Reinforcement Learning',
  'advantage-estimation-and-gae': 'Reinforcement Learning',
  'exploration-vs-exploitation': 'Reinforcement Learning',
  'contextual-bandits': 'Reinforcement Learning',
  'reward-shaping': 'Reinforcement Learning',
  'multi-agent-reinforcement-learning': 'Reinforcement Learning',
  'rl-environments-and-graders': 'Reinforcement Learning',
  'robotics-policy-learning': 'Reinforcement Learning',

  // Computer Vision
  'cnn-architecture': 'Computer Vision',
  'resnet': 'Computer Vision',
  'vision-transformers': 'Computer Vision',
  'object-detection-overview': 'Computer Vision',
  'anchor-boxes-and-nms': 'Computer Vision',
  'convolution-as-matmul': 'Computer Vision',
  'semantic-segmentation': 'Computer Vision',
  'domain-adaptation': 'Computer Vision',
  'adversarial-robustness': 'Computer Vision',
  'multimodal-foundation-models': 'Computer Vision',

  // NLP & Speech
  'automatic-speech-recognition': 'NLP & Speech',
  'bert-and-masked-language-modeling': 'NLP & Speech',
  'conditional-random-fields': 'NLP & Speech',
  'connectionist-temporal-classification': 'NLP & Speech',
  'lstm-and-gru': 'NLP & Speech',
  'rnn-transducer': 'NLP & Speech',
  'word-embeddings': 'NLP & Speech',
  'streaming-asr': 'NLP & Speech',
  'speaker-recognition': 'NLP & Speech',
  'hybrid-vs-end-to-end-asr': 'NLP & Speech',

  // Retrieval & Recommenders
  'alternating-least-squares': 'Retrieval & Recommenders',
  'approximate-nearest-neighbors': 'Retrieval & Recommenders',
  'content-based-filtering': 'Retrieval & Recommenders',
  'embedding-spaces-and-similarity': 'Retrieval & Recommenders',
  'factorization-machines': 'Retrieval & Recommenders',
  'knowledge-graph-embeddings': 'Retrieval & Recommenders',
  'matrix-factorization-recsys': 'Retrieval & Recommenders',
  'tf-idf-and-bm25': 'Retrieval & Recommenders',
  'two-tower-retrieval': 'Retrieval & Recommenders',

  // LLM Internals
  'flashattention': 'LLM Internals',
  'kv-cache': 'LLM Internals',
  'speculative-decoding': 'LLM Internals',
  'quantization': 'LLM Internals',
  'rlhf-and-dpo': 'LLM Internals',
  'rag-overview': 'LLM Internals',
  'tokenization': 'LLM Internals',
  'positional-encoding': 'LLM Internals',
  'rotary-position-embeddings': 'LLM Internals',
  'transformer-architecture': 'LLM Internals',
  'grouped-query-attention': 'LLM Internals',
  'sparse-attention': 'LLM Internals',
  'linear-attention': 'LLM Internals',
  'mixture-of-experts': 'LLM Internals',
  'paged-attention': 'LLM Internals',
  'continuous-batching': 'LLM Internals',
  'long-context-llms': 'LLM Internals',
  'prefill-vs-decode': 'LLM Internals',
  'decoding-strategies': 'LLM Internals',
  'multi-head-attention': 'LLM Internals',
  'self-attention-vs-cross-attention': 'LLM Internals',
  'mechanistic-interpretability': 'LLM Internals',
  'chain-of-thought-monitorability': 'LLM Internals',
  'scalable-oversight-and-ai-control': 'LLM Internals',
  'model-organisms-of-misalignment': 'LLM Internals',
  'llm-security-threat-models': 'LLM Internals',
  'preference-data-and-reward-models': 'LLM Internals',

  // Training Fundamentals
  'cross-entropy-softmax': 'Training Fundamentals',
  'adam-and-adamw': 'Training Fundamentals',
  'regularization': 'Training Fundamentals',
  'batchnorm-vs-layernorm': 'Training Fundamentals',
  'calibration': 'Training Fundamentals',
  'mixed-precision-training': 'Training Fundamentals',
  'learning-rate-schedules': 'Training Fundamentals',
  'gradient-clipping': 'Training Fundamentals',
  'weight-initialization': 'Training Fundamentals',
  'dropout': 'Training Fundamentals',
  'label-smoothing': 'Training Fundamentals',
  'sgd-with-momentum': 'Training Fundamentals',
  'activation-checkpointing': 'Training Fundamentals',
  'gradient-accumulation': 'Training Fundamentals',
  'weight-decay-vs-l2': 'Training Fundamentals',
  'mixup-and-cutmix': 'Training Fundamentals',
  'microannealing': 'Training Fundamentals',
  'neural-network-training-recipe': 'Training Fundamentals',
  'wsd-and-wsd-s': 'Training Fundamentals',
  'z-loss': 'Training Fundamentals',
  'foundation-model-data-curation': 'Training Fundamentals',
  'loss-spikes-at-scale': 'Training Fundamentals',

  // Systems & Infrastructure
  'gpu-memory-hierarchy': 'Systems & Infrastructure',
  'fsdp-and-zero': 'Systems & Infrastructure',
  'sequence-packing': 'Systems & Infrastructure',
  'floating-point-formats': 'Systems & Infrastructure',
  'tensor-parallelism': 'Systems & Infrastructure',
  'pipeline-parallelism': 'Systems & Infrastructure',
  'all-reduce-and-collectives': 'Systems & Infrastructure',
  'knowledge-distillation': 'Systems & Infrastructure',
  'pruning': 'Systems & Infrastructure',
  'fault-tolerant-collectives': 'Systems & Infrastructure',

  // ML Systems & Evaluation
  'ab-testing-for-ml': 'ML Systems & Evaluation',
  'cross-validation-strategies': 'ML Systems & Evaluation',
  'roc-pr-auc': 'ML Systems & Evaluation',
  'perplexity-and-bits-per-token': 'ML Systems & Evaluation',
  'precision-recall-f1': 'ML Systems & Evaluation',
  'ranking-metrics-ndcg-map-mrr': 'ML Systems & Evaluation',
  'confusion-matrix-and-classification-metrics': 'ML Systems & Evaluation',
  'expected-calibration-error': 'ML Systems & Evaluation',
  'model-interpretability': 'ML Systems & Evaluation',
};

export const REFERENCE_ORDER = [
  'Linear Algebra & Math',
  'Probability & Statistics',
  'Classical ML',
  'Deep Learning Foundations',
  'Generative Models',
  'Probabilistic Models',
  'Reinforcement Learning',
  'Computer Vision',
  'NLP & Speech',
  'Retrieval & Recommenders',
  'LLM Internals',
  'Training Fundamentals',
  'Systems & Infrastructure',
  'ML Systems & Evaluation',
];

export type Category = 'guides' | 'questions' | 'concepts';

export function getSubcategoryMap(category: Category): { map: Record<string, string>; order: string[] } | null {
  if (category === 'questions') return { map: INTERVIEW_SUBCATEGORY, order: INTERVIEW_ORDER };
  if (category === 'concepts') return { map: REFERENCE_SUBCATEGORY, order: REFERENCE_ORDER };
  return null; // essays stay flat (small + naturally chronological)
}
