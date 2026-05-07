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
  'implement-attention-from-scratch': 'LLM Systems',
  'walk-through-speculative-decoding': 'LLM Systems',
  'rag-for-legal-docs': 'LLM Systems',
  'evaluate-an-agent': 'LLM Systems',
  'evals-for-coding-assistant': 'LLM Systems',
  'ab-test-chatbot': 'LLM Systems',
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

  'decide-what-to-work-on': 'Behavioral',
  'disagreed-with-senior': 'Behavioral',
  'most-overrated-technique': 'Behavioral',
  'scope-ambiguous-problem': 'Behavioral',
  'most-ambitious-project': 'Behavioral',

  'derive-logistic-regression': 'Math',
  'softmax-cross-entropy-pairing': 'Math',
  'reparameterization-trick': 'Math',

  'implement-knn': 'Coding',
  'debug-training-loop': 'Coding',
};

export const INTERVIEW_ORDER = [
  'ML Fundamentals',
  'Deep Learning Production',
  'LLM Systems',
  'Recsys & Search',
  'ML System Design',
  'Behavioral',
  'Math',
  'Coding',
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
  'matrix-factorization-recsys': 'Classical ML',

  // Deep Learning Foundations
  'activation-functions': 'Deep Learning Foundations',
  'universal-approximation-theorem': 'Deep Learning Foundations',
  'backpropagation': 'Deep Learning Foundations',
  'exploding-vanishing-gradients': 'Deep Learning Foundations',
  'residual-connections': 'Deep Learning Foundations',
  'attention-mechanism': 'Deep Learning Foundations',
  'encoder-decoder-architectures': 'Deep Learning Foundations',
  'autoregressive-vs-diffusion': 'Deep Learning Foundations',

  // Generative Models
  'variational-autoencoders': 'Generative Models',
  'normalizing-flows': 'Generative Models',
  'gans-overview': 'Generative Models',
  'diffusion-models': 'Generative Models',

  // Probabilistic Models
  'expectation-maximization': 'Probabilistic Models',
  'hidden-markov-models': 'Probabilistic Models',
  'gaussian-mixture-models': 'Probabilistic Models',
  'graphical-models': 'Probabilistic Models',

  // Reinforcement Learning
  'q-learning': 'Reinforcement Learning',
  'policy-gradient': 'Reinforcement Learning',
  'ppo': 'Reinforcement Learning',
  'value-vs-policy-rl': 'Reinforcement Learning',

  // Computer Vision
  'cnn-architecture': 'Computer Vision',
  'resnet': 'Computer Vision',
  'vision-transformers': 'Computer Vision',
  'object-detection-overview': 'Computer Vision',

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

  // Systems & Infrastructure
  'gpu-memory-hierarchy': 'Systems & Infrastructure',
  'fsdp-and-zero': 'Systems & Infrastructure',
  'sequence-packing': 'Systems & Infrastructure',
  'floating-point-formats': 'Systems & Infrastructure',
  'tensor-parallelism': 'Systems & Infrastructure',
  'pipeline-parallelism': 'Systems & Infrastructure',
  'all-reduce-and-collectives': 'Systems & Infrastructure',

  // ML Systems & Evaluation
  'ab-testing-for-ml': 'ML Systems & Evaluation',
  'embedding-spaces-and-similarity': 'ML Systems & Evaluation',
  'two-tower-retrieval': 'ML Systems & Evaluation',
  'cross-validation-strategies': 'ML Systems & Evaluation',
  'roc-pr-auc': 'ML Systems & Evaluation',
  'perplexity-and-bits-per-token': 'ML Systems & Evaluation',
  'precision-recall-f1': 'ML Systems & Evaluation',
  'ranking-metrics-ndcg-map-mrr': 'ML Systems & Evaluation',
  'confusion-matrix-and-classification-metrics': 'ML Systems & Evaluation',
  'expected-calibration-error': 'ML Systems & Evaluation',
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
  'LLM Internals',
  'Training Fundamentals',
  'Systems & Infrastructure',
  'ML Systems & Evaluation',
];

export type Category = 'essays' | 'interviews' | 'reference';

export function getSubcategoryMap(category: Category): { map: Record<string, string>; order: string[] } | null {
  if (category === 'interviews') return { map: INTERVIEW_SUBCATEGORY, order: INTERVIEW_ORDER };
  if (category === 'reference') return { map: REFERENCE_SUBCATEGORY, order: REFERENCE_ORDER };
  return null; // essays stay flat (small + naturally chronological)
}
