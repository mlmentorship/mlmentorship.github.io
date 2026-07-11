# Frontier ML implementation set

These five exercises are ML problems in code. They are not a substitute for a general algorithms curriculum.

## Setup

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Use a Python version supported by current PyTorch wheels, typically Python 3.11 to 3.13.

Each starter raises `NotImplementedError` or contains a clearly marked candidate section. Implement one exercise per timed session. Do not read or modify a test merely to weaken its contract.

## Exercises

| File | Time | Core signal |
| --- | --- | --- |
| `decoder.py` | 45 min | Tensor shapes, causal masking, residual structure, stable attention |
| `kv_cache.py` | 45 min | Incremental decoding, cache invariants, equivalence to full attention |
| `beam_search.py` | 40 min | Sequence scores, EOS handling, length normalization, bounded state |
| `lora.py` | 35 min | Low-rank parameterization, frozen base weights, scaling, initialization |
| `autograd.py` | 50 min | Computation graph, local derivatives, topological reverse pass, accumulation |

## Graduation rule

A passing implementation must:

1. satisfy the public tests;
2. include at least two candidate-written edge tests;
3. state time and memory complexity;
4. explain one production difference from the toy implementation;
5. survive one changed constraint without a rewrite.

Question pages:

- https://mlmentorship.com/questions/implement-transformer-decoder/
- https://mlmentorship.com/questions/implement-kv-cache-decode/
- https://mlmentorship.com/questions/implement-beam-search/
- https://mlmentorship.com/questions/implement-lora-adapter/
- https://mlmentorship.com/questions/implement-reverse-mode-autograd/
