# Hands-on Transformer

An educational implementation of a small Transformer architecture using PyTorch.

## Project goal

Build a compact, CPU-friendly Transformer whose internal components are implemented explicitly and can be modified in experiments.

The first task is binary sentiment classification on movie reviews (IMDB). Later directions include machine translation and biological sequence classification.

## Current implementation

The first PyTorch model module is in `src/model.py`. It contains:

- token embeddings and learned positional embeddings;
- multi-head self-attention with a padding mask;
- feed-forward layers;
- residual connections and LayerNorm;
- masked mean pooling and a classification head.

Basic shape and padding checks are in `tests/test_model.py`.

## Repository structure

- `src/` — the main PyTorch implementation;
- `tests/` — basic tests for model behavior;
- `notebooks/` are not required: numbered learning notebooks are kept in the repository root;
- `sandbox/` — the original JAX prototype supplied by the instructor. It is preserved separately and is not the main implementation.

## Project dependencies

Install the packages from the repository root:

```bash
pip install -r requirements.txt
```

Run the model checks:

```bash
python -m unittest discover -s tests
```

## Planned experiments

- compare model width, depth, number of attention heads and sequence length;
- measure accuracy, F1, parameter count, training time and throughput;
- compare the Transformer with a TF-IDF + Logistic Regression baseline.

## References

- [nanoGPT — Andrej Karpathy](https://github.com/karpathy/nanoGPT)
- [PyTorch](https://pytorch.org/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
